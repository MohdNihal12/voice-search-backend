from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
import shutil
import os
import logging
import time
import gc
import numpy as np
import subprocess
import warnings

# Suppress librosa PySoundFile warnings (we're using audioread for WebM)
warnings.filterwarnings('ignore', message='PySoundFile failed')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')

# ------------------------------
# Setup logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------
# Check for audio processing libraries
# ------------------------------
try:
    import librosa
    LIBROSA_AVAILABLE = True
    logger.info("Using librosa for audio processing")
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available (install with: pip install librosa), audio energy check disabled")

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Load Whisper model
# ------------------------------
MODEL_SIZE = "medium"
logger.info(f"Loading Whisper model: {MODEL_SIZE} on CUDA")
try:
    model = whisper.load_model(MODEL_SIZE, device="cuda")
    logger.info("Model loaded successfully on CUDA")
except Exception as e:
    logger.warning(f"Failed to load on CUDA: {e}, falling back to CPU")
    model = whisper.load_model(MODEL_SIZE, device="cpu")

# ------------------------------
# Hallucination Detection
# ------------------------------
HALLUCINATION_PHRASES = [
    "thanks for watching",
    "thank you for watching",
    "thank you",
    "please subscribe",
    "like and subscribe",
    "don't forget to subscribe",
    "hit the bell",
    "see you next time",
    "catch you later",
    "stay tuned"
]

def is_hallucination(text: str) -> bool:
    """Check if transcription is likely a hallucination"""
    text_lower = text.lower().strip()
    
    # Empty or very short
    if len(text_lower) < 2:
        return True
    
    # Common hallucination phrases
    for phrase in HALLUCINATION_PHRASES:
        if phrase in text_lower:
            return True
    
    return False

def convert_to_wav(input_path: str) -> str:
    """Convert audio file to WAV using ffmpeg (for librosa compatibility)"""
    output_path = input_path.replace(os.path.splitext(input_path)[1], "_converted.wav")
    
    try:
        # Use ffmpeg to convert to 16kHz mono WAV
        subprocess.run([
            'ffmpeg', '-i', input_path,
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',      # Mono
            '-y',            # Overwrite output
            output_path
        ], check=True, capture_output=True)
        
        logger.info(f"Converted {input_path} to WAV format")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg conversion failed: {e.stderr.decode()}")
        raise
    except FileNotFoundError:
        logger.error("ffmpeg not found in PATH")
        raise

def has_speech_improved(audio_path: str) -> dict:
    """
    Enhanced speech detection with multiple checks using librosa.
    Returns dict with 'has_speech' bool and 'reason' string.
    """
    if not LIBROSA_AVAILABLE:
        return {"has_speech": True, "reason": "librosa not available"}
    
    wav_path = None
    try:
        # Convert WebM/other formats to WAV first (librosa handles WAV better)
        if not audio_path.lower().endswith('.wav'):
            wav_path = convert_to_wav(audio_path)
            audio_path = wav_path
        
        # Load audio with librosa (force audioread backend for WebM support)
        audio, sr = librosa.load(audio_path, sr=16000, mono=True, res_type='kaiser_best')
        
        # Normalize audio to boost quiet recordings
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.8  # Normalize to 80% of max amplitude
            logger.info(f"Normalized audio (original peak: {peak:.6f})")
        
        # Check 1: File duration
        duration = len(audio) / sr
        if duration < 0.1:  # Less than 100ms
            return {"has_speech": False, "reason": "audio too short"}
        
        # Check 2: RMS Energy (more lenient threshold)
        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.005:  # LOWERED from 0.01 - less strict
            return {"has_speech": False, "reason": f"low energy (RMS: {rms:.6f})"}
        
        # Check 3: Zero Crossing Rate (voice has moderate ZCR)
        zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        if zcr < 0.005 or zcr > 0.95:  # RELAXED from 0.01/0.9
            return {"has_speech": False, "reason": f"abnormal ZCR: {zcr:.4f}"}
        
        # Check 4: Peak amplitude (more lenient)
        peak = np.max(np.abs(audio))
        if peak < 0.005:  # LOWERED from 0.01
            return {"has_speech": False, "reason": f"low peak: {peak:.6f}"}
        
        logger.info(f"Audio stats - RMS: {rms:.4f}, ZCR: {zcr:.4f}, Peak: {peak:.4f}, Duration: {duration:.2f}s")
        return {"has_speech": True, "reason": "passed all checks"}
        
    except Exception as e:
        logger.warning(f"Audio analysis failed: {e}")
        return {"has_speech": True, "reason": "analysis failed"}
    finally:
        # Clean up converted WAV file
        if wav_path and os.path.exists(wav_path):
            safe_delete(wav_path)

# ------------------------------
# Safe file deletion with retries
# ------------------------------
def safe_delete(filepath: str, max_retries: int = 3, delay: float = 0.5):
    """Safely delete a file with retries for Windows file locking issues"""
    for attempt in range(max_retries):
        try:
            if os.path.exists(filepath):
                os.unlink(filepath)
                logger.info(f"✓ Cleaned up temp file: {filepath}")
                return True
        except PermissionError:
            if attempt < max_retries - 1:
                logger.debug(f"File locked, retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                logger.warning(f"Could not delete temp file after {max_retries} attempts: {filepath}")
                return False
        except Exception as e:
            logger.warning(f"Error deleting temp file: {e}")
            return False
    return False

# ------------------------------
# Transcription endpoint
# ------------------------------
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    logger.info(f"Starting transcription for file: {file.filename}")
    
    # Create temp file and close it immediately to avoid locking
    tmp = tempfile.NamedTemporaryFile(
        delete=False, 
        suffix=os.path.splitext(file.filename)[1] or ".webm"
    )
    tmp_path = tmp.name
    tmp.close()  # Close immediately to release file handle
    
    try:
        # Save uploaded file
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"File saved to: {tmp_path}")
        
        # Enhanced speech detection (converts to WAV internally if needed)
        speech_check = has_speech_improved(tmp_path)
        if not speech_check["has_speech"]:
            logger.warning(f"No speech detected: {speech_check['reason']}")
            return JSONResponse({
                "ok": True,
                "text": "",
                "segments": [],
                "language": "en",
                "warning": f"No speech detected: {speech_check['reason']}"
            }, status_code=200)
        
        # Transcribe with stronger anti-hallucination settings
        result = model.transcribe(
            tmp_path, 
            verbose=False,
            language="en",
            fp16=True,
            condition_on_previous_text=False,  # Prevent context carryover
            no_speech_threshold=0.6,  # INCREASED from 0.6 - more aggressive silence detection
            logprob_threshold=-1.0,  # STRICTER (was -1.0) - filter low-confidence segments
            compression_ratio_threshold=2.4,  # STRICTER (was 2.4) - detect repetitive text
            temperature=0.0,  # More deterministic transcription
            beam_size=1,  # Faster, less creative
        )
        
        text = result.get("text", "").strip()
        segments = result.get("segments", [])
        language = result.get("language", "en")
        
        # Filter segments with low confidence (high no_speech_prob)
        filtered_segments = [
            seg for seg in segments 
            if seg.get("no_speech_prob", 0) < 0.8  # Keep only high-confidence speech
        ]
        
        # Rebuild text from filtered segments
        if len(filtered_segments) < len(segments):
            text = " ".join(seg["text"].strip() for seg in filtered_segments)
            logger.info(f"Filtered {len(segments) - len(filtered_segments)} low-confidence segments")
        
        # Check for hallucination
        if is_hallucination(text) or len(text) < 3:
            logger.warning(f"Hallucination detected: '{text}'")
            return JSONResponse({
                "ok": True,
                "text": "",
                "segments": [],
                "language": language,
                "warning": "Possible hallucination detected"
            }, status_code=200)
        
        print("\n" + "="*80)
        print("TRANSCRIPTION:")
        print("="*80)
        print(text)
        print("="*80 + "\n")
        
        logger.info(f"Transcription completed: {len(text)} chars")
        
        return JSONResponse({
            "ok": True,
            "text": text,
            "segments": filtered_segments,
            "language": language
        })
        
    except FileNotFoundError as e:
        logger.error(f"ffmpeg not found: {e}")
        raise HTTPException(
            status_code=500,
            detail="ffmpeg not found. Please install ffmpeg and add it to PATH."
        )
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Force garbage collection to release file handles
        gc.collect()
        
        # Attempt to delete temp file with retries
        safe_delete(tmp_path)

# ------------------------------
# Health check endpoint
# ------------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": MODEL_SIZE,
        "device": "cuda" if model.device.type == "cuda" else "cpu",
        "audio_backend": "librosa+ffmpeg" if LIBROSA_AVAILABLE else "none"
    }