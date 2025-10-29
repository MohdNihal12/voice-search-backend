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
    logger.info("✓ Model loaded successfully on CUDA")
except Exception as e:
    logger.warning(f"⚠️  Failed to load on CUDA: {e}, falling back to CPU")
    model = whisper.load_model(MODEL_SIZE, device="cpu")

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
    logger.info(f"📝 Starting transcription for file: {file.filename}")
    
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
        
        logger.info(f"💾 File saved to: {tmp_path}")
        
        # Transcribe (English only)
        result = model.transcribe(
            tmp_path, 
            verbose=False,
            language="en",  # Force English language
            fp16=True  # Use fp16 for faster processing on GPU
        )
        
        text = result.get("text", "").strip()
        segments = result.get("segments", [])
        language = result.get("language", "en")
        print("\n" + "="*80)
        print("TRANSCRIPTION:")
        print("="*80)
        print(text)
        print("="*80 + "\n")
        
        logger.info(f"✅ Transcription completed: {len(text)} chars")
        
        return JSONResponse({
            "text": text,
            "segments": segments,
            "language": language
        })
        
    except FileNotFoundError as e:
        logger.error(f"❌ ffmpeg not found: {e}")
        raise HTTPException(
            status_code=500,
            detail="ffmpeg not found. Please install ffmpeg and add it to PATH."
        )
    except Exception as e:
        logger.error(f"❌ Transcription error: {e}", exc_info=True)
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
        "device": "cuda" if model.device.type == "cuda" else "cpu"
    }

