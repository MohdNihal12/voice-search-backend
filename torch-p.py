import whisper
import torch

print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")

try:
    print("\nLoading Whisper model on CUDA...")
    model = whisper.load_model("medium", device="cuda")
    print(f"✓ Model loaded successfully!")
    print(f"Model device: {model.device}")
    print(f"Model is on CUDA: {next(model.parameters()).is_cuda}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
except Exception as e:
    print(f"✗ Failed to load on CUDA: {e}")
    import traceback
    traceback.print_exc()