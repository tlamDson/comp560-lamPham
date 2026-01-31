"""
Diagnostic script to check CUDA availability and PyTorch installation.
"""
import sys
import platform

print("=" * 60)
print("CUDA AVAILABILITY CHECK")
print("=" * 60)

print(f"Platform: {platform.system()} {platform.release()}")
print(f"Python: {sys.version.split()[0]}")

try:
    import torch
    print(f"✓ PyTorch installed: {torch.__version__}")
except ImportError:
    print("✗ PyTorch not installed")
    sys.exit(1)

print(f"\nCUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ CUDA is available!")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Device count: {torch.cuda.device_count()}")
    print(f"  Device name: {torch.cuda.get_device_name(0)}")
    print(f"  BF16 supported: {torch.cuda.is_bf16_supported()}")
    
    # Check Triton (required for torch.compile)
    print(f"\nTriton (for torch.compile):")
    try:
        import triton
        print(f"  ✓ Triton installed: {triton.__version__}")
        compile_available = True
    except ImportError:
        print(f"  ✗ Triton not installed")
        print(f"    Note: torch.compile disabled on Windows due to limited Triton support")
        compile_available = False
else:
    print("✗ CUDA is NOT available")
    print("\nYour PyTorch installation is CPU-only.")
    print("\n" + "=" * 60)
    print("FIX: Install CUDA-enabled PyTorch")
    print("=" * 60)
    print("\n1. Uninstall current PyTorch:")
    print("   pip uninstall torch torchvision torchaudio")
    print("\n2. Install CUDA version (for RTX 4060):")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("\n   (This installs PyTorch with CUDA 12.1 support)")
    print("\n3. Re-run this script to verify:")
    print("   python check_cuda.py")
    print("\n" + "=" * 60)
    
print("\n" + "=" * 60)
print("CONFIGURATION AUTO-DETECT")
print("=" * 60)
print("\nYour config will auto-detect and use:")
if torch.cuda.is_available():
    print("  Device: CUDA")
    print("  Dtype: bfloat16")
    is_windows = sys.platform == 'win32'
    if is_windows:
        print("  Compile: False (disabled on Windows)")
        print("\n✓ GPU mode enabled (BF16 speedup)")
        print("  Expected training time: ~1-2 minutes")
        print("  Note: torch.compile disabled due to Triton compatibility on Windows")
    else:
        try:
            import triton
            print("  Compile: True")
            print("\n✓ Full speedrun mode enabled!")
            print("  Expected training time: <1 minute")
        except ImportError:
            print("  Compile: False (Triton not available)")
            print("\n✓ GPU mode enabled (no compile)")
            print("  Expected training time: ~1-2 minutes")
else:
    print("  Device: CPU")
    print("  Dtype: float32")
    print("  Compile: False")
    print("\n⚠ Running in CPU fallback mode (slower)")
    print("  Expected training time: ~6 minutes")
