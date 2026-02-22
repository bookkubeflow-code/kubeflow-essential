#!/bin/bash
# Install PyTorch for local development
# Choose the appropriate section based on your hardware

echo "Select your platform:"
echo "1) CPU-only (testing and development)"
echo "2) NVIDIA GPU - CUDA 12.1"
echo "3) NVIDIA GPU - CUDA 11.8 (older systems)"
echo "4) NVIDIA GPU - CUDA 12.4 (latest)"
echo "5) Apple Silicon (M1/M2/M3 Macs)"
read -p "Choice [1-5]: " choice

# Create virtual environment (recommended)
python -m venv pytorch-env
source pytorch-env/bin/activate  # On macOS/Linux
# pytorch-env\Scripts\activate   # On Windows

case $choice in
  1)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ;;
  2)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ;;
  3)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ;;
  4)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ;;
  5)
    # Native Apple Silicon support with MPS backend
    pip install torch torchvision torchaudio
    ;;
  *)
    echo "Invalid choice"
    exit 1
    ;;
esac

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Verify torchrun is available
torchrun --help > /dev/null 2>&1 && echo "torchrun is available" || echo "torchrun not found"
