FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install dependencies
RUN pip install torchvision --no-cache-dir

# Copy training script
COPY train_cifar10.py /workspace/train_cifar10.py

# Set working directory
WORKDIR /workspace

CMD ["python", "train_cifar10.py"]
