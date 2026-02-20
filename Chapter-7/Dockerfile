# =============================================================================
# Katib Training Container
# =============================================================================
# This Dockerfile creates a lightweight container for running scikit-learn
# based hyperparameter tuning experiments with Katib.
#
# Build:
#   docker build -t your-registry/katib-sklearn-example:v1.0 .
#
# Test locally:
#   docker run your-registry/katib-sklearn-example:v1.0 \
#       --n-estimators 100 --max-depth 10 --criterion gini
# =============================================================================

# Use official Python slim image for smaller container size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
# - PYTHONUNBUFFERED: Ensure Python output is sent straight to terminal
#   without being buffered first (critical for Katib metrics collection)
# - PYTHONDONTWRITEBYTECODE: Prevent Python from writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
# - No additional system packages needed for scikit-learn
# - For GPU support, you would add CUDA libraries here
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # Basic utilities (optional, helpful for debugging)
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
# Changes to source code won't invalidate the dependency layer
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY src/train.py .

# Create a non-root user for security
# Kubernetes best practice: don't run containers as root
RUN useradd -m -u 1000 trainer && \
    chown -R trainer:trainer /app

# Switch to non-root user
USER trainer

# Set the entrypoint to the training script
# Katib will pass hyperparameters as command-line arguments
ENTRYPOINT ["python", "train.py"]

# Default arguments (used when running locally without Katib)
CMD ["--n-estimators", "100", "--max-depth", "10", "--criterion", "gini"]

