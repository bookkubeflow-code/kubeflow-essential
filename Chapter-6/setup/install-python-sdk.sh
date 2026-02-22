#!/bin/bash
# Install Kubeflow SDK for programmatic job creation (v2 only)

pip install git+https://github.com/kubeflow/sdk.git@main#subdirectory=python

# Verify installation
python -c "from kubeflow.trainer import TrainerClient; print('SDK installed successfully')"
