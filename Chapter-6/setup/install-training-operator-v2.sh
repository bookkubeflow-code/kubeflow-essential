#!/bin/bash
# Install Training Operator v2 (Kubeflow Trainer)
# For PyTorch, DeepSpeed, MLX, and LLM fine-tuning

# Install Kubeflow Trainer v2 control plane
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref=v2.0.0"

# Verify controller manager and JobSet controller are running
echo "=== Controller pods ==="
kubectl get pods -n kubeflow-system
# Should show:
# jobset-controller-manager-xxx
# kubeflow-trainer-controller-manager-xxx

# Install pre-configured Training Runtimes
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/runtimes?ref=v2.0.0"

# Verify runtimes are available
echo ""
echo "=== Cluster Training Runtimes ==="
kubectl get clustertrainingruntimes
# Should show runtimes like:
# NAME                    AGE
# deepspeed-distributed   12s
# mlx-distributed         12s
# mpi-distributed         12s
# torch-distributed       12s
# torchtune-llama3.2-1b   12s
# torchtune-llama3.2-3b   12s
