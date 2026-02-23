# Chapter 6: Training Models at Scale

This chapter covers distributed model training using Kubeflow Training Operators, with complete examples for both PyTorch (v2) and TensorFlow (v1).

## Directory Structure

```
Chapter-6/
├── setup/                # Operator installation scripts
├── v2-pytorch/           # PyTorch CIFAR-10 distributed training (v2)
├── v1-tensorflow/        # TensorFlow MNIST distributed training (v1)
├── examples/             # Reference manifests and examples
└── manifests/            # Namespace setup
```

## Training Operator Versions

| Criteria | v1 | v2 |
|----------|----|----|
| Framework | TensorFlow, JAX, XGBoost | PyTorch, DeepSpeed, MLX |
| API Style | Framework-specific CRDs (TFJob, PyTorchJob) | Unified TrainJob with runtimes |
| Maturity | Stable, production-ready | Active development |
| Best For | TF/JAX production workloads | Modern PyTorch, LLM fine-tuning |

## Setup

### Install Training Operator v1 (TensorFlow/JAX/XGBoost)

```bash
./setup/install-training-operator-v1.sh
```

### Install Training Operator v2 (PyTorch/DeepSpeed/MLX)

```bash
./setup/install-training-operator-v2.sh
./setup/fix-runtime-labels.sh   # Required for v2.0.0
./setup/install-python-sdk.sh
```

### Install PyTorch Locally (for testing)

```bash
./setup/install-pytorch-local.sh
```

### Verify Installation

```bash
./setup/verify-installation.sh
```

## PyTorch Distributed Training (v2)

Complete CIFAR-10 image classification using PyTorch DDP.

### Test Locally First

```bash
cd v2-pytorch
mkdir -p checkpoints data
python train_cifar10.py --batch-size 64 --epochs 2 --lr 0.001 --checkpoint-dir ./checkpoints
```

Multi-GPU local test:
```bash
torchrun --nproc_per_node=2 train_cifar10.py --batch-size 64 --epochs 2
```

### Deploy to Kubeflow

**Method 1: Python SDK**
```bash
# Build and push container
docker build -t <your-registry>/cifar10-training:v1 .
docker push <your-registry>/cifar10-training:v1

# Submit job
python submit_training_v2.py
```

**Method 2: YAML Manifest**
```bash
kubectl apply -f storage.yaml
kubectl apply -f cifar10-trainjob.yaml
kubectl get trainjobs -n kubeflow-user-example-com -w
```

### Monitor Training

```bash
kubectl logs -n kubeflow-user-example-com \
  -l trainer.kubeflow.org/trainjob-name=cifar10-distributed-training,trainer.kubeflow.org/replica-index=0 \
  -f
```

## TensorFlow Distributed Training (v1)

Complete MNIST classification using MultiWorkerMirroredStrategy.

### Deploy to Kubeflow

```bash
cd v1-tensorflow

# Create ConfigMap with training script
kubectl create configmap mnist-tf-training-code \
  --from-file=train_mnist_tf.py \
  -n kubeflow-user-example-com

# Apply storage and TFJob
kubectl apply -f tf-storage.yaml
kubectl apply -f mnist-tfjob.yaml
```

### Monitor with TensorBoard

```bash
kubectl apply -f tensorboard-deployment.yaml
kubectl port-forward -n kubeflow-user-example-com svc/tensorboard-mnist 6006:6006
# Open http://localhost:6006
```

## Examples

| File | Description |
|------|-------------|
| `v1-pytorchjob-example.yaml` | v1 PyTorchJob CRD structure |
| `v2-trainjob-example.yaml` | v2 TrainJob with runtime reference |
| `cluster-training-runtime.yaml` | ClusterTrainingRuntime template |
| `gang-scheduling.yaml` | All-or-nothing pod scheduling |
| `fault-tolerance.yaml` | Failure policy configuration |
| `deepspeed-training.py` | DeepSpeed for billion-parameter models |
| `tfjob-sdk-example.py` | TFJob submission via Python SDK |
| `test-v1-tfjob.yaml` | Quick v1 installation test |
| `test-v2-trainjob.py` | Quick v2 installation test |

## Troubleshooting

- **MacOS DataLoader crash**: Set `num_workers=0` in DataLoader (see chapter text)
- **Runtime labels missing**: Run `./setup/fix-runtime-labels.sh`
- **SDK import fails**: Reinstall with `pip install git+https://github.com/kubeflow/sdk.git@main`
- **torchrun not found**: Ensure PyTorch is installed in your active environment
