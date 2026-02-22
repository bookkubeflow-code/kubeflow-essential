#!/bin/bash
# Fix missing runtime labels for Training Operator v2
# The default runtime manifests from the Kubeflow repository are missing
# required labels that the Training Operator uses to identify and validate runtimes.
# Without these labels, you'll hit cryptic errors when submitting training jobs.
# This is a known issue in the v2.0.0 release.

kubectl label clustertrainingruntime torch-distributed \
  trainer.kubeflow.org/framework=pytorch --overwrite

kubectl label clustertrainingruntime deepspeed-distributed \
  trainer.kubeflow.org/framework=deepspeed --overwrite

kubectl label clustertrainingruntime mlx-distributed \
  trainer.kubeflow.org/framework=mlx --overwrite

kubectl label clustertrainingruntime mpi-distributed \
  trainer.kubeflow.org/framework=mpi --overwrite

kubectl label clustertrainingruntime torchtune-llama3.2-1b \
  trainer.kubeflow.org/framework=torchtune --overwrite

kubectl label clustertrainingruntime torchtune-llama3.2-3b \
  trainer.kubeflow.org/framework=torchtune --overwrite

echo "Runtime labels fixed."
