#!/bin/bash
# Create namespace for training jobs (if not using Kubeflow profiles)

kubectl create namespace kubeflow-user-example-com

echo "Namespace created. Verify with:"
echo "  kubectl get namespaces"
