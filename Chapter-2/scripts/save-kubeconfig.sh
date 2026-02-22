#!/bin/bash
# Extract and set kubeconfig for the Kubeflow Kind cluster

kind get kubeconfig --name kubeflow > /tmp/kubeflow-config
export KUBECONFIG=/tmp/kubeflow-config

echo "KUBECONFIG set to /tmp/kubeflow-config"
echo "Run: export KUBECONFIG=/tmp/kubeflow-config"
