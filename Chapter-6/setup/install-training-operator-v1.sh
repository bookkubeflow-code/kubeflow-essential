#!/bin/bash
# Install Training Operator v1 for TensorFlow, JAX, XGBoost workloads

# Verify Kind cluster is running
# To set up a local kubernetes cluster using kind, please refer to
# Chapter 2: Getting started with Kubeflow, under the "create kind clusters" section.
kind get clusters

# Install Training Operator v1.7+
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"

# Verify installation
echo "=== Pods ==="
kubectl get pods -n kubeflow

echo ""
echo "=== CRDs ==="
kubectl get crd | grep kubeflow.org
