#!/bin/bash
# Verify that all Kubeflow components are running correctly

echo "=== Kubeflow namespace ==="
kubectl get pods -n kubeflow

echo ""
echo "=== Auth namespace ==="
kubectl get pods -n auth

echo ""
echo "=== Cert-Manager namespace ==="
kubectl get pods -n cert-manager

echo ""
echo "=== Istio-System namespace ==="
kubectl get pods -n istio-system

echo ""
echo "=== Knative-Serving namespace ==="
kubectl get pods -n knative-serving

echo ""
echo "=== Deployments in kubeflow namespace ==="
kubectl -n kubeflow get deployments

echo ""
echo "=== Services in kubeflow namespace ==="
kubectl -n kubeflow get services
