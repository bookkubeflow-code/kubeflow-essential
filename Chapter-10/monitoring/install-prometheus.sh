#!/bin/bash
# Install Prometheus and Grafana using Helm

# Add the Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Create monitoring namespace
kubectl create namespace monitoring

# Install kube-prometheus-stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --values prometheus-values.yaml

# Watch pods come up
echo "Watching pods in monitoring namespace..."
kubectl get pods -n monitoring -w

# To access Grafana via port-forward:
# kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
# Open http://localhost:3000 (admin / <password from values file>)

# To install NVIDIA GPU exporter for GPU monitoring:
# kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-monitoring-tools/master/exporters/prometheus-dcgm/dcgm-exporter.yaml
