#!/bin/bash
# Port-forward to access the Kubeflow dashboard
# Access at http://localhost:8080
# Default credentials: user@example.com / 12341234

echo "Starting port-forward to Kubeflow dashboard..."
echo "Access at: http://localhost:8080"
echo "Default credentials:"
echo "  Email: user@example.com"
echo "  Password: 12341234"
echo ""
echo "Press Ctrl+C to stop"

kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
