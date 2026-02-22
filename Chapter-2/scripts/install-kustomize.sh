#!/bin/bash
# Install Kustomize standalone (v5.4.3+ required for Kubeflow)

# Download the latest Kustomize release
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash

# Move kustomize to a directory in your PATH
sudo mv kustomize /usr/local/bin/

# Verify the installation
kustomize version
