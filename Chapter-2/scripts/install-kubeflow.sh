#!/bin/bash
# Install Kubeflow using the manifests approach (single-command installation)

# Clone the Kubeflow manifests repository
git clone https://github.com/kubeflow/manifests.git
cd manifests

# Install all Kubeflow components
# This command may run for several minutes and might need multiple attempts
while ! kustomize build example | kubectl apply --server-side --force-conflicts -f -; do
    echo "Retrying to apply resources"
    sleep 20
done

echo "Kubeflow installation complete!"
echo "Verify with: kubectl -n kubeflow get deployments"
