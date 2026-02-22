#!/bin/bash
# Create a Kubernetes secret for Docker Hub authentication
# Required for pulling Kubeflow container images during installation

docker login

kubectl create secret generic regcred \
    --from-file=.dockerconfigjson=$HOME/.docker/config.json \
    --type=kubernetes.io/dockerconfigjson
