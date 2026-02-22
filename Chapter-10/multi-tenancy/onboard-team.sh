#!/bin/bash
# onboard-team.sh - Automated team onboarding for Kubeflow

set -e

TEAM_NAME=$1
OWNER_EMAIL=$2
CPU_QUOTA=$3
MEMORY_QUOTA=$4
GPU_QUOTA=$5

if [ -z "$TEAM_NAME" ] || [ -z "$OWNER_EMAIL" ]; then
    echo "Usage: onboard-team.sh <team-name> <owner-email> <cpu-quota> <memory-quota> <gpu-quota>"
    echo "Example: onboard-team.sh team-ml-fraud jane@example.com 64 256Gi 8"
    exit 1
fi

echo "Onboarding team: $TEAM_NAME"
echo "Owner: $OWNER_EMAIL"
echo "Quotas: CPU=$CPU_QUOTA, Memory=$MEMORY_QUOTA, GPU=$GPU_QUOTA"

# Create Profile
cat <<EOF | kubectl apply -f -
apiVersion: kubeflow.org/v1
kind: Profile
metadata:
  name: $TEAM_NAME
spec:
  owner:
    kind: User
    name: $OWNER_EMAIL
  resourceQuotaSpec:
    hard:
      requests.cpu: "$CPU_QUOTA"
      requests.memory: "$MEMORY_QUOTA"
      requests.nvidia.com/gpu: "$GPU_QUOTA"
      persistentvolumeclaims: "20"
      requests.storage: "500Gi"
      pods: "100"
EOF

echo "Profile created"

# Wait for namespace to be created
echo "Waiting for namespace to be ready..."
kubectl wait --for=condition=Ready --timeout=60s profile/$TEAM_NAME

NAMESPACE="kubeflow-user-$(echo $OWNER_EMAIL | sed 's/@/-at-/g' | sed 's/\./-/g')"

# Create team's shared storage
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: team-shared-storage
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: nfs-storage
  resources:
    requests:
      storage: 100Gi
EOF

echo "Shared storage created"

# Create S3 ServiceAccount for team
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ${TEAM_NAME}-s3-sa
  namespace: $NAMESPACE
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/${TEAM_NAME}-s3-access
EOF

echo "ServiceAccount created"

# Output onboarding summary
cat <<EOF

Team onboarded successfully!

Team Name: $TEAM_NAME
Namespace: $NAMESPACE
Owner: $OWNER_EMAIL

Resource Quotas:
  CPU: $CPU_QUOTA cores
  Memory: $MEMORY_QUOTA
  GPUs: $GPU_QUOTA
  Storage: 500Gi
  PVCs: 20
  Pods: 100

Next steps:
1. The team owner should log into Kubeflow at https://kubeflow.example.com
2. Create their first notebook or pipeline in the $NAMESPACE namespace
3. For S3 access, use ServiceAccount: ${TEAM_NAME}-s3-sa

EOF
