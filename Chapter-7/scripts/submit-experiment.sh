#!/bin/bash
# =============================================================================
# Submit Katib Experiment
# =============================================================================
# Submits an experiment to the Kubernetes cluster and provides initial status.
#
# Usage:
#   ./scripts/submit-experiment.sh <experiment-yaml>
#
# Examples:
#   ./scripts/submit-experiment.sh experiments/01-basic-random-search.yaml
#   ./scripts/submit-experiment.sh experiments/04-hyperband.yaml
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: Missing experiment YAML file${NC}"
    echo ""
    echo "Usage: $0 <experiment-yaml>"
    echo ""
    echo "Examples:"
    echo "  $0 experiments/01-basic-random-search.yaml"
    echo "  $0 experiments/03-bayesian-optimization.yaml"
    exit 1
fi

EXPERIMENT_FILE="$1"

# Validate file exists
if [ ! -f "$EXPERIMENT_FILE" ]; then
    echo -e "${RED}Error: File not found: ${EXPERIMENT_FILE}${NC}"
    exit 1
fi

echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}Submitting Katib Experiment${NC}"
echo -e "${GREEN}=============================================${NC}"
echo ""
echo -e "File: ${YELLOW}${EXPERIMENT_FILE}${NC}"
echo ""

# Extract experiment name and namespace from YAML
EXPERIMENT_NAME=$(grep -E "^\s*name:" "$EXPERIMENT_FILE" | head -1 | awk '{print $2}')
NAMESPACE=$(grep -E "^\s*namespace:" "$EXPERIMENT_FILE" | head -1 | awk '{print $2}')
NAMESPACE="${NAMESPACE:-kubeflow}"

echo -e "Experiment: ${BLUE}${EXPERIMENT_NAME}${NC}"
echo -e "Namespace:  ${BLUE}${NAMESPACE}${NC}"
echo ""

# Check if experiment already exists
if kubectl get experiment "${EXPERIMENT_NAME}" -n "${NAMESPACE}" &> /dev/null; then
    echo -e "${YELLOW}Warning: Experiment '${EXPERIMENT_NAME}' already exists${NC}"
    echo ""
    read -p "Delete and recreate? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting existing experiment..."
        kubectl delete experiment "${EXPERIMENT_NAME}" -n "${NAMESPACE}"
        # Wait for deletion
        echo "Waiting for deletion to complete..."
        kubectl wait --for=delete experiment/"${EXPERIMENT_NAME}" -n "${NAMESPACE}" --timeout=60s 2>/dev/null || true
        echo -e "${GREEN}✓ Deleted${NC}"
    else
        echo "Aborting."
        exit 0
    fi
fi

# Submit the experiment
echo "Submitting experiment..."
kubectl apply -f "$EXPERIMENT_FILE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Experiment submitted successfully${NC}"
else
    echo -e "${RED}✗ Failed to submit experiment${NC}"
    exit 1
fi
echo ""

# Wait for experiment to be created
echo "Waiting for experiment to initialize..."
sleep 2

# Show initial status
echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}Experiment Status${NC}"
echo -e "${GREEN}=============================================${NC}"
echo ""

kubectl get experiment "${EXPERIMENT_NAME}" -n "${NAMESPACE}" -o wide

echo ""
echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}Monitoring Commands${NC}"
echo -e "${GREEN}=============================================${NC}"
echo ""
echo "Watch experiment progress:"
echo -e "  ${BLUE}kubectl get experiment ${EXPERIMENT_NAME} -n ${NAMESPACE} -w${NC}"
echo ""
echo "List trials:"
echo -e "  ${BLUE}kubectl get trials -n ${NAMESPACE} -l katib.kubeflow.org/experiment=${EXPERIMENT_NAME}${NC}"
echo ""
echo "Get detailed status:"
echo -e "  ${BLUE}kubectl describe experiment ${EXPERIMENT_NAME} -n ${NAMESPACE}${NC}"
echo ""
echo "View trial logs:"
echo -e "  ${BLUE}kubectl logs -n ${NAMESPACE} -l katib.kubeflow.org/experiment=${EXPERIMENT_NAME} --tail=50${NC}"
echo ""
echo "Get optimal parameters:"
echo -e "  ${BLUE}./scripts/get-optimal-params.sh ${EXPERIMENT_NAME}${NC}"
echo ""

