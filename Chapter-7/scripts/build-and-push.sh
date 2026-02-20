#!/bin/bash
# =============================================================================
# Build and Push Docker Image
# =============================================================================
# This script builds the training container and pushes it to a registry.
#
# Usage:
#   ./scripts/build-and-push.sh [REGISTRY] [TAG]
#
# Examples:
#   ./scripts/build-and-push.sh                           # Uses defaults
#   ./scripts/build-and-push.sh docker.io/myuser         # Custom registry
#   ./scripts/build-and-push.sh docker.io/myuser v2.0    # Custom tag
# =============================================================================

set -euo pipefail

# Default values
DEFAULT_REGISTRY="docker.io/your-username"
DEFAULT_TAG="v1.0"
IMAGE_NAME="katib-sklearn-example"

# Parse arguments
REGISTRY="${1:-$DEFAULT_REGISTRY}"
TAG="${2:-$DEFAULT_TAG}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}Building Katib Training Container${NC}"
echo -e "${GREEN}=============================================${NC}"
echo ""
echo -e "Image: ${YELLOW}${FULL_IMAGE}${NC}"
echo ""

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Working directory: $PROJECT_ROOT"
echo ""

# Step 1: Build the Docker image
echo -e "${GREEN}Step 1: Building Docker image...${NC}"
docker build -t "${FULL_IMAGE}" .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi
echo ""

# Step 2: Test locally (optional)
echo -e "${GREEN}Step 2: Testing image locally...${NC}"
echo "Running quick test with default parameters..."

docker run --rm "${FULL_IMAGE}" \
    --n-estimators 50 \
    --max-depth 5 \
    --criterion gini \
    --num-epochs 2 2>&1 | tail -20

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Local test passed${NC}"
else
    echo -e "${YELLOW}⚠ Local test may have issues - check output above${NC}"
fi
echo ""

# Step 3: Push to registry
echo -e "${GREEN}Step 3: Pushing to registry...${NC}"
echo "Pushing ${FULL_IMAGE}..."

docker push "${FULL_IMAGE}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Push successful${NC}"
else
    echo -e "${RED}✗ Push failed - check your registry credentials${NC}"
    echo "Try: docker login ${REGISTRY%%/*}"
    exit 1
fi
echo ""

# Summary
echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}Build Complete!${NC}"
echo -e "${GREEN}=============================================${NC}"
echo ""
echo "Image: ${FULL_IMAGE}"
echo ""
echo "Next steps:"
echo "  1. Update experiment YAMLs with your image:"
echo "     sed -i 's|your-username/katib-sklearn-example:v1.0|${IMAGE_NAME}:${TAG}|g' experiments/*.yaml"
echo ""
echo "  2. Submit an experiment:"
echo "     kubectl apply -f experiments/01-basic-random-search.yaml"
echo ""

