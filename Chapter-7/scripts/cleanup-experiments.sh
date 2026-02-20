#!/bin/bash
# =============================================================================
# Cleanup Katib Experiments
# =============================================================================
# Removes completed or failed experiments and their associated trials.
#
# Usage:
#   ./scripts/cleanup-experiments.sh [namespace] [--all|--completed|--failed]
#
# Options:
#   --all        Delete all experiments
#   --completed  Delete only succeeded experiments (default)
#   --failed     Delete only failed experiments
#   --dry-run    Show what would be deleted without deleting
#
# Examples:
#   ./scripts/cleanup-experiments.sh                    # Delete completed in kubeflow
#   ./scripts/cleanup-experiments.sh kubeflow --all     # Delete all in kubeflow
#   ./scripts/cleanup-experiments.sh kubeflow --dry-run # Preview deletions
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Defaults
NAMESPACE="${1:-kubeflow}"
FILTER="${2:---completed}"
DRY_RUN=false

# Parse additional args
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
    esac
done

echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}Katib Experiment Cleanup${NC}"
echo -e "${GREEN}=============================================${NC}"
echo ""
echo -e "Namespace: ${YELLOW}${NAMESPACE}${NC}"
echo -e "Filter:    ${YELLOW}${FILTER}${NC}"
echo -e "Dry Run:   ${YELLOW}${DRY_RUN}${NC}"
echo ""

# Build selector based on filter
case "$FILTER" in
    --all)
        SELECTOR=""
        ;;
    --completed)
        SELECTOR="--field-selector=status.conditions.type=Succeeded"
        ;;
    --failed)
        SELECTOR="--field-selector=status.conditions.type=Failed"
        ;;
    *)
        echo -e "${RED}Unknown filter: ${FILTER}${NC}"
        exit 1
        ;;
esac

# Get experiments
echo "Finding experiments..."
EXPERIMENTS=$(kubectl get experiments -n "${NAMESPACE}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")

if [ -z "$EXPERIMENTS" ]; then
    echo -e "${YELLOW}No experiments found in namespace '${NAMESPACE}'${NC}"
    exit 0
fi

# Filter based on status
FILTERED_EXPERIMENTS=""
for exp in $EXPERIMENTS; do
    STATUS=$(kubectl get experiment "$exp" -n "${NAMESPACE}" -o jsonpath='{.status.conditions[-1].type}' 2>/dev/null || echo "Unknown")
    
    case "$FILTER" in
        --all)
            FILTERED_EXPERIMENTS="$FILTERED_EXPERIMENTS $exp"
            ;;
        --completed)
            if [ "$STATUS" == "Succeeded" ]; then
                FILTERED_EXPERIMENTS="$FILTERED_EXPERIMENTS $exp"
            fi
            ;;
        --failed)
            if [ "$STATUS" == "Failed" ]; then
                FILTERED_EXPERIMENTS="$FILTERED_EXPERIMENTS $exp"
            fi
            ;;
    esac
done

FILTERED_EXPERIMENTS=$(echo "$FILTERED_EXPERIMENTS" | xargs)

if [ -z "$FILTERED_EXPERIMENTS" ]; then
    echo -e "${YELLOW}No experiments match the filter '${FILTER}'${NC}"
    exit 0
fi

# Show experiments to delete
echo ""
echo "Experiments to delete:"
for exp in $FILTERED_EXPERIMENTS; do
    STATUS=$(kubectl get experiment "$exp" -n "${NAMESPACE}" -o jsonpath='{.status.conditions[-1].type}' 2>/dev/null || echo "Unknown")
    TRIALS=$(kubectl get trials -n "${NAMESPACE}" -l "katib.kubeflow.org/experiment=$exp" --no-headers 2>/dev/null | wc -l | tr -d ' ')
    echo -e "  - ${exp} (${STATUS}, ${TRIALS} trials)"
done
echo ""

# Confirm deletion
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}Dry run mode - no changes made${NC}"
    exit 0
fi

read -p "Delete these experiments? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Delete experiments
echo ""
echo "Deleting experiments..."
for exp in $FILTERED_EXPERIMENTS; do
    echo -n "  Deleting ${exp}... "
    kubectl delete experiment "$exp" -n "${NAMESPACE}" --wait=false 2>/dev/null && \
        echo -e "${GREEN}✓${NC}" || \
        echo -e "${RED}✗${NC}"
done

echo ""
echo -e "${GREEN}Cleanup complete!${NC}"
echo ""
echo "Note: Trials and pods will be garbage collected automatically."
echo "To verify cleanup:"
echo "  kubectl get experiments -n ${NAMESPACE}"
echo "  kubectl get trials -n ${NAMESPACE}"

