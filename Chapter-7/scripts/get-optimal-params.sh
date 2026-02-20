#!/bin/bash
# =============================================================================
# Get Optimal Parameters from Katib Experiment
# =============================================================================
# Extracts the best hyperparameters found during a Katib experiment.
# Supports multiple output formats for easy integration with other tools.
#
# Usage:
#   ./scripts/get-optimal-params.sh <experiment-name> [namespace] [format]
#
# Formats:
#   json    - JSON output (default)
#   yaml    - YAML output
#   env     - Environment variable format
#   cli     - Command-line argument format
#
# Examples:
#   ./scripts/get-optimal-params.sh sklearn-random-search
#   ./scripts/get-optimal-params.sh sklearn-random-search kubeflow json
#   ./scripts/get-optimal-params.sh sklearn-random-search kubeflow cli
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: Missing experiment name${NC}" >&2
    echo "" >&2
    echo "Usage: $0 <experiment-name> [namespace] [format]" >&2
    echo "" >&2
    echo "Formats: json (default), yaml, env, cli" >&2
    exit 1
fi

EXPERIMENT_NAME="$1"
NAMESPACE="${2:-kubeflow}"
FORMAT="${3:-json}"

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: 'jq' is required${NC}" >&2
    exit 1
fi

# Get experiment JSON
EXPERIMENT_JSON=$(kubectl get experiment "${EXPERIMENT_NAME}" -n "${NAMESPACE}" -o json 2>/dev/null)

if [ -z "$EXPERIMENT_JSON" ]; then
    echo -e "${RED}Error: Experiment '${EXPERIMENT_NAME}' not found in namespace '${NAMESPACE}'${NC}" >&2
    exit 1
fi

# Check if experiment has completed
STATUS=$(echo "$EXPERIMENT_JSON" | jq -r '.status.conditions[-1].type // "Unknown"')
if [ "$STATUS" != "Succeeded" ] && [ "$STATUS" != "Running" ]; then
    echo -e "${YELLOW}Warning: Experiment status is '${STATUS}'${NC}" >&2
fi

# Extract optimal trial
OPTIMAL_TRIAL=$(echo "$EXPERIMENT_JSON" | jq -r '.status.currentOptimalTrial // empty')

if [ -z "$OPTIMAL_TRIAL" ] || [ "$OPTIMAL_TRIAL" == "null" ]; then
    echo -e "${RED}Error: No optimal trial found (experiment may not have completed any trials)${NC}" >&2
    exit 1
fi

# Output based on format
case "$FORMAT" in
    json)
        # Pretty JSON output
        echo "$OPTIMAL_TRIAL" | jq '{
            bestTrialName: .bestTrialName,
            parameters: [.parameterAssignments[] | {(.name): .value}] | add,
            metrics: [.observation.metrics[] | {(.name): .latest}] | add
        }'
        ;;
    
    yaml)
        # YAML output
        echo "# Optimal Hyperparameters from Katib Experiment: ${EXPERIMENT_NAME}"
        echo "# Best Trial: $(echo "$OPTIMAL_TRIAL" | jq -r '.bestTrialName')"
        echo ""
        echo "parameters:"
        echo "$OPTIMAL_TRIAL" | jq -r '.parameterAssignments[] | "  \(.name): \(.value)"'
        echo ""
        echo "metrics:"
        echo "$OPTIMAL_TRIAL" | jq -r '.observation.metrics[] | "  \(.name): \(.latest)"'
        ;;
    
    env)
        # Environment variable format (for sourcing)
        echo "# Source this file: source <(./get-optimal-params.sh experiment-name kubeflow env)"
        echo "$OPTIMAL_TRIAL" | jq -r '.parameterAssignments[] | 
            "export " + (.name | gsub("-"; "_") | ascii_upcase) + "=\"" + .value + "\""'
        ;;
    
    cli)
        # Command-line argument format
        echo "$OPTIMAL_TRIAL" | jq -r '.parameterAssignments[] | 
            "--" + .name + "=" + .value' | tr '\n' ' '
        echo ""
        ;;
    
    *)
        echo -e "${RED}Error: Unknown format '${FORMAT}'${NC}" >&2
        echo "Valid formats: json, yaml, env, cli" >&2
        exit 1
        ;;
esac

# Print summary to stderr (so stdout remains clean for piping)
if [ "$FORMAT" == "json" ] || [ "$FORMAT" == "yaml" ]; then
    echo "" >&2
    echo -e "${GREEN}✓ Optimal parameters extracted from experiment '${EXPERIMENT_NAME}'${NC}" >&2
    echo -e "  Best Trial: $(echo "$OPTIMAL_TRIAL" | jq -r '.bestTrialName')" >&2
    echo -e "  Objective: $(echo "$OPTIMAL_TRIAL" | jq -r '.observation.metrics[0] | "\(.name) = \(.latest)"')" >&2
fi

