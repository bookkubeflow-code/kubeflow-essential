#!/bin/bash
# =============================================================================
# Monitor Katib Experiment
# =============================================================================
# Provides real-time monitoring of a running Katib experiment, including:
# - Experiment status
# - Trial progress
# - Current optimal parameters
# - Resource usage
#
# Usage:
#   ./scripts/monitor-experiment.sh <experiment-name> [namespace]
#
# Examples:
#   ./scripts/monitor-experiment.sh sklearn-random-search
#   ./scripts/monitor-experiment.sh sklearn-bayesian-optimization kubeflow
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: Missing experiment name${NC}"
    echo ""
    echo "Usage: $0 <experiment-name> [namespace]"
    exit 1
fi

EXPERIMENT_NAME="$1"
NAMESPACE="${2:-kubeflow}"
REFRESH_INTERVAL=5

# Clear screen function
clear_screen() {
    clear
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║          KATIB EXPERIMENT MONITOR                           ║${NC}"
    echo -e "${CYAN}║          Refreshing every ${REFRESH_INTERVAL} seconds (Ctrl+C to exit)            ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Main monitoring loop
monitor() {
    while true; do
        clear_screen
        
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        echo -e "Timestamp: ${YELLOW}${TIMESTAMP}${NC}"
        echo -e "Experiment: ${BLUE}${EXPERIMENT_NAME}${NC}"
        echo -e "Namespace: ${BLUE}${NAMESPACE}${NC}"
        echo ""
        
        # =====================================================================
        # Experiment Status
        # =====================================================================
        echo -e "${GREEN}┌─────────────────────────────────────────────────────────────┐${NC}"
        echo -e "${GREEN}│ EXPERIMENT STATUS                                           │${NC}"
        echo -e "${GREEN}└─────────────────────────────────────────────────────────────┘${NC}"
        
        # Get experiment details
        EXPERIMENT_JSON=$(kubectl get experiment "${EXPERIMENT_NAME}" -n "${NAMESPACE}" -o json 2>/dev/null)
        
        if [ -z "$EXPERIMENT_JSON" ]; then
            echo -e "${RED}Experiment not found!${NC}"
            sleep $REFRESH_INTERVAL
            continue
        fi
        
        # Parse status
        STATUS=$(echo "$EXPERIMENT_JSON" | jq -r '.status.conditions[-1].type // "Unknown"')
        TRIALS_SUCCEEDED=$(echo "$EXPERIMENT_JSON" | jq -r '.status.trialsSucceeded // 0')
        TRIALS_FAILED=$(echo "$EXPERIMENT_JSON" | jq -r '.status.trialsFailed // 0')
        TRIALS_RUNNING=$(echo "$EXPERIMENT_JSON" | jq -r '.status.trialsRunning // 0')
        TRIALS_PENDING=$(echo "$EXPERIMENT_JSON" | jq -r '.status.trialsPending // 0')
        MAX_TRIALS=$(echo "$EXPERIMENT_JSON" | jq -r '.spec.maxTrialCount // "N/A"')
        
        # Status color
        case "$STATUS" in
            "Succeeded") STATUS_COLOR="${GREEN}" ;;
            "Running"|"Created") STATUS_COLOR="${YELLOW}" ;;
            "Failed") STATUS_COLOR="${RED}" ;;
            *) STATUS_COLOR="${NC}" ;;
        esac
        
        echo ""
        echo -e "  Status:     ${STATUS_COLOR}${STATUS}${NC}"
        echo -e "  Max Trials: ${MAX_TRIALS}"
        echo ""
        echo -e "  Trial Progress:"
        echo -e "    ✓ Succeeded: ${GREEN}${TRIALS_SUCCEEDED}${NC}"
        echo -e "    ▶ Running:   ${YELLOW}${TRIALS_RUNNING}${NC}"
        echo -e "    ○ Pending:   ${BLUE}${TRIALS_PENDING}${NC}"
        echo -e "    ✗ Failed:    ${RED}${TRIALS_FAILED}${NC}"
        
        # Progress bar
        TOTAL_COMPLETED=$((TRIALS_SUCCEEDED + TRIALS_FAILED))
        if [ "$MAX_TRIALS" != "N/A" ] && [ "$MAX_TRIALS" -gt 0 ]; then
            PROGRESS=$((TOTAL_COMPLETED * 100 / MAX_TRIALS))
            BAR_WIDTH=40
            FILLED=$((PROGRESS * BAR_WIDTH / 100))
            EMPTY=$((BAR_WIDTH - FILLED))
            echo ""
            echo -n "  Progress: ["
            printf "%${FILLED}s" | tr ' ' '█'
            printf "%${EMPTY}s" | tr ' ' '░'
            echo "] ${PROGRESS}%"
        fi
        echo ""
        
        # =====================================================================
        # Optimal Trial
        # =====================================================================
        echo -e "${GREEN}┌─────────────────────────────────────────────────────────────┐${NC}"
        echo -e "${GREEN}│ CURRENT OPTIMAL TRIAL                                       │${NC}"
        echo -e "${GREEN}└─────────────────────────────────────────────────────────────┘${NC}"
        echo ""
        
        OPTIMAL_TRIAL=$(echo "$EXPERIMENT_JSON" | jq -r '.status.currentOptimalTrial // empty')
        
        if [ -n "$OPTIMAL_TRIAL" ] && [ "$OPTIMAL_TRIAL" != "null" ]; then
            OPTIMAL_NAME=$(echo "$OPTIMAL_TRIAL" | jq -r '.bestTrialName // "N/A"')
            echo -e "  Best Trial: ${CYAN}${OPTIMAL_NAME}${NC}"
            echo ""
            
            # Parameters
            echo "  Parameters:"
            echo "$OPTIMAL_TRIAL" | jq -r '.parameterAssignments[]? | "    \(.name): \(.value)"' 2>/dev/null || echo "    N/A"
            echo ""
            
            # Metrics
            echo "  Metrics:"
            echo "$OPTIMAL_TRIAL" | jq -r '.observation.metrics[]? | "    \(.name): \(.latest)"' 2>/dev/null || echo "    N/A"
        else
            echo -e "  ${YELLOW}No optimal trial yet (waiting for first trial to complete)${NC}"
        fi
        echo ""
        
        # =====================================================================
        # Recent Trials
        # =====================================================================
        echo -e "${GREEN}┌─────────────────────────────────────────────────────────────┐${NC}"
        echo -e "${GREEN}│ RECENT TRIALS                                               │${NC}"
        echo -e "${GREEN}└─────────────────────────────────────────────────────────────┘${NC}"
        echo ""
        
        kubectl get trials -n "${NAMESPACE}" \
            -l "katib.kubeflow.org/experiment=${EXPERIMENT_NAME}" \
            --sort-by=.metadata.creationTimestamp \
            -o custom-columns=\
"NAME:.metadata.name,\
STATUS:.status.conditions[-1].type,\
ACCURACY:.status.observation.metrics[0].latest,\
AGE:.metadata.creationTimestamp" \
            2>/dev/null | tail -8 || echo "  No trials found"
        
        echo ""
        
        # =====================================================================
        # Running Pods
        # =====================================================================
        RUNNING_PODS=$(kubectl get pods -n "${NAMESPACE}" \
            -l "katib.kubeflow.org/experiment=${EXPERIMENT_NAME}" \
            --field-selector=status.phase=Running \
            --no-headers 2>/dev/null | wc -l | tr -d ' ')
        
        if [ "$RUNNING_PODS" -gt 0 ]; then
            echo -e "${GREEN}┌─────────────────────────────────────────────────────────────┐${NC}"
            echo -e "${GREEN}│ RUNNING PODS (${RUNNING_PODS})                                              │${NC}"
            echo -e "${GREEN}└─────────────────────────────────────────────────────────────┘${NC}"
            echo ""
            kubectl get pods -n "${NAMESPACE}" \
                -l "katib.kubeflow.org/experiment=${EXPERIMENT_NAME}" \
                --field-selector=status.phase=Running \
                -o custom-columns="NAME:.metadata.name,STATUS:.status.phase,CPU:.spec.containers[0].resources.requests.cpu,MEMORY:.spec.containers[0].resources.requests.memory" \
                2>/dev/null || true
            echo ""
        fi
        
        # Sleep before next refresh
        sleep $REFRESH_INTERVAL
    done
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${GREEN}Monitoring stopped.${NC}"; exit 0' INT

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: 'jq' is required for this script${NC}"
    echo "Install with: brew install jq (macOS) or apt-get install jq (Linux)"
    exit 1
fi

# Start monitoring
monitor

