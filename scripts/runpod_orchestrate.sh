#!/bin/bash
set -euo pipefail

# RunPod 5-GPU Parallel Training Orchestrator
#
# Prerequisites:
#   1. runpodctl installed: wget -qO- cli.runpod.net | sudo bash
#   2. runpodctl configured: runpodctl config --apiKey YOUR_KEY
#   3. WandB API key ready
#
# Usage:
#   bash scripts/runpod_orchestrate.sh [--gpu-type GPU_ID] [--wandb-key KEY]
#
# Provisions 5 H100 pods, bootstraps them in parallel, runs training,
# monitors completion, downloads results, and terminates pods.

# --- Configuration ---
GPU_TYPE="NVIDIA H100 80GB HBM3"
GPU_TYPE_FALLBACK="NVIDIA A100 80GB PCIe"
CLOUD_TYPE="SECURE"
VOLUME_SIZE=50
CONTAINER_DISK=20
IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
TASKS=("gru_retrain" "iter5" "iter6" "distilbert_hier" "distilbert_concat")
WANDB_KEY=""
MAX_WAIT=14400  # 4 hours max per task
POLL_INTERVAL=60
RESULTS_DIR="runpod_results"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-type) GPU_TYPE="$2"; shift 2 ;;
        --wandb-key) WANDB_KEY="$2"; shift 2 ;;
        --tasks-only) IFS=',' read -ra TASKS <<< "$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { log "FATAL: $*"; exit 1; }

# --- Preflight checks ---
log "=== RunPod 5-GPU Orchestrator ==="
command -v runpodctl >/dev/null 2>&1 || die "runpodctl not installed. Run: wget -qO- cli.runpod.net | sudo bash"

if [ -z "$WANDB_KEY" ]; then
    [ -f "$HOME/.wandb_key" ] && WANDB_KEY=$(cat "$HOME/.wandb_key")
    [ -f "/root/.wandb_key" ] && WANDB_KEY=$(cat "/root/.wandb_key")
    [ -z "$WANDB_KEY" ] && die "No WandB key. Pass --wandb-key KEY or put it in ~/.wandb_key"
fi
log "WandB key: ${WANDB_KEY:0:8}..."

log "Tasks: ${TASKS[*]}"
log "GPU: $GPU_TYPE"
log "Fallback GPU: $GPU_TYPE_FALLBACK"

mkdir -p "$RESULTS_DIR"

# --- Pod tracking ---
declare -A POD_IDS
declare -A POD_STATUS
declare -A POD_START

# Cleanup on exit
cleanup() {
    log "Cleaning up pods..."
    for task in "${!POD_IDS[@]}"; do
        local pid="${POD_IDS[$task]}"
        if [ -n "$pid" ]; then
            log "  Terminating pod $pid ($task)..."
            runpodctl remove pod "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT

# --- Step 1: Create all pods in parallel ---
log ""
log "=== Step 1: Provisioning ${#TASKS[@]} pods ==="

for task in "${TASKS[@]}"; do
    log "  Creating pod for: $task"

    STARTUP_CMD="echo '$WANDB_KEY' > /root/.wandb_key && apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1 && git clone -b feature/phase3-training-pipeline https://github.com/rocklambros/multiturn-injection-detection.git /workspace/multiturn-injection-detection 2>/dev/null && cd /workspace/multiturn-injection-detection && pip install -q -r requirements.txt 2>&1 | tail -1 && bash scripts/bootstrap_runpod.sh $task 2>&1 | tee /workspace/${task}.log; echo EXIT_CODE=\$? >> /workspace/${task}.log"

    POD_ID=$(runpodctl create pod \
        --name "train-${task}" \
        --gpuType "$GPU_TYPE" \
        --gpuCount 1 \
        --volumeSize "$VOLUME_SIZE" \
        --containerDiskSize "$CONTAINER_DISK" \
        --imageName "$IMAGE" \
        --cloudType "$CLOUD_TYPE" \
        --startJupyter false \
        --startSsh true \
        --args "$STARTUP_CMD" \
        2>&1 | grep -oP 'pod "?\K[a-z0-9]+' || echo "")

    if [ -z "$POD_ID" ]; then
        log "  WARN: H100 unavailable for $task, trying fallback ($GPU_TYPE_FALLBACK)..."
        POD_ID=$(runpodctl create pod \
            --name "train-${task}" \
            --gpuType "$GPU_TYPE_FALLBACK" \
            --gpuCount 1 \
            --volumeSize "$VOLUME_SIZE" \
            --containerDiskSize "$CONTAINER_DISK" \
            --imageName "$IMAGE" \
            --cloudType "$CLOUD_TYPE" \
            --startJupyter false \
            --startSsh true \
            --args "$STARTUP_CMD" \
            2>&1 | grep -oP 'pod "?\K[a-z0-9]+' || echo "")
    fi

    [ -z "$POD_ID" ] && die "Failed to create pod for $task"

    POD_IDS[$task]="$POD_ID"
    POD_STATUS[$task]="PROVISIONING"
    POD_START[$task]=$(date +%s)
    log "  Pod $POD_ID created for $task"
done

log "All ${#TASKS[@]} pods provisioned"
log ""

# --- Step 2: Monitor until all complete ---
log "=== Step 2: Monitoring training ==="
log "Max wait: ${MAX_WAIT}s per task"
log "Poll interval: ${POLL_INTERVAL}s"
log ""

COMPLETED=0
FAILED=0

while [ $COMPLETED -lt ${#TASKS[@]} ]; do
    for task in "${TASKS[@]}"; do
        [ "${POD_STATUS[$task]}" = "DONE" ] && continue
        [ "${POD_STATUS[$task]}" = "FAILED" ] && continue

        pid="${POD_IDS[$task]}"
        elapsed=$(( $(date +%s) - ${POD_START[$task]} ))

        # Check if pod still exists
        pod_info=$(runpodctl get pod "$pid" 2>/dev/null || echo "NOT_FOUND")

        if echo "$pod_info" | grep -q "NOT_FOUND\|terminated"; then
            POD_STATUS[$task]="FAILED"
            FAILED=$((FAILED + 1))
            COMPLETED=$((COMPLETED + 1))
            log "[$task] Pod $pid terminated unexpectedly after ${elapsed}s"
            continue
        fi

        # Check if training completed by looking for WandB artifact
        artifact_check=$(python3 -c "
import wandb
api = wandb.Api()
try:
    art = api.artifact('rockcyber/multiturn-injection-detection-v2/${task}_results:latest')
    print('FOUND')
except wandb.errors.CommError:
    print('NOT_FOUND')
" 2>/dev/null || echo "ERROR")

        if [ "$artifact_check" = "FOUND" ]; then
            POD_STATUS[$task]="DONE"
            COMPLETED=$((COMPLETED + 1))
            log "[$task] COMPLETED in ${elapsed}s — results artifact found"
            # Terminate pod
            runpodctl remove pod "$pid" 2>/dev/null || true
            unset POD_IDS[$task]
            continue
        fi

        # Timeout check
        if [ $elapsed -gt $MAX_WAIT ]; then
            POD_STATUS[$task]="FAILED"
            FAILED=$((FAILED + 1))
            COMPLETED=$((COMPLETED + 1))
            log "[$task] TIMEOUT after ${elapsed}s"
            runpodctl remove pod "$pid" 2>/dev/null || true
            unset POD_IDS[$task]
            continue
        fi

        # Status update
        if [ "${POD_STATUS[$task]}" = "PROVISIONING" ] && echo "$pod_info" | grep -qi "running"; then
            POD_STATUS[$task]="RUNNING"
            log "[$task] Pod running (${elapsed}s elapsed)"
        fi
    done

    # Print summary every poll
    running=$(printf '%s\n' "${POD_STATUS[@]}" | grep -c "RUNNING\|PROVISIONING" || true)
    done_count=$(printf '%s\n' "${POD_STATUS[@]}" | grep -c "DONE" || true)
    fail_count=$(printf '%s\n' "${POD_STATUS[@]}" | grep -c "FAILED" || true)

    if [ $COMPLETED -lt ${#TASKS[@]} ]; then
        log "Status: $done_count done, $running running, $fail_count failed — waiting ${POLL_INTERVAL}s"
        sleep "$POLL_INTERVAL"
    fi
done

# --- Step 3: Download results ---
log ""
log "=== Step 3: Downloading results ==="

for task in "${TASKS[@]}"; do
    if [ "${POD_STATUS[$task]}" = "DONE" ]; then
        log "Downloading $task results..."
        python3 -c "
import wandb
run = wandb.init(project='multiturn-injection-detection-v2', job_type='collect', name='collect_${task}')
art = run.use_artifact('${task}_results:latest')
art.download('$RESULTS_DIR/${task}')
run.finish()
print(f'  Downloaded to $RESULTS_DIR/${task}/')
"
    else
        log "SKIPPING $task (status: ${POD_STATUS[$task]})"
    fi
done

# --- Summary ---
log ""
log "=== SUMMARY ==="
for task in "${TASKS[@]}"; do
    elapsed=$(( $(date +%s) - ${POD_START[$task]} ))
    status="${POD_STATUS[$task]}"
    if [ "$status" = "DONE" ]; then
        log "  ✓ $task — completed in ${elapsed}s"
    else
        log "  ✗ $task — $status after ${elapsed}s"
    fi
done

if [ $FAILED -gt 0 ]; then
    log ""
    log "WARNING: $FAILED tasks failed. Check WandB for partial results."
    exit 1
fi

log ""
log "All ${#TASKS[@]} tasks completed. Results in $RESULTS_DIR/"
