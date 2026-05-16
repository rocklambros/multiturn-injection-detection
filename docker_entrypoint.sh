#!/bin/bash
set -euo pipefail

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

TASK_NAME="${TASK_NAME:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
REPO_BRANCH="${REPO_BRANCH:-feature/phase3-training-pipeline}"
REPO_URL="${REPO_URL:-https://github.com/rocklambros/multiturn-injection-detection.git}"

if [ -z "$TASK_NAME" ]; then
    log "ERROR: TASK_NAME env var not set. Set it to one of: gru_retrain iter5 iter6 distilbert_hier distilbert_concat"
    log "Keeping container alive for debugging..."
    sleep infinity
fi

if [ -z "$WANDB_API_KEY" ]; then
    log "ERROR: WANDB_API_KEY env var not set"
    sleep infinity
fi

log "=== RunPod Training Container ==="
log "Task: $TASK_NAME"
log "Branch: $REPO_BRANCH"

echo "$WANDB_API_KEY" > /root/.wandb_key
export WANDB_API_KEY

log "Cloning repository..."
if [ -d "/workspace/mt/.git" ]; then
    cd /workspace/mt
    git fetch origin
    git checkout "$REPO_BRANCH" 2>/dev/null || git checkout -b "$REPO_BRANCH" "origin/$REPO_BRANCH"
    git reset --hard "origin/$REPO_BRANCH"
else
    git clone -b "$REPO_BRANCH" "$REPO_URL" /workspace/mt
    cd /workspace/mt
fi

log "Installing dependencies..."
pip install -q -r requirements.txt 2>&1 | tail -3

log "Running bootstrap..."
bash scripts/bootstrap_runpod.sh "$TASK_NAME"
EXIT_CODE=$?

log "Bootstrap finished with exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    log "Training completed successfully. Container will stay alive for 5 minutes then exit."
    sleep 300
else
    log "Training FAILED. Container will stay alive for debugging."
    sleep infinity
fi
