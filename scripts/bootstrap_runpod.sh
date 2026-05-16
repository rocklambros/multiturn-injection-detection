#!/bin/bash
set -euo pipefail

TASK="${1:-}"
WANDB_KEY_FILE="/root/.wandb_key"
REPO_URL="https://github.com/rocklambros/multiturn-injection-detection.git"
REPO_BRANCH="${REPO_BRANCH:-feature/phase3-training-pipeline}"
REPO_DIR="/workspace/multiturn-injection-detection"
ARTIFACT="rockcyber/multiturn-injection-detection-v2/synthetic_v2_data:latest"
HEARTBEAT_INTERVAL=60

usage() {
    cat <<EOF
Usage: bash bootstrap_runpod.sh <task>

Tasks: gru_retrain | iter5 | iter6 | distilbert_hier | distilbert_concat

Prerequisites:
  - WandB API key in /root/.wandb_key
  - Internet access for git clone + artifact download

This script: clones repo, installs deps, downloads artifacts,
runs training, uploads results to WandB, and shuts down on completion.
EOF
    exit 1
}

[ -z "$TASK" ] && usage

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { log "FATAL: $*"; exit 1; }

trap 'log "ERROR on line $LINENO (exit $?)"; upload_partial_results' ERR

upload_partial_results() {
    log "Uploading partial results before exit..."
    cd "$REPO_DIR" 2>/dev/null || return
    python3 -c "
import wandb, os, glob
try:
    run = wandb.init(project='multiturn-injection-detection-v2', job_type='partial-upload', name='${TASK}_partial')
    art = wandb.Artifact('${TASK}_partial', type='model')
    for p in glob.glob('models/v2_${TASK}*') + glob.glob('results/v2_${TASK}*/**', recursive=True):
        if os.path.isfile(p):
            art.add_file(p)
    if art.manifest.entries:
        run.log_artifact(art)
        print('Partial results uploaded')
    run.finish()
except Exception as e:
    print(f'Partial upload failed: {e}')
" 2>/dev/null || true
}

# --- Validate environment ---
log "=== RunPod Bootstrap: task=$TASK ==="
log "Validating environment..."

nvidia-smi || die "No GPU detected"
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || die "PyTorch CUDA not available"

GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
GPU_MEM=$(python3 -c "import torch; p=torch.cuda.get_device_properties(0); m=getattr(p,'total_memory',getattr(p,'total_mem',0)); print(f'{m/1e9:.1f}GB')")
log "GPU: $GPU_NAME ($GPU_MEM)"

[ -f "$WANDB_KEY_FILE" ] || die "WandB key not found at $WANDB_KEY_FILE. Run: echo 'YOUR_KEY' > $WANDB_KEY_FILE"
export WANDB_API_KEY=$(cat "$WANDB_KEY_FILE")
[ -n "$WANDB_API_KEY" ] || die "WandB key file is empty"

wandb login --relogin 2>/dev/null || die "WandB login failed"
log "WandB authenticated"

# --- Clone and setup ---
log "Setting up repository..."
if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR"
    git fetch origin
    git checkout "$REPO_BRANCH" 2>/dev/null || git checkout -b "$REPO_BRANCH" "origin/$REPO_BRANCH"
    git reset --hard "origin/$REPO_BRANCH"
    log "Repository updated (branch: $REPO_BRANCH)"
else
    git clone -b "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    log "Repository cloned (branch: $REPO_BRANCH)"
fi

log "Installing dependencies..."
pip install -q -r requirements.txt 2>&1 | tail -3

# --- Download artifacts ---
log "Downloading WandB artifacts..."
python3 -c "
import wandb
run = wandb.init(project='multiturn-injection-detection-v2', job_type='download', name='${TASK}_download')
artifact = run.use_artifact('$ARTIFACT')
base = artifact.download('.')
run.finish()
print(f'Artifact downloaded to {base}')
"

# Verify critical files exist
for f in data/synthetic_v2/multiturn_train.json data/processed/single_turn_train.csv models/vocab.json results/encoder_decision.json; do
    [ -f "$f" ] || die "Missing artifact file: $f"
done
log "All artifact files verified"

# iter5/iter6 need the retrained GRU encoder
if [[ "$TASK" == "iter5" || "$TASK" == "iter6" ]]; then
    [ -f "models/v2_gru_retrain.pt" ] || die "Missing models/v2_gru_retrain.pt — gru_retrain must complete first"
    log "Frozen encoder verified: models/v2_gru_retrain.pt"
fi

# --- Run training ---
log "Starting training: $TASK"
log "Command: python scripts/run_training.py --task $TASK"

TRAIN_START=$(date +%s)

python3 scripts/run_training.py --task "$TASK" 2>&1 | tee "/workspace/${TASK}_training.log"
TRAIN_EXIT=$?

TRAIN_END=$(date +%s)
TRAIN_DURATION=$(( TRAIN_END - TRAIN_START ))
log "Training finished in ${TRAIN_DURATION}s (exit code: $TRAIN_EXIT)"

[ $TRAIN_EXIT -eq 0 ] || die "Training failed with exit code $TRAIN_EXIT"

# --- Verify outputs ---
log "Verifying training outputs..."

case "$TASK" in
    gru_retrain)
        [ -f "models/v2_gru_retrain.pt" ] || die "Model file missing"
        [ -f "results/v2_gru_retrain/training_history.json" ] || die "History file missing"
        ;;
    iter5)
        [ -f "models/v2_iter5_multiturn.pt" ] || die "Model file missing"
        [ -f "results/v2_iter5_multiturn/training_history.json" ] || die "History file missing"
        ;;
    iter6)
        [ -f "models/v2_iter6_attention.pt" ] || die "Model file missing"
        [ -f "results/v2_iter6_attention/training_history.json" ] || die "History file missing"
        ;;
    distilbert_hier)
        [ -f "models/v2_distilbert_hier.pt" ] || die "Model file missing"
        [ -f "results/v2_distilbert_hier/training_history.json" ] || die "History file missing"
        ;;
    distilbert_concat)
        [ -f "models/v2_distilbert_concat.pt" ] || die "Model file missing"
        [ -f "results/v2_distilbert_concat/training_history.json" ] || die "History file missing"
        ;;
esac
log "Output verification passed"

# --- Upload results ---
log "Uploading results to WandB..."
python3 -c "
import wandb, os, glob

run = wandb.init(project='multiturn-injection-detection-v2', job_type='results-upload', name='${TASK}_results')

art = wandb.Artifact('${TASK}_results', type='model', description='Training results for $TASK')

for pattern in ['models/v2_${TASK}*', 'results/v2_${TASK}*/**']:
    for p in glob.glob(pattern, recursive=True):
        if os.path.isfile(p):
            art.add_file(p)
            print(f'  Added: {p}')

# Also upload encoder_decision if gru_retrain updated it
if '${TASK}' == 'gru_retrain' and os.path.exists('results/encoder_decision.json'):
    art.add_file('results/encoder_decision.json')
    print('  Added: results/encoder_decision.json')

run.log_artifact(art)
run.finish()
print('Results uploaded successfully')
"

log "=== Task $TASK completed successfully ==="
log "Duration: ${TRAIN_DURATION}s"
log "GPU: $GPU_NAME ($GPU_MEM)"
log "Results uploaded to WandB artifact: ${TASK}_results"
