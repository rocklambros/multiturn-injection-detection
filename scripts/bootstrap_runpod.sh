#!/bin/bash
set -euo pipefail

TASK="${1:-}"
WANDB_KEY_FILE="/root/.wandb_key"
GH_TOKEN_FILE="/root/.gh_token"
REPO_BRANCH="${REPO_BRANCH:-feature/v3-clean-retrain}"
REPO_DIR="/workspace/multiturn-injection-detection"
ARTIFACT="rockcyber/multiturn-injection-detection-v2/synthetic_v3_data:latest"
HEARTBEAT_INTERVAL=60

usage() {
    cat <<EOF
Usage: bash bootstrap_runpod.sh <task>

Tasks: gru_retrain | iter5 | iter6 | distilbert_hier | distilbert_concat
       ablation_shuffled | ablation_reversed | ablation_prefix
       ablation_continuation | ablation_autoencoder
       ablation_mean_pool | ablation_max_pool

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
    run = wandb.init(project='multiturn-injection-detection-v2', job_type='partial-upload', name='v3_${TASK}_partial')
    art = wandb.Artifact('v3_${TASK}_partial', type='model')
    for p in glob.glob('models/v3_${TASK}*') + glob.glob('results/v3_${TASK}*/**', recursive=True):
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

if [ -f "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY=$(cat "$WANDB_KEY_FILE")
fi
[ -n "$WANDB_API_KEY" ] || die "WandB key not found. Set WANDB_API_KEY env var or put key in $WANDB_KEY_FILE"

wandb login "$WANDB_API_KEY" 2>&1 || wandb login --relogin "$WANDB_API_KEY" 2>&1 || {
    log "WARN: wandb login command failed, proceeding with env var only"
}
log "WandB key configured: ${WANDB_API_KEY:0:15}..."

# --- Clone and setup ---
log "Setting up repository..."
if [ -f "$GH_TOKEN_FILE" ]; then
    GH_TOKEN=$(cat "$GH_TOKEN_FILE")
elif [ -n "${GH_TOKEN:-}" ]; then
    true
else
    die "GitHub token not found. Set GH_TOKEN env var or put token in $GH_TOKEN_FILE"
fi
REPO_URL="https://${GH_TOKEN}@github.com/rocklambros/multiturn-injection-detection.git"

if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR"
    git remote set-url origin "$REPO_URL"
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
import wandb, os, shutil
run = wandb.init(project='multiturn-injection-detection-v2', job_type='download', name='${TASK}_download')
artifact = run.use_artifact('$ARTIFACT')
base = artifact.download('/tmp/wandb_artifact')
run.finish()
print(f'Artifact downloaded to {base}')

# Move files to correct repo-relative paths
mapping = {
    'synthetic_v3': 'data/synthetic_v3',
    'processed': 'data/processed',
    'embeddings': 'data/embeddings',
    'models': 'models',
    'results': 'results',
}
for art_dir, repo_dir in mapping.items():
    src = os.path.join(base, art_dir)
    if os.path.exists(src):
        os.makedirs(repo_dir, exist_ok=True)
        for fname in os.listdir(src):
            src_file = os.path.join(src, fname)
            dst_file = os.path.join(repo_dir, fname)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
                print(f'  {art_dir}/{fname} -> {dst_file}')
print('Artifact files placed in repo structure')
"

# Verify critical files exist
for f in data/synthetic_v3/multiturn_train.json data/processed/single_turn_train.csv models/vocab.json results/encoder_decision.json; do
    [ -f "$f" ] || die "Missing artifact file: $f"
done
log "All artifact files verified"

# Tasks that need the retrained GRU encoder
NEEDS_ENCODER="iter5 iter6 ablation_shuffled ablation_reversed ablation_prefix ablation_continuation ablation_autoencoder ablation_mean_pool ablation_max_pool"
if echo "$NEEDS_ENCODER" | grep -qw "$TASK"; then
    if [ ! -f "models/v3_gru_retrain.pt" ]; then
        log "Downloading GRU encoder from v3_gru_retrain_results artifact..."
        python3 -c "
import wandb, shutil, os
run = wandb.init(project='multiturn-injection-detection-v2', job_type='download', name='${TASK}_encoder_download')
artifact = run.use_artifact('rockcyber/multiturn-injection-detection-v2/v3_gru_retrain_results:latest')
base = artifact.download('/tmp/gru_artifact')
run.finish()
for root, dirs, files in os.walk(base):
    for f in files:
        src = os.path.join(root, f)
        # Flatten into models/ or results/
        if f.endswith('.pt'):
            os.makedirs('models', exist_ok=True)
            shutil.copy2(src, os.path.join('models', f))
            print(f'  {f} -> models/{f}')
        elif f.endswith('.json'):
            os.makedirs('results', exist_ok=True)
            shutil.copy2(src, os.path.join('results', f))
            print(f'  {f} -> results/{f}')
" || die "Failed to download GRU encoder artifact"
    fi
    [ -f "models/v3_gru_retrain.pt" ] || die "Missing models/v3_gru_retrain.pt — gru_retrain must complete first"
    log "Frozen encoder verified: models/v3_gru_retrain.pt"
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

TASK_PREFIX="v3_${TASK}"
case "$TASK" in
    gru_retrain)     TASK_PREFIX="v3_gru_retrain" ;;
    iter5)           TASK_PREFIX="v3_iter5_multiturn" ;;
    iter6)           TASK_PREFIX="v3_iter6_attention" ;;
    distilbert_hier) TASK_PREFIX="v3_distilbert_hier" ;;
    distilbert_concat) TASK_PREFIX="v3_distilbert_concat" ;;
    ablation_*)      TASK_PREFIX="v3_${TASK}" ;;
esac

[ -f "models/${TASK_PREFIX}.pt" ] || die "Model file missing: models/${TASK_PREFIX}.pt"
[ -f "results/${TASK_PREFIX}/training_history.json" ] || die "History file missing: results/${TASK_PREFIX}/training_history.json"
log "Output verification passed"

# --- Upload results ---
log "Uploading results to WandB..."
python3 -c "
import wandb, os, glob

run = wandb.init(project='multiturn-injection-detection-v2', job_type='results-upload', name='v3_${TASK}_results')

art = wandb.Artifact('v3_${TASK}_results', type='model', description='v3 results for $TASK')

for pattern in ['models/v3_${TASK}*', 'models/${TASK_PREFIX}*', 'results/v3_${TASK}*/**', 'results/${TASK_PREFIX}*/**']:
    for p in glob.glob(pattern, recursive=True):
        if os.path.isfile(p):
            art.add_file(p)
            print(f'  Added: {p}')

if '${TASK}' == 'gru_retrain' and os.path.exists('results/encoder_decision.json'):
    art.add_file('results/encoder_decision.json')
    print('  Added: results/encoder_decision.json')

if '${TASK}' == 'ablation_autoencoder' and os.path.exists('models/v3_turn_autoencoder.pt'):
    art.add_file('models/v3_turn_autoencoder.pt')
    print('  Added: models/v3_turn_autoencoder.pt')

run.log_artifact(art)
run.finish()
print('Results uploaded successfully')
"

log "=== Task $TASK completed successfully ==="
log "Duration: ${TRAIN_DURATION}s"
log "GPU: $GPU_NAME ($GPU_MEM)"
log "Results uploaded to WandB artifact: v3_${TASK}_results"
