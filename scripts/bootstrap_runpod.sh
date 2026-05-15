#!/bin/bash
set -euo pipefail

echo "=== RunPod Instance Bootstrap ==="

# Clone repo
if [ ! -d "multiturn-injection-detection" ]; then
    git clone https://github.com/rocklambros/multiturn-injection-detection.git
    cd multiturn-injection-detection
else
    cd multiturn-injection-detection
    git pull
fi

# Install dependencies
pip install -r requirements.txt

# Set WandB API key
export WANDB_API_KEY=$(cat /root/.wandb_key 2>/dev/null || echo "")
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set. Pass via: echo 'KEY' > /root/.wandb_key"
fi

# Download data artifacts from WandB
if command -v wandb &>/dev/null && [ -n "$WANDB_API_KEY" ]; then
    echo "Downloading data artifacts from WandB..."
    python -c "
import wandb
run = wandb.init(project='multiturn-injection-detection-v2', job_type='download')
artifact = run.use_artifact('synthetic_v2_data:latest')
artifact.download('data/synthetic_v2')
run.finish()
"
fi

echo "=== Bootstrap complete ==="
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
