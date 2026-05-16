# RunPod 5-GPU Parallel Training

## Option A: Automated (requires runpodctl)

```bash
# Install runpodctl
wget -qO- cli.runpod.net | sudo bash
runpodctl config --apiKey YOUR_RUNPOD_API_KEY

# Run orchestrator
bash scripts/runpod_orchestrate.sh --wandb-key YOUR_WANDB_KEY
```

This provisions 5 H100 pods, runs training in parallel, monitors completion,
downloads results, and terminates pods automatically.

## Option B: Manual (RunPod web UI)

### 1. Provision 5 pods

Create 5 GPU pods from the RunPod dashboard:
- **Template**: RunPod PyTorch 2.4 (CUDA 12.4)
- **GPU**: H100 SXM 80GB (or A100 80GB if unavailable)
- **Volume**: 50 GB
- **Container disk**: 20 GB

Name them: `train-gru_retrain`, `train-iter5`, `train-iter6`, `train-distilbert_hier`, `train-distilbert_concat`

### 2. Bootstrap each pod (run in 5 terminal tabs)

SSH into each pod and run ONE command:

**Pod 1 — gru_retrain:**
```bash
echo 'YOUR_WANDB_KEY' > /root/.wandb_key && \
apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1 && \
git clone -b feature/phase3-training-pipeline https://github.com/rocklambros/multiturn-injection-detection.git /workspace/mt && \
cd /workspace/mt && pip install -q -r requirements.txt && \
bash scripts/bootstrap_runpod.sh gru_retrain
```

**Pod 2 — iter5:**
```bash
echo 'YOUR_WANDB_KEY' > /root/.wandb_key && \
apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1 && \
git clone -b feature/phase3-training-pipeline https://github.com/rocklambros/multiturn-injection-detection.git /workspace/mt && \
cd /workspace/mt && pip install -q -r requirements.txt && \
bash scripts/bootstrap_runpod.sh iter5
```

**Pod 3 — iter6:**
```bash
echo 'YOUR_WANDB_KEY' > /root/.wandb_key && \
apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1 && \
git clone -b feature/phase3-training-pipeline https://github.com/rocklambros/multiturn-injection-detection.git /workspace/mt && \
cd /workspace/mt && pip install -q -r requirements.txt && \
bash scripts/bootstrap_runpod.sh iter6
```

**Pod 4 — distilbert_hier:**
```bash
echo 'YOUR_WANDB_KEY' > /root/.wandb_key && \
apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1 && \
git clone -b feature/phase3-training-pipeline https://github.com/rocklambros/multiturn-injection-detection.git /workspace/mt && \
cd /workspace/mt && pip install -q -r requirements.txt && \
bash scripts/bootstrap_runpod.sh distilbert_hier
```

**Pod 5 — distilbert_concat:**
```bash
echo 'YOUR_WANDB_KEY' > /root/.wandb_key && \
apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1 && \
git clone -b feature/phase3-training-pipeline https://github.com/rocklambros/multiturn-injection-detection.git /workspace/mt && \
cd /workspace/mt && pip install -q -r requirements.txt && \
bash scripts/bootstrap_runpod.sh distilbert_concat
```

### 3. Monitor

Watch all 5 runs at: https://wandb.ai/rockcyber/multiturn-injection-detection-v2

Each pod:
- Logs to WandB in real time (loss curves, accuracy, grad norms)
- Uploads results artifact on completion
- Uploads partial results on failure (trap handler)

### 4. Collect results locally

After all 5 complete:
```bash
python scripts/collect_runpod_results.py
```

### 5. Terminate pods

Delete all 5 pods from the RunPod dashboard. The bootstrap script does NOT
auto-terminate to give you time to inspect logs if needed.

## Task dependencies

```
gru_retrain ──→ iter5  (needs frozen encoder)
             └─→ iter6  (needs frozen encoder)
distilbert_hier   (independent)
distilbert_concat (independent)
```

The WandB artifact includes the pre-trained GRU encoder (`models/v2_gru_retrain.pt`),
so iter5/iter6 can run immediately in parallel without waiting for gru_retrain.
If you want a fresh retrain, run gru_retrain first and re-upload the artifact.

## Expected timelines on H100

| Task | Epochs | Est. time |
|------|--------|-----------|
| gru_retrain | 30 (early stop ~7) | 2-3 min |
| iter5 | 30 | 8-10 min |
| iter6 | 30 | 10-12 min |
| distilbert_hier | 20 (early stop ~8) | 15-25 min |
| distilbert_concat | 10 (early stop ~5) | 10-15 min |

Total wall time: ~25 min (parallel) + ~5 min bootstrap per pod.

## Cost estimate

5 x H100 SXM for ~30 min = ~$10-15 total at RunPod community rates.
