# Installation and Setup

## Prerequisites

- Python 3.10+
- CUDA 12.x (optional, for GPU acceleration)
- ~4 GB disk space for data and model weights
- ~8 GB RAM minimum (16 GB+ recommended)
- Git

## Quick Start

```bash
git clone https://anonymous.4open.science/r/multiturn-injection-detection-73E6.git
cd multiturn-injection-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.download && python -m src.data.download_extra
jupyter notebook notebooks/multiturn_injection_detection.ipynb
```

## Detailed Setup

### Virtual Environment

Using the standard library `venv` module:

```bash
python3.10 -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows
```

Using Conda:

```bash
conda create -n injection-detection python=3.10
conda activate injection-detection
```

### Installing Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Key packages installed by this command:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.2.0 | Neural network training and inference |
| `transformers` | ≥ 4.36.0, < 5 | Pre-trained encoder baselines (DistilBERT) |
| `scikit-learn` | ≥ 1.3.0 | Logistic regression and TF-IDF baselines |
| `datasets` | ≥ 2.14.0 | HuggingFace dataset loading |
| `matplotlib` / `seaborn` | ≥ 3.7 / ≥ 0.12 | Plots and figures |
| `nltk` | ≥ 3.8.0 | Tokenization utilities |
| `tqdm` | ≥ 4.65.0 | Progress bars |

### NLTK Data

One additional download is required for the tokenizer:

```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

### GloVe Embeddings (Iteration 2 only)

GloVe embeddings are optional. They are used only in the second model iteration. To download them:

```bash
python -m src.data.download_glove
```

This fetches the 6B-token, 100-dimensional GloVe vectors from Stanford NLP and places them in `data/embeddings/`. If the automated download hangs, see the Troubleshooting section below.

### CUDA Verification

To confirm PyTorch detects your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected output on a CUDA-enabled machine: `True <device name>`. On CPU-only hardware: `False CPU`.

## Using Published HuggingFace Artifacts

Running the full data pipeline and training all model iterations takes under 30 minutes on GPU. As an alternative, pre-processed data and trained weights are available directly from HuggingFace. Both repositories are gated and require approval before access is granted.

- **Dataset:** [REDACTED/multiturn-injection-detection](https://anonymous.4open.science/r/multiturn-injection-detection-73E6) — processed single-turn CSVs and synthetic multi-turn conversation JSONs
- **Model:** [REDACTED/multiturn-injection-detector](https://anonymous.4open.science/r/multiturn-injection-detection-73E6) — trained weights for the GRU encoder, multi-turn LSTM with attention, and ablation variants

After your access request is approved, install the HuggingFace CLI and authenticate:

```bash
pip install huggingface_hub
huggingface-cli login
```

Then download the artifacts:

```python
from huggingface_hub import snapshot_download
snapshot_download("REDACTED/multiturn-injection-detection", repo_type="dataset", local_dir="data/")
snapshot_download("REDACTED/multiturn-injection-detector", local_dir="models/")
```

Note: all eight source HuggingFace datasets used in this project are publicly accessible without authentication. Only the derived artifacts above require approval.

## Hardware Notes

The project was developed and evaluated on two platforms:

- **Primary development:** NVIDIA Jetson Orin AGX (64 GB unified RAM, 2048-core Ampere GPU, CUDA 12.6)
- **Extended evaluation:** RunPod cloud GPU (RTX 4090) for ablation studies and transformer comparisons

All code falls back to CPU automatically. Training times on CPU are approximately 3× longer than on GPU.

Training batch sizes are set to 64 for single-turn models and 32 for multi-turn models. These values can be adjusted in `src/utils/config.py` if memory is constrained.

Model sizes:

- Multi-turn LSTM with attention: **27K trainable parameters**
- DistilBERT baseline: **66M total parameters, 99K trainable** (head only)

## Reproducing Results

All random operations are seeded with the value `42` via `src/utils/seed.py`, which sets seeds for Python's `random` module, NumPy, PyTorch, and cuDNN deterministic mode. Given identical hardware and software versions, results should match the reported figures within floating-point rounding.

Expected notebook execution times:

- GPU: under 30 minutes
- CPU: under 90 minutes

All training data originates from public HuggingFace datasets. Trained model weights are saved to `models/`; metrics, loss curves, and figures are saved to `results/`.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `torch.cuda.is_available()` returns `False` | Verify CUDA installation matches PyTorch build: compare `nvidia-smi` output with `torch.version.cuda` |
| HuggingFace download timeout or failure | Set `HF_HUB_ENABLE_HF_TRANSFER=0` and retry |
| Out of memory during training | Reduce batch size in `src/utils/config.py` |
| Missing NLTK tokenizer | Run `python -c "import nltk; nltk.download('punkt_tab')"` |
| GloVe download hangs | Download manually from the [Stanford NLP GloVe page](https://nlp.stanford.edu/projects/glove/) and place the extracted file in `data/embeddings/` |
