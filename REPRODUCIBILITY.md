# Reproducibility

This file records the exact commands that regenerate the datasets and metrics, and the results they produced. Every headline number in the paper reproduces from public sources and the committed code. All randomness is seeded with 42.

## Environment

- Python 3.13, PyTorch 2.x (MPS or CUDA), scikit-learn, transformers, datasets.
- Reproduced on an Apple M3 Max (MPS). The original training ran on RunPod A100; extended evaluation on RunPod RTX 4090. Inference results are deterministic across these devices to three decimal places.

```bash
python3 -m venv .venv
.venv/bin/pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm nltk transformers datasets
```

## Artifacts

The full datasets and trained weights are on Zenodo (concept DOI resolves to the latest version):

- **DOI: 10.5281/zenodo.20628935**

Download from there, or regenerate the single-turn data locally with the commands below. The multi-turn evaluation needs the model weights and `vocab.json` from Zenodo placed under `models/`.

## Single-turn dataset (73,390 samples from 8 public datasets)

```bash
.venv/bin/python -m src.data.download         # deepset, safe-guard, neuralchemy
.venv/bin/python -m src.data.download_extra    # imoxto (subsampled 40K), spml, TrustAIRLab x2, jackhhao
.venv/bin/python -m src.data.clean             # 9-step clean, 70/15/15 stratified split, seed 42
# -> data/processed/single_turn_{train,val,test}.csv
```

Result (exact match to the paper):

| Quantity | Reproduced | Paper |
|---|---|---|
| Combined raw | 89,690 | -- |
| After cleaning | 73,390 | 73,390 |
| Train / Val / Test | 51,373 / 11,008 / 11,009 | 51,373 / 11,008 / 11,009 |
| Class balance | 64.2% benign / 35.8% injection | ~64% / 36% |

### Single-turn baseline (Chollet heuristic)

The TF-IDF bag-of-bigrams **Random Forest** pipeline in `src/models/baselines.py` (`TfidfVectorizer(max_features=10000, ngram_range=(1,2))` + `RandomForestClassifier(n_estimators=100, random_state=42)`), trained on `single_turn_train.csv` and evaluated on `single_turn_test.csv`, gives **test F1 = 0.834** (the Logistic Regression variant gives 0.814).

## Multi-turn evaluation (regenerates `results/v3_evaluation/`)

The evaluation script reads the multi-turn test set from `data/synthetic_v3/`. Point it at the Zenodo multi-turn JSON (or `data/hf_dataset/`):

```bash
mkdir -p data/synthetic_v3
ln -sf ../hf_dataset/multiturn_test.json  data/synthetic_v3/multiturn_test.json
ln -sf ../hf_dataset/multiturn_train.json data/synthetic_v3/multiturn_train.json
.venv/bin/python scripts/run_evaluation.py    # writes per-tier F1 + bootstrap CIs to results/v3_evaluation/
```

Result (exact match to the paper):

| Model | Reproduced F1 | Paper |
|---|---|---|
| Temporal LSTM (iter 5) | 0.8368 | 0.837 |
| LSTM + attention (iter 6) | 0.8370 | 0.837 |
| Shuffled-turns ablation | 0.7609 | 0.760 |
| DistilBERT hierarchical | 0.9756 | 0.976 |
| DistilBERT concatenated | 0.9915 | 0.992 |

The committed JSONs under `results/v3_evaluation/` are the outputs of this run, so the notebook executes against them directly.
