# Documentation Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update all GitHub repo documentation to reflect the final project state — architecture, results, dataflow, HuggingFace artifacts, and licensing.

**Architecture:** Six files: LICENSE and CITATION.cff (standalone), CONTRIBUTING.md (standalone), docs/INSTALLATION.md and docs/ARCHITECTURE.md (reference project artifacts), README.md (links to all of the above). Build leaf files first, README last.

**Tech Stack:** Markdown, Mermaid diagrams, CFF (YAML), CC BY-NC 4.0 license text.

**Spec:** `docs/superpowers/specs/2026-05-17-documentation-update-design.md`

**Writing constraints (apply to ALL tasks):**
- No AI vocabulary ("delve", "tapestry", "synergy", "leverage" as verb, "utilize")
- No sentences starting with conjunctions ("And", "But", "So")
- Academic tone with accessible explanations
- All numerical claims sourced from actual `results/*/metrics.json` files
- No mention of AI tooling or assistance in any documentation

---

### Task 1: Create LICENSE file

**Files:**
- Create: `LICENSE`

- [ ] **Step 1: Write the CC BY-NC 4.0 license file**

Write the full Creative Commons Attribution-NonCommercial 4.0 International license to `LICENSE`. Use the canonical legal text from Creative Commons. The file must begin with:

```
Creative Commons Attribution-NonCommercial 4.0 International

=======================================================================

Creative Commons Corporation ("Creative Commons") is not a law firm and
does not provide legal services or legal advice...
```

Include all four sections of the legal code: Section 1 (Definitions), Section 2 (Scope), Section 3 (License Conditions), Section 4 (Sui Generis Database Rights), Section 5 (Disclaimer), Section 6 (Term and Termination), Section 7 (Other Terms and Conditions), Section 8 (Interpretation).

- [ ] **Step 2: Verify the file exists and has reasonable length**

Run: `wc -l LICENSE`
Expected: ~400 lines (the canonical CC BY-NC 4.0 text)

- [ ] **Step 3: Commit**

```bash
git add LICENSE
git commit -m "Add CC BY-NC 4.0 license"
```

---

### Task 2: Create CITATION.cff

**Files:**
- Create: `CITATION.cff`

- [ ] **Step 1: Write the citation file**

Write the following exact content to `CITATION.cff`:

```yaml
cff-version: 1.2.0
title: "Multi-Turn Distributed Prompt Injection Detection"
message: "If you use this software, please cite it as below."
type: software
authors:
  - family-names: Lambros
    given-names: Rock
    email: rock@rockcyber.com
date-released: "2026-05-17"
url: "https://github.com/rocklambros/multiturn-injection-detection"
repository-code: "https://github.com/rocklambros/multiturn-injection-detection"
license: CC-BY-NC-4.0
keywords:
  - prompt-injection
  - multi-turn
  - deep-learning
  - nlp
  - security
  - gru
  - lstm
  - attention
abstract: >-
  A dual-encoder deep learning system that detects prompt injection attacks
  distributed across multiple conversation turns. A frozen GRU turn encoder
  produces per-message representations, and a trainable LSTM sequence
  classifier with attention reads the conversation-level pattern. Achieves
  F1=0.995 on multi-turn detection, a +10 F1 point improvement over the
  best single-turn approach.
```

- [ ] **Step 2: Validate YAML syntax**

Run: `python3 -c "import yaml; yaml.safe_load(open('CITATION.cff')); print('Valid YAML')"`
Expected: `Valid YAML`

- [ ] **Step 3: Commit**

```bash
git add CITATION.cff
git commit -m "Add CITATION.cff for GitHub citation widget"
```

---

### Task 3: Create CONTRIBUTING.md

**Files:**
- Create: `CONTRIBUTING.md`

- [ ] **Step 1: Write the contributing guide**

Write the following content to `CONTRIBUTING.md`:

```markdown
# Contributing

Thank you for your interest in this project. Contributions are welcome for extensions, bug fixes, and documentation improvements.

## Getting Started

1. Fork the repository and clone your fork
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Install dependencies: `pip install -r requirements.txt`
4. Make your changes and add tests

## Code Standards

This project follows strict conventions to ensure reproducibility and consistency:

- **Seed everything.** Every file that performs random operations must import and call `set_global_seed(42)` from `src.utils.seed` before any stochastic code runs. This covers Python's `random`, NumPy, PyTorch, and cuDNN deterministic mode.
- **Docstrings required.** Every function needs a docstring explaining inputs, outputs, and side effects.
- **Print shapes.** Log tensor/array shapes at every data transformation step. This catches dimension mismatches early and documents the data flow.
- **PyTorch for models, sklearn for baselines.** All neural network models use PyTorch. Scikit-learn is used only for TF-IDF baselines (logistic regression, random forest).
- **Save all artifacts.** Metrics go to `results/` as JSON. Model weights go to `models/` as `.pt` files. Plots go to `results/` as PNG.

## Project Layout

| Directory | Contents |
|-----------|----------|
| `src/data/` | Data download, cleaning, synthetic generation, loading |
| `src/models/` | Model architectures and training orchestration |
| `src/evaluation/` | Metrics, analysis, visualization, bootstrap CIs |
| `src/training/` | Training loop with early stopping |
| `src/utils/` | Seed, tokenizer, configuration |
| `scripts/` | Standalone utility scripts (data generation, evaluation, RunPod) |
| `tests/` | Pytest test suite |
| `notebooks/` | Jupyter notebook with full walkthrough |
| `report/` | Academic report (LaTeX + PDF) and presentation |

## Testing

Run the existing test suite before submitting:

```bash
pytest tests/ -v
```

The test suite covers:
- End-to-end pipeline validation (`test_e2e_pipeline.py`)
- Fragment engine correctness (`test_fragment_engine.py`)
- Data partitioning (`test_partition.py`)
- Validation gates (`test_validation_gate.py`)
- Loss function migration (`test_bce_migration.py`)
- Attention masking (`test_mask_fix.py`)

New features should include corresponding tests.

## Pull Requests

- Branch from `main`
- Write a clear description of what changed and why
- Ensure all tests pass
- Do not break existing iteration results (metrics stored in `results/`)
- Keep commits focused and atomic

## Hardware Considerations

The project targets the NVIDIA Jetson Orin AGX (64GB RAM, Ampere GPU). Keep models under 50M parameters and batch sizes at 64 (single-turn) or 32 (multi-turn).
```

- [ ] **Step 2: Verify the file renders the table correctly**

Run: `grep -c '|' CONTRIBUTING.md`
Expected: at least 12 (header + separator + 9 data rows for the table)

- [ ] **Step 3: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "Add contributing guidelines"
```

---

### Task 4: Create docs/INSTALLATION.md

**Files:**
- Create: `docs/INSTALLATION.md`

- [ ] **Step 1: Write the installation guide**

Write `docs/INSTALLATION.md` with these sections and content:

**Section 1 — Prerequisites:**
- Python 3.10+
- CUDA 12.x (optional, for GPU acceleration)
- ~4GB disk space for data and model weights
- ~8GB RAM minimum (16GB+ recommended)
- Git

**Section 2 — Quick Start** (exact commands):
```bash
git clone https://github.com/rocklambros/multiturn-injection-detection.git
cd multiturn-injection-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.download && python -m src.data.download_extra
jupyter notebook notebooks/multiturn_injection_detection.ipynb
```

**Section 3 — Detailed Setup:**
- Virtual environment creation (show both venv and conda commands)
- List key dependencies from requirements.txt: torch>=2.2.0, transformers>=4.36.0, scikit-learn>=1.3.0, datasets>=2.14.0, matplotlib, seaborn, nltk, tqdm
- NLTK data: `python -c "import nltk; nltk.download('punkt_tab')"`
- GloVe embeddings (optional, for iteration 2 only): `python -m src.data.download_glove`
- CUDA verification: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"`

**Section 4 — Using Published HuggingFace Artifacts:**

Explain that pre-processed data and trained weights are available as an alternative to running the full pipeline:

- **Dataset:** [rockCO78/multiturn-injection-detection](https://huggingface.co/datasets/rockCO78/multiturn-injection-detection) — contains processed single-turn CSVs and synthetic multi-turn conversation JSONs
- **Model:** [rockCO78/multiturn-injection-detector](https://huggingface.co/rockCO78/multiturn-injection-detector) — contains trained model weights (GRU encoder, multi-turn LSTM with attention, ablation variants)

Both artifacts require gated access. To use them:
```bash
pip install huggingface_hub
huggingface-cli login
# Request access at the dataset/model pages, then:
from huggingface_hub import snapshot_download
snapshot_download("rockCO78/multiturn-injection-detection", repo_type="dataset", local_dir="data/")
snapshot_download("rockCO78/multiturn-injection-detector", local_dir="models/")
```

Note: all 8 source HuggingFace datasets used in this project are publicly accessible without authentication.

**Section 5 — Hardware Notes:**
- Primary development: NVIDIA Jetson Orin AGX (64GB unified RAM, 2048-core Ampere GPU, CUDA 12.6)
- Extended evaluation: RunPod cloud GPU (RTX 4090) for ablation studies and transformer comparisons
- CPU fallback: all code runs on CPU; training takes approximately 3x longer
- Batch sizes: 64 for single-turn models, 32 for multi-turn models
- Largest model: DistilBERT (66M parameters, 99K trainable). Multi-turn LSTM: 27K trainable parameters.

**Section 6 — Reproducing Results:**
- All random operations seeded with 42 (Python `random`, NumPy, PyTorch manual seed, cuDNN deterministic mode via `src/utils/seed.py`)
- Expected notebook execution time: under 30 minutes on GPU, under 90 minutes on CPU
- All training data sourced from public HuggingFace datasets
- Model weights saved to `models/`, metrics and plots to `results/`

**Section 7 — Troubleshooting** (table format):

| Issue | Solution |
|-------|----------|
| `torch.cuda.is_available()` returns False | Verify CUDA installation matches PyTorch build: `nvidia-smi` vs `torch.version.cuda` |
| HuggingFace download timeout/failure | Set `HF_HUB_ENABLE_HF_TRANSFER=0` and retry |
| Out of memory during training | Reduce batch size in `src/utils/config.py` |
| Missing NLTK tokenizer | Run `python -c "import nltk; nltk.download('punkt_tab')"` |
| GloVe download hangs | Download manually from Stanford NLP and place in `data/embeddings/` |

- [ ] **Step 2: Verify all commands in the doc are syntactically correct**

Run: `python3 -c "print('syntax check')" && grep -c '```' docs/INSTALLATION.md`
Expected: `syntax check` and a count of code fence pairs (should be even number, at least 8)

- [ ] **Step 3: Commit**

```bash
git add docs/INSTALLATION.md
git commit -m "Add installation and setup guide"
```

---

### Task 5: Create docs/ARCHITECTURE.md

**Files:**
- Create: `docs/ARCHITECTURE.md`

- [ ] **Step 1: Read the source metric files for accurate numbers**

Run these commands and record the outputs — every number in the architecture doc must come from these files:

```bash
cat results/encoder_decision.json
cat results/chollet_analysis.json
cat results/core_finding.json
cat results/iter7_threshold/metrics.json
cat results/null_calibration.json | python3 -c "import sys,json; d=json.load(sys.stdin); print('bow_mean:', d['thresholds']['bow_scores_mean'], 'voting_mean:', d['thresholds']['voting_scores_mean'])"
```

- [ ] **Step 2: Write the architecture document**

Write `docs/ARCHITECTURE.md` with these sections using the exact numbers from step 1:

**Section 1 — Overview:**
Dual-encoder architecture diagram (mermaid). Explain the decomposition: a frozen turn encoder trained on abundant single-turn data (51,373 samples) produces per-message representations, while a trainable sequence classifier trained on scarcer multi-turn data (5,000 conversations) reads the conversation-level attack pattern. This separation lets each component train on data matched to its task.

**Section 2 — Encoder Selection:**
GRU chosen over LSTM, BiLSTM, GloVe LSTM. Source: `encoder_decision.json`.
- Iter 1 LSTM: F1=0.8143
- Iter 2 GloVe LSTM: F1=0.8134
- Iter 3 BiLSTM (dropout=0.3): F1=0.8145
- Iter 4 GRU: F1=0.8151
- GRU wins: competitive F1 with fewer parameters (no separate cell state vector). Dropout=0.3 selected from iter3 comparison (0.3 vs 0.5).

**Section 3 — Chollet Heuristic Analysis:**
Source: `chollet_analysis.json`.
- Training samples: 51,373. Mean words per sample: 87.3.
- Chollet ratio = 51,373 / 87.3 = 588
- Threshold: 1,500. Below threshold, bag-of-bigrams models outperform sequence models.
- Empirical confirmation table:

| Model Family | Best F1 | Params |
|-------------|---------|--------|
| TF-IDF + Random Forest | 0.834 | — |
| GRU (sequence) | 0.815 | 2.6M |
| Custom Transformer | 0.808 | 2.8M |
| DistilBERT (frozen, 99K trainable) | 0.806 | 66M |

Conclusion: model family selection should follow the data, not the hype cycle. Transformers require ratio > 1,500 to become competitive.

**Section 4 — Multi-Turn Architecture:**
Source: `core_finding.json`.
- Single-turn GRU applied per-turn to multi-turn conversations: F1=0.887
- Multi-turn LSTM (iter 5): F1=0.989
- F1 gap: +10.2 points — temporal context is necessary for detecting distributed attacks
- Design: frozen GRU produces 32-dimensional turn embeddings. Sequence LSTM (64-dim hidden state) reads the sequence of turn embeddings. Max 10 turns per conversation.

**Section 5 — Attention Mechanism:**
- Additive (Bahdanau-style) attention over LSTM hidden states
- Produces per-turn importance weights, enabling interpretability
- Security analysts can see which conversation turns most influenced the detection decision
- F1 improvement: 0.989 (iter 5) to 0.992 (iter 6)
- Implementation: `src/models/attention.py`

**Section 6 — Threshold Tuning:**
- In security applications, false negatives (missed attacks) carry higher cost than false positives (false alarms)
- Default threshold 0.5 optimized to 0.64 via validation set sweep
- F1 improvement: 0.992 to 0.995
- Final confusion matrix (1,000 test conversations): 498 TN, 2 FP, 3 FN, 497 TP

**Section 7 — Ablation Studies:**
Seven variants tested in `src/models/ablations.py` to validate that the model genuinely learns temporal patterns:

| Ablation | What it tests |
|----------|---------------|
| Shuffled turns | Whether turn order matters (random permutation) |
| Reversed turns | Whether attack directionality matters |
| Mean pooling | LSTM replaced with mean of turn embeddings |
| Max pooling | LSTM replaced with max of turn embeddings |
| Autoencoder | Unsupervised turn representations vs supervised |
| Prefix-only | Detection using only the first N turns |
| Continuation | Detection using only the last N turns |

**Section 8 — Confound Gates:**
Validation that the model learns genuine patterns rather than shortcuts:
- **Null calibration** (`results/null_calibration.json`): BoW overlap mean=1.0 (expected for template-based generation), voting mean=0.679 — confirms the model doesn't rely on simple lexical overlap between attack/benign pairs
- **Shared-prefix testing**: attack and benign conversations share identical opening turns, forcing the model to discriminate based on later turns only
- **Implementation**: `src/data/confound_gates.py`, `src/data/shared_prefix_generator.py`

**Section 9 — Data Design Decisions:**
Four synthetic attack strategies based on published research:

| Strategy | Distribution | Source |
|----------|-------------|--------|
| Fragment distribution | 40% | Split injection payload across turns, interleave with benign messages |
| Gradual escalation | 30% | Crescendo pattern (Russinovich et al., USENIX Security 2025) |
| Context priming | 20% | Establish a persona or authority, exploit it in later turns |
| Instruction layering | 10% | Cumulative constraint override across turns |

Note the evolution: v1 synthetic data (template-based), v2 synthetic data (added topic diversity and harder examples), shared-prefix generation for controlled evaluation.

- [ ] **Step 3: Verify all section headers exist**

Run: `grep '^##' docs/ARCHITECTURE.md | wc -l`
Expected: at least 9

- [ ] **Step 4: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "Add architecture decisions document"
```

---

### Task 6: Rewrite README.md

**Files:**
- Modify: `README.md`

This is the largest task. The README links to all other docs created in Tasks 1-5.

- [ ] **Step 1: Read current README for content to preserve**

Run: `cat README.md` — preserve the following sections with light editing only:
- "What Is This Project?" attack table and explanation (lines ~8-33)
- "How It Works" dual-encoder explanation (lines ~38-62)
- Mermaid diagram for dual-encoder architecture (lines ~208-244)
- References section (lines ~416-424)

Everything else is rewritten or restructured.

- [ ] **Step 2: Write the new README.md**

The new README follows this exact structure. Each section is described below with its content:

**Title + Badges:**
```markdown
# Multi-Turn Distributed Prompt Injection Detection

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Dataset on HF](https://img.shields.io/badge/HuggingFace-Dataset-yellow.svg)](https://huggingface.co/datasets/rockCO78/multiturn-injection-detection)
[![Model on HF](https://img.shields.io/badge/HuggingFace-Model-yellow.svg)](https://huggingface.co/rockCO78/multiturn-injection-detector)

**A deep learning system that detects prompt injection attacks hidden across multiple conversation turns — where no single message looks malicious on its own.**
```

**What Is This Project?** — Preserve the existing attack table example and surrounding prose from the current README (lines 8-33). Light editing only for tone consistency.

**How It Works** — Preserve the existing Step 1 / Step 2 explanation and the inline code sequence diagram (lines 38-62). Verify the "2.6M params" claim for the GRU is accurate.

**Key Results** — Rewrite with verified numbers. Table format:

| Iteration | Model | Single-Turn F1 | Multi-Turn F1 |
|-----------|-------|----------------|---------------|
| 0 | TF-IDF + Logistic Regression | 0.814 | 0.656 |
| 0 | TF-IDF + Random Forest | 0.834 | 0.739 |
| 1 | LSTM | 0.814 | — |
| 2 | LSTM + GloVe | 0.813 | — |
| 3 | BiLSTM + Dropout | 0.815 | — |
| 4 | GRU | 0.815 | 0.887 (per-turn) |
| 4b | Custom Transformer | 0.808 | — |
| 4c | DistilBERT (frozen) | 0.806 | — |
| **5** | **Multi-turn LSTM** | — | **0.989** |
| **6** | **+ Attention** | — | **0.992** |
| **7** | **+ Threshold tuning** | — | **0.995** |

Follow with: callout block highlighting the +10 F1 point gap (core finding). One paragraph on what this means in practical terms (missing 1 in 9 attacks vs 1 in 200).

**Transformer Comparison** — Keep the Chollet heuristic explanation. Verify ratio=588, threshold=1500. Reference `docs/ARCHITECTURE.md` for the full analysis.

**Architecture Diagrams** — Four mermaid diagrams. Update:
1. **Data Pipeline** — Add nodes for `shared_prefix_generator.py`, `synthetic_v2.py`, `confound_gates.py`. Keep existing nodes. Verify data counts: 51,373/11,008/11,009 single-turn; 5,000/1,000/1,000 multi-turn.
2. **Model Training Pipeline** — Add nodes for ablation studies (7 variants), null calibration. Add v2/v3 retraining flow. Keep existing iteration nodes.
3. **Dual-Encoder Architecture** — Keep as-is from current README (already accurate).
4. **Deliverables Flow** — Add LaTeX report node (final_report.tex -> final_report.pdf), Gamma presentation node. Update HTML export node.

**Project Structure** — Full tree rewrite. Group by directory with one-line annotations:

```
multiturn-injection-detection/
├── notebooks/
│   └── multiturn_injection_detection.ipynb    # Full walkthrough: all iterations, 24+ visualizations
├── src/
│   ├── data/
│   │   ├── download.py                # Base HuggingFace datasets (3 sources)
│   │   ├── download_extra.py          # Additional datasets (5 sources, 73K total)
│   │   ├── download_glove.py          # GloVe embeddings (optional, iter 2)
│   │   ├── clean.py                   # 9-step cleaning pipeline
│   │   ├── synthetic.py               # Multi-turn conversation generator (4 strategies)
│   │   ├── synthetic_v2.py            # V2 generation with topic diversity
│   │   ├── shared_prefix_generator.py # Controlled attack/benign pairs
│   │   ├── loader.py                  # PyTorch DataLoaders
│   │   ├── batch_generator.py         # Batch construction utilities
│   │   ├── confound_gates.py          # Data quality validation gates
│   │   ├── intent_extractor.py        # Turn-level intent classification
│   │   ├── manifest.py                # Dataset manifest tracking
│   │   ├── partitioner.py             # Train/val/test splitting
│   │   ├── response_stripper.py       # Assistant response removal
│   │   ├── topic_pool.py              # Topic diversity for generation
│   │   └── validation_gate.py         # Pre-training data validation
│   ├── models/
│   │   ├── single_turn.py             # LSTM/GRU/BiLSTM architectures
│   │   ├── transformer.py             # Custom Transformer classifier
│   │   ├── multi_turn.py              # Dual-encoder multi-turn classifier
│   │   ├── attention.py               # Additive attention mechanism
│   │   ├── baselines.py               # TF-IDF + LR/RF baselines
│   │   ├── ablations.py               # 7 ablation variants
│   │   ├── concat_distilbert.py       # DistilBERT concatenation baseline
│   │   ├── transformer_multiturn.py   # Transformer multi-turn variant
│   │   ├── run_single_turn.py         # Iterations 1-4 training
│   │   ├── run_transformers.py        # Iterations 4b-4c + Chollet analysis
│   │   └── run_multi_turn.py          # Iterations 5-7 training
│   ├── evaluation/
│   │   ├── metrics.py                 # F1, precision, recall, ROC-AUC
│   │   ├── analysis.py                # Error analysis, confusion matrices
│   │   ├── visualization.py           # Training curves, threshold plots
│   │   ├── bootstrap.py               # Bootstrap confidence intervals
│   │   └── per_tier.py                # Per-difficulty-tier evaluation
│   ├── training/
│   │   └── train.py                   # Training loop with early stopping
│   └── utils/
│       ├── seed.py                    # Global seed (42) for reproducibility
│       ├── tokenizer.py               # 20K-token vocabulary builder
│       └── config.py                  # All hyperparameters and paths
├── scripts/
│   ├── generate_data.py               # End-to-end data generation
│   ├── generate_v3_data.py            # V3 shared-prefix data
│   ├── run_training.py                # Full training pipeline
│   ├── run_ablations.py               # Ablation study runner
│   ├── run_evaluation.py              # Evaluation pipeline
│   ├── run_extended_evaluation.py     # Extended eval with bootstrap
│   ├── run_null_calibration.py        # Null calibration gates
│   ├── run_trivial_baselines.py       # Trivial baseline comparisons
│   ├── generate_embedding_tsne.py     # t-SNE embedding visualization
│   ├── generate_gate_activations.py   # GRU gate activation analysis
│   ├── generate_loss_landscape.py     # Loss landscape visualization
│   ├── add_viz_cells.py               # Notebook visualization cells
│   ├── patch_notebook_adversarial.py  # Notebook adversarial patches
│   ├── update_notebook_v3.py          # V3 notebook updates
│   ├── update_notebook_v3_part2.py    # V3 notebook updates (cont.)
│   ├── regenerate_clean.py            # Re-run cleaning pipeline
│   ├── collect_runpod_results.py      # Collect RunPod GPU results
│   ├── upload_wandb_artifact.py       # WandB artifact upload
│   ├── provision_runpod.py            # RunPod instance setup
│   ├── bootstrap_runpod.sh            # RunPod bootstrap script
│   └── runpod_orchestrate.sh          # RunPod job orchestration
├── tests/
│   ├── conftest.py                    # Pytest fixtures
│   ├── test_e2e_pipeline.py           # End-to-end pipeline test
│   ├── test_fragment_engine.py        # Fragment distribution test
│   ├── test_partition.py              # Data partitioning test
│   ├── test_validation_gate.py        # Validation gate test
│   ├── test_bce_migration.py          # BCE loss migration test
│   └── test_mask_fix.py               # Attention mask test
├── data/
│   ├── processed/                     # Cleaned single-turn CSVs
│   └── synthetic/                     # Generated multi-turn JSONs
├── models/                            # Saved weights (.pt) + vocab.json
├── results/                           # Metrics (JSON) + plots (PNG)
├── report/
│   ├── final_report.tex               # LaTeX source
│   ├── final_report.pdf               # Compiled report
│   ├── final_report.md                # Markdown version
│   ├── presentation.md                # Slide content
│   └── gamma_prompt.md                # Gamma presentation prompt
├── docs/
│   ├── ARCHITECTURE.md                # Architecture decisions
│   └── INSTALLATION.md                # Setup guide
├── CONTRIBUTING.md                    # Contribution guidelines
├── CITATION.cff                       # Machine-readable citation
├── LICENSE                            # CC BY-NC 4.0
└── requirements.txt                   # Python dependencies
```

**Published Artifacts** — New section:
```markdown
## Published Artifacts

Pre-processed data and trained model weights are available on HuggingFace Hub (gated access — request approval on each page):

| Artifact | Link | Contents |
|----------|------|----------|
| **Dataset** | [rockCO78/multiturn-injection-detection](https://huggingface.co/datasets/rockCO78/multiturn-injection-detection) | Processed single-turn CSVs (73K samples), synthetic multi-turn JSONs (7K conversations), vocabulary |
| **Model** | [rockCO78/multiturn-injection-detector](https://huggingface.co/rockCO78/multiturn-injection-detector) | Trained GRU encoder, multi-turn LSTM+attention weights, ablation model variants |

See [Installation Guide](docs/INSTALLATION.md#using-published-huggingface-artifacts) for download instructions.
```

**Datasets** — Keep existing 8-dataset table from current README. All source datasets are publicly accessible (no gating). Add a row or paragraph noting the 7,000 synthetic multi-turn conversations with four attack strategies (fragment distribution 40%, gradual escalation 30%, context priming 20%, instruction layering 10%).

**Hardware** — One paragraph: primary target NVIDIA Jetson Orin AGX (64GB RAM, 2048-core Ampere GPU, CUDA 12.6). Extended evaluation on RunPod RTX 4090. Notebook execution under 30 minutes on GPU. All models trainable on consumer hardware.

**Quick Links:**
```markdown
## Documentation

- **[Installation Guide](docs/INSTALLATION.md)** — Environment setup, data download, troubleshooting
- **[Architecture Decisions](docs/ARCHITECTURE.md)** — Encoder selection, Chollet analysis, ablation findings, confound gates
- **[Contributing](CONTRIBUTING.md)** — Code standards, testing, pull request process
- **[Dataset on HuggingFace](https://huggingface.co/datasets/rockCO78/multiturn-injection-detection)** — Pre-processed data (gated)
- **[Model on HuggingFace](https://huggingface.co/rockCO78/multiturn-injection-detector)** — Trained weights (gated)
```

**Citation:**
```markdown
## Citation

If you use this work, please cite:

```bibtex
@software{lambros2026multiturn,
  author = {Lambros, Rock},
  title = {Multi-Turn Distributed Prompt Injection Detection},
  year = {2026},
  url = {https://github.com/rocklambros/multiturn-injection-detection}
}
```

Or in plain text:

> Lambros, R. (2026). *Multi-Turn Distributed Prompt Injection Detection.* GitHub. https://github.com/rocklambros/multiturn-injection-detection
```

**License:**
```markdown
## License

This project is licensed under [CC BY-NC 4.0](LICENSE) — free for non-commercial use with attribution.
```

**References** — Keep the existing 5 references from the current README unchanged.

**Author** — `**Author:** Rock Lambros | May 2026`

- [ ] **Step 3: Verify all internal links resolve**

Run:
```bash
for f in docs/ARCHITECTURE.md docs/INSTALLATION.md CONTRIBUTING.md LICENSE CITATION.cff; do
  [ -f "$f" ] && echo "OK: $f" || echo "MISSING: $f"
done
```
Expected: all 5 files show OK

- [ ] **Step 4: Verify mermaid diagram count**

Run: `grep -c 'mermaid' README.md`
Expected: 8 (4 opening + 4 closing fences)

- [ ] **Step 5: Verify no AI vocabulary slipped in**

Run: `grep -iE 'delve|tapestry|synergy|utilize|leverage[sd]?' README.md docs/ARCHITECTURE.md docs/INSTALLATION.md CONTRIBUTING.md`
Expected: no output

- [ ] **Step 6: Verify no conjunction-starting sentences**

Run: `grep -nE '^\s*(And|But|So|Yet) ' README.md docs/ARCHITECTURE.md docs/INSTALLATION.md CONTRIBUTING.md`
Expected: no output (may show markdown list items starting with "So" inside code blocks — those are fine)

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "Rewrite README with final results, HF artifacts, updated project tree"
```

---

### Task 7: Final Verification

**Files:** All 6 files created/modified in Tasks 1-6.

- [ ] **Step 1: Verify all files exist**

Run:
```bash
for f in LICENSE CITATION.cff CONTRIBUTING.md docs/INSTALLATION.md docs/ARCHITECTURE.md README.md; do
  lines=$(wc -l < "$f")
  echo "$f: $lines lines"
done
```

Expected output (approximate line counts):
```
LICENSE: ~400 lines
CITATION.cff: ~25 lines
CONTRIBUTING.md: ~70 lines
docs/INSTALLATION.md: ~120 lines
docs/ARCHITECTURE.md: ~250 lines
README.md: ~500 lines
```

- [ ] **Step 2: Run the full writing constraint check across all docs**

```bash
echo "=== AI vocabulary check ==="
grep -rilE 'delve|tapestry|synergy|utilize|leverage[sd]?' LICENSE CITATION.cff CONTRIBUTING.md docs/INSTALLATION.md docs/ARCHITECTURE.md README.md || echo "PASS"

echo "=== Conjunction-start check ==="
grep -rnE '^\s*(And|But|So|Yet) ' CONTRIBUTING.md docs/INSTALLATION.md docs/ARCHITECTURE.md README.md || echo "PASS"

echo "=== AI mention check ==="
grep -rilE 'claude|anthropic|ai.assisted|ai.generated|chatgpt|copilot' LICENSE CITATION.cff CONTRIBUTING.md docs/INSTALLATION.md docs/ARCHITECTURE.md README.md || echo "PASS"
```

Expected: all three checks PASS

- [ ] **Step 3: Verify YAML is valid**

Run: `python3 -c "import yaml; yaml.safe_load(open('CITATION.cff')); print('CITATION.cff: valid')"`
Expected: `CITATION.cff: valid`

- [ ] **Step 4: Verify HuggingFace links are accessible**

Run:
```bash
curl -sI "https://huggingface.co/datasets/rockCO78/multiturn-injection-detection" | head -1
curl -sI "https://huggingface.co/rockCO78/multiturn-injection-detector" | head -1
```
Expected: both return `HTTP/2 200` or `HTTP/1.1 200`

- [ ] **Step 5: Check git status is clean**

Run: `git status`
Expected: `nothing to commit, working tree clean` (all 6 commits from Tasks 1-6 landed)
