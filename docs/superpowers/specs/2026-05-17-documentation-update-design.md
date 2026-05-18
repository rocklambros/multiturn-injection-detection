# Documentation Update Design Spec

**Date:** 2026-05-17
**Goal:** Update all GitHub repo documentation to reflect the final project state, architecture, dataflow, and results.
**Audience:** Dual — academic reviewers (COMP 4531, potential venue submission) and security practitioners.

---

## Files to Create/Update

| File | Action | Purpose |
|------|--------|---------|
| `README.md` | Rewrite | Refreshed with final results, updated project tree, accurate diagrams |
| `docs/ARCHITECTURE.md` | Create | Architecture decisions, dual-encoder rationale, Chollet analysis, ablations |
| `docs/INSTALLATION.md` | Create | Environment setup, data download, notebook execution, hardware notes |
| `CONTRIBUTING.md` | Create | Root-level for GitHub auto-detection. Code standards, PR process |
| `CITATION.cff` | Create | Machine-readable citation for GitHub "Cite this repository" widget |
| `LICENSE` | Create | CC BY-NC 4.0 full license text |

---

## README.md

### Sections (in order)

1. **Title + badges** — Project name, license badge (CC BY-NC 4.0), Python version badge. No badge overload.

2. **What Is This Project?** — Keep the existing accessible multi-turn attack table example. Light editing only. One paragraph explaining the problem, one showing the conversation table, one stating the solution.

3. **How It Works** — Keep the dual-encoder explanation (Step 1: GRU turn encoder, Step 2: LSTM sequence classifier with attention). Verify parameter counts match actual models. Keep the ASCII-style sequence diagram showing confidence building across turns.

4. **Key Results** — Table with all iterations and their F1 scores. Values sourced from `results/*/metrics.json`:
   - Baselines: LR F1=0.814, RF F1=0.834
   - Single-turn RNNs: LSTM 0.814, GloVe LSTM 0.813, BiLSTM 0.815, GRU 0.815
   - Transformers: Custom 0.808, DistilBERT 0.806
   - Multi-turn: LSTM 0.989, +Attention 0.992, +Threshold 0.995
   - Highlight the +10 F1 gap (core finding from core_finding.json)

5. **Transformer Comparison** — Chollet heuristic section. Ratio=588 < threshold 1500. Bag-of-bigrams prediction confirmed empirically. Values from chollet_analysis.json.

6. **Architecture Diagrams** (mermaid) — Four diagrams, all updated:
   - **Data Pipeline** — Update to reflect actual file list including shared_prefix_generator.py, synthetic_v2.py, confound_gates.py. Data counts: 51,373/11,008/11,009 single-turn; 5,000/1,000/1,000 multi-turn.
   - **Model Training Pipeline** — Update to include ablation studies (7 ablation variants), v2/v3 retraining steps, null calibration. Show all iteration directories.
   - **Dual-Encoder Architecture** — Keep as-is (accurate). Frozen GRU (2.6M params) + trainable sequence LSTM (~27K params) + attention + classification head.
   - **Deliverables Flow** — Update to include LaTeX PDF report (final_report.tex/pdf), Gamma presentation.

7. **Project Structure** — Full rewrite. Must include ALL files:
   - `src/data/`: download.py, download_extra.py, download_glove.py, clean.py, synthetic.py, synthetic_v2.py, shared_prefix_generator.py, loader.py, batch_generator.py, confound_gates.py, intent_extractor.py, manifest.py, partitioner.py, response_stripper.py, topic_pool.py, validation_gate.py
   - `src/models/`: single_turn.py, transformer.py, multi_turn.py, attention.py, baselines.py, ablations.py, concat_distilbert.py, transformer_multiturn.py, run_single_turn.py, run_transformers.py, run_multi_turn.py
   - `src/evaluation/`: metrics.py, analysis.py, visualization.py, bootstrap.py, per_tier.py
   - `src/training/`: train.py
   - `src/utils/`: seed.py, tokenizer.py, config.py
   - `scripts/`: ~20 utility scripts (generate_data.py, run_training.py, run_ablations.py, etc.)
   - `tests/`: 6 test files
   - `report/`: final_report.md, final_report.tex, final_report.pdf, presentation.md, gamma_prompt.md
   - Group files logically with brief annotations

8. **Iteration Roadmap** — Keep the existing table format. All F1 values verified against metrics.json.

9. **Published Artifacts** — New section documenting gated HuggingFace Hub releases:
   - **Dataset:** [rockCO78/multiturn-injection-detection](https://huggingface.co/datasets/rockCO78/multiturn-injection-detection) — processed single-turn CSVs + synthetic multi-turn JSONs. Gated access.
   - **Model:** [rockCO78/multiturn-injection-detector](https://huggingface.co/rockCO78/multiturn-injection-detector) — trained model weights (GRU encoder, multi-turn LSTM+attention). Gated access.
   - Note that these are gated artifacts requiring HuggingFace approval to access.
   - Brief explanation of what's included in each artifact.

10. **Datasets** — Keep the existing 8-dataset table. Add a column or note indicating which source datasets require authentication or are gated on HuggingFace. Add a line about 7,000 synthetic multi-turn conversations with four attack strategies (fragment distribution 40%, gradual escalation 30%, context priming 20%, instruction layering 10%).

11. **Hardware** — Keep Jetson Orin AGX note. Add mention of RunPod RTX 4090 for extended evaluation. Execution time <30 minutes.

12. **Quick Links** — Bullet list linking to docs/ARCHITECTURE.md, docs/INSTALLATION.md, CONTRIBUTING.md, and the HuggingFace artifact pages.

13. **Citation** — BibTeX block + plain text citation. Point to CITATION.cff.

14. **License** — CC BY-NC 4.0 one-liner with link to LICENSE file.

15. **References** — Keep existing 5 references unchanged.

16. **Author** — "Rock Lambros | May 2026"

### Sections removed from README (moved elsewhere)
- Detailed "Reproducibility" setup instructions -> docs/INSTALLATION.md
- "Concepts You'll Learn" section -> trimmed to one sentence pointing to the notebook
- Deep architecture rationale -> docs/ARCHITECTURE.md

### Constraints
- No AI vocabulary per user feedback memory (feedback_writing_quality.md)
- No conjunction-starting sentences
- Match academic tone with accessible explanations
- All numerical claims sourced from actual metrics.json files

---

## docs/ARCHITECTURE.md

### Sections

1. **Overview** — Dual-encoder design. Frozen turn encoder produces per-message representations; trainable sequence classifier reads the conversation-level pattern. Why this decomposition: allows the turn encoder to be trained on abundant single-turn data, while the sequence classifier trains on scarce multi-turn data.

2. **Encoder Selection** — GRU chosen over LSTM/BiLSTM. Decision from encoder_decision.json:
   - GRU F1=0.8151 vs BiLSTM F1=0.8145 vs LSTM F1=0.8143
   - GRU has fewer parameters (no separate cell state)
   - Competitive performance with lower computational cost
   - Dropout=0.3 selected via iter3 comparison

3. **Chollet Heuristic Analysis** — From chollet_analysis.json:
   - n=51,373 training samples, mean 87.3 words/sample
   - Ratio = 51,373 / 87.3 = 588
   - Threshold = 1,500 (below: bag-of-bigrams wins; above: sequence models competitive; well above: transformers win)
   - Empirical confirmation: RF (0.834) > GRU (0.815) > Transformer (0.808) > DistilBERT (0.806)
   - Lesson: model family selection should be data-driven, not hype-driven

4. **Multi-Turn Architecture** — Why sequence-level classification beats per-turn approaches:
   - Single-turn GRU applied per-turn on multi-turn data: F1=0.887
   - Multi-turn LSTM on conversation sequences: F1=0.989
   - The +10.2 F1 gap (from core_finding.json) demonstrates that temporal context is necessary
   - Design: frozen GRU produces 32-dim embeddings per turn, sequence LSTM (64-dim hidden) reads the sequence

5. **Attention Mechanism** — Additive attention (Bahdanau-style):
   - Produces per-turn importance weights
   - Enables interpretability: security analysts can see which turns triggered the alert
   - Improvement: F1 0.989 -> 0.992
   - Implementation in src/models/attention.py (123 lines)

6. **Threshold Tuning** — For security applications, false negatives (missed attacks) are costlier than false positives:
   - Default threshold 0.5 -> optimized to 0.64
   - F1 improvement: 0.992 -> 0.995
   - Confusion matrix: 498 TN, 2 FP, 3 FN, 497 TP (on 1000 test conversations)

7. **Ablation Studies** — Seven ablation variants tested (src/models/ablations.py):
   - Shuffled turns: tests whether turn order matters
   - Reversed turns: tests directionality
   - Mean pooling: replaces LSTM with mean of turn embeddings
   - Max pooling: replaces LSTM with max of turn embeddings
   - Autoencoder: unsupervised turn representations
   - Prefix-only: uses only first N turns
   - Continuation: uses only last N turns

8. **Confound Gates** — Validation that the model learns genuine patterns:
   - Null calibration (results/null_calibration.json): BoW overlap score=1.0, voting score=0.679
   - Shared-prefix testing: attack/benign pairs share identical opening turns
   - Confound gates in src/data/confound_gates.py validate data quality

9. **Data Design** — Four synthetic attack strategies and their rationale:
   - Fragment distribution (40%): mirrors real-world split-payload attacks
   - Gradual escalation (30%): Crescendo pattern (Russinovich et al.)
   - Context priming (20%): persona establishment then exploitation
   - Instruction layering (10%): cumulative constraint override
   - v2 synthetic data adds harder examples with topic diversity
   - Shared-prefix generation creates controlled evaluation pairs

---

## docs/INSTALLATION.md

### Sections

1. **Prerequisites**
   - Python 3.10+
   - CUDA 12.x (optional, for GPU acceleration)
   - ~4GB disk for data + model weights
   - ~8GB RAM minimum (16GB+ recommended)

2. **Quick Start** (5 commands)
   ```
   git clone https://github.com/rocklambros/multiturn-injection-detection.git
   cd multiturn-injection-detection
   pip install -r requirements.txt
   python -m src.data.download && python -m src.data.download_extra
   jupyter notebook notebooks/multiturn_injection_detection.ipynb
   ```

3. **Detailed Setup**
   - Virtual environment creation (venv or conda)
   - requirements.txt contents and version constraints
   - NLTK data download (punkt_tab tokenizer)
   - GloVe embeddings download (optional, for iter2)
   - Verify CUDA availability

4. **Using Published HuggingFace Artifacts (alternative to local generation)**
   - Dataset: [rockCO78/multiturn-injection-detection](https://huggingface.co/datasets/rockCO78/multiturn-injection-detection) — download pre-processed data instead of running the pipeline
   - Model: [rockCO78/multiturn-injection-detector](https://huggingface.co/rockCO78/multiturn-injection-detector) — download trained weights instead of retraining
   - Both are gated: explain how to request access and authenticate (`huggingface-cli login`)
   - Note which source datasets also require gated access or authentication on HuggingFace

5. **Hardware Notes**
   - Primary: NVIDIA Jetson Orin AGX (64GB RAM, 2048-core Ampere GPU, CUDA 12.6)
   - Also tested: RunPod RTX 4090 (for extended evaluation runs)
   - CPU fallback: works but training takes ~3x longer
   - Batch sizes: 64 single-turn, 32 multi-turn
   - No model exceeds 50M parameters (largest: DistilBERT at 66M with 99K trainable)

6. **Reproducing Results**
   - All random operations seeded with 42 (Python, NumPy, PyTorch, cuDNN deterministic)
   - Expected notebook execution: <30 minutes on GPU, <90 minutes on CPU
   - All training data from public HuggingFace datasets
   - Model weights and metrics saved to models/ and results/

7. **Troubleshooting**
   - CUDA version mismatch: check torch.cuda.is_available() and nvidia-smi
   - HuggingFace download failures: retry with HF_HUB_ENABLE_HF_TRANSFER=0
   - Memory issues: reduce batch size in src/utils/config.py
   - Missing NLTK data: run nltk.download('punkt_tab')

---

## CONTRIBUTING.md

### Sections

1. **Getting Started** — Fork, clone, create branch, install deps.

2. **Code Standards**
   - Every file imports and calls set_global_seed(42) before random operations
   - Every function has a docstring (inputs, outputs, side effects)
   - Print shapes at data transformation steps
   - PyTorch for all models; sklearn for baselines only
   - Save results to results/ and models/

3. **Testing**
   - Run existing tests: `pytest tests/`
   - Test files cover: BCE migration, e2e pipeline, fragment engine, mask fix, partition, validation gate
   - New features should include tests

4. **Pull Requests**
   - Branch from main
   - Clear description of what and why
   - All tests passing
   - No breaking changes to existing iteration results

---

## CITATION.cff

Standard Citation File Format:
- Title: "Multi-Turn Distributed Prompt Injection Detection"
- Author: Rock Lambros
- Date: 2026-05
- Type: software
- URL: repository URL
- Keywords: prompt injection, multi-turn, deep learning, NLP, security

Plus BibTeX equivalent in README:
```bibtex
@software{lambros2026multiturn,
  author = {Lambros, Rock},
  title = {Multi-Turn Distributed Prompt Injection Detection},
  year = {2026},
  url = {https://github.com/rocklambros/multiturn-injection-detection}
}
```

---

## LICENSE

Full text of Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).

---

## Writing Constraints (all files)

1. No AI vocabulary ("delve", "tapestry", "synergy", "leverage" as verb, "utilize")
2. No sentences starting with conjunctions ("And", "But", "So")
3. Academic tone with accessible explanations
4. All numerical claims sourced from actual JSON artifacts
5. No mention of AI tooling or assistance in any documentation
