# Spec: Synthetic Multi-Turn Data Redesign & Systematic Fix

**Date:** 2026-05-15
**Author:** Rock Lambros
**Status:** Approved
**Scope:** Fix all issues identified in adversarial review; redesign synthetic data generation; add ablation suite; prepare for NeurIPS + security venue submission.

---

## 1. Problem Statement

The adversarial review identified 18 findings across 5 rounds. Four are critical:

1. **Synthetic data circularity:** 60% of attack strategies paste the raw injection text as the final turn, making attacks trivially detectable per-turn. 68.8% of MT test source injections overlap with the turn encoder's training data. 55.8% overlap between MT train and MT test.
2. **Mask bug:** `multi_turn.py` and `attention.py` accept a mask parameter but never use it for the LSTM input. Padded turns corrupt recurrent state.
3. **Threshold tuning on test set:** `run_iteration_7()` sweeps thresholds on test data. The notebook falsely claims validation set was used.
4. **No ablation studies:** The temporal reasoning claim is unsupported without controls.

Additional findings: BCELoss numerical stability, seed-at-import-time, missing statistical significance, no transformer baseline, PRD/report mismatch.

## 2. Design Decisions (from Brainstorming)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Per-turn detectability threshold | Moderate (< 0.5) for val/test hard gate | Realistic: suggestive language OK, overt injection forbidden |
| Paraphrasing mechanism | Fragment-all: shared fragment engine + strategy-specific placement | Uniform distribution mechanism, preserves strategy diversity |
| Source text partitioning | Independent 3-way partition of all injection + benign texts | Most rigorous, prevents all leakage vectors |
| Validation gate | Hard gate for val/test (< 0.5), soft metric for train | Clean evaluation, diverse training |
| Benign filler pools | Separate per split | Full separation: nothing in MT test was seen during encoder training |
| LLM generation model | Sonnet 4.6 for Easy/Medium/Hard, Opus 4.7 for Adversarial | Best cost/quality ratio; hard gate ensures quality floor |
| Dataset size | 36,000+ sequences (30K LLM + 7K template) | Sufficient for NeurIPS statistical rigor |
| Difficulty tiers | Easy / Medium / Hard / Adversarial | Per-difficulty evaluation for the paper |
| Full dialogue | Hard + Adversarial tiers include AI assistant responses | Realistic for security venue reviewers |

### Premortem Additions

| ID | Addition | Rationale |
|----|----------|-----------|
| PM-1 | Multi-turn transformer baseline (DistilBERT + [SEP] concatenation) | Addresses "why not just concatenate?" reviewer objection |
| PM-2 | Three-subset evaluation: Full / Hard / Easy | Quantifies temporal reasoning value; avoids selection bias |
| PM-3 | Full-dialogue generation for Hard/Adversarial tiers | Realistic for security venue |

## 3. Data Generation Architecture

### 3.1 Dataset Composition

| Category | Train | Val | Test | Total |
|----------|-------|-----|------|-------|
| LLM Easy | 6,000 | 1,000 | 1,500 | 8,500 |
| LLM Medium | 6,000 | 1,000 | 1,500 | 8,500 |
| LLM Hard (full-dialogue) | 6,000 | 1,000 | 1,500 | 8,500 |
| LLM Adversarial (full-dialogue) | 2,000 | 500 | 1,000 | 3,500 |
| Template-based baseline | 5,000 | 1,000 | 1,000 | 7,000 |
| **Total** | **25,000** | **4,500** | **6,500** | **36,000** |
| Human-validated subset | -- | -- | 200 | 200 |

50/50 attack/benign balance within each category. Each attack split is balanced across 4 strategies (fragment_distributed 40%, gradual_escalation 25%, context_priming 20%, instruction_layering 15%).

### 3.2 Source Text Pool Partitioning

Independent 3-way partition of ALL texts from the merged single-turn dataset:

**Injection texts (~26,000):**
- Pool A (70%): ~18,200 -- MT train generation only
- Pool B (15%): ~3,900 -- MT val generation only
- Pool C (15%): ~3,900 -- MT test generation only

**Benign texts (~47,000):**
- Pool A (70%): ~32,900 -- MT train fillers + benign sequences
- Pool B (15%): ~7,050 -- MT val fillers + benign sequences
- Pool C (15%): ~7,050 -- MT test fillers + benign sequences

**Manifest:** `data/synthetic_v2/partition_manifest.json` records SHA-256 hash per text, pool assignment, generation timestamp, cross-pool overlap verification (must be zero).

### 3.3 LLM Generation Strategy

**Per-strategy fragment distribution:**

All 4 strategies use sentence/word-level fragmentation. Each strategy applies a distinct PLACEMENT PATTERN:

- **Fragment distributed (40%):** Random positions across turns, interleaved with benign fillers. Fragments can land anywhere.
- **Gradual escalation (25%):** Fragments appear in ascending severity order. Early turns get setup fragments, later turns get payload fragments. Conversational framing escalates progressively.
- **Context priming (20%):** First 1-2 fragments establish a persona/context. Middle turns are benign. Final 1-2 fragments exploit the established context.
- **Instruction layering (15%):** Each fragment is embedded within a constraint-setting turn. Cumulative effect builds toward the injection goal.

**Difficulty definitions for LLM prompts:**

- **Easy:** 3-5 fragments, generic conversational framing, moderate vocabulary overlap with known injections. LLM instruction: "Create a multi-turn conversation where the attack fragments are distributed but may use security-related vocabulary."
- **Medium:** 2-3 fragments, natural conversational flow, low vocabulary overlap. LLM instruction: "Create a natural-sounding conversation where attack intent is subtle. Avoid obvious security keywords."
- **Hard (full-dialogue):** 2 fragments with strong benign framing. Includes AI assistant responses. LLM instruction: "Create a realistic dialogue between a user and an AI assistant. The user's messages should each appear completely innocent. The attack intent should only be visible when reading all user messages in sequence."
- **Adversarial (full-dialogue):** Explicit evasion constraint. Includes AI assistant responses. LLM instruction: "Create a dialogue that would evade a text classifier trained on known injection patterns. Every user message must score below 0.3 on an injection classifier. Use synonyms, indirect phrasing, and plausible deniability."

**Template-based baseline:** Uses the existing `synthetic.py` fragment-all approach with the fixed strategies (no raw injection as final turn). Serves as a controlled comparison.

### 3.4 Validation Gate

- **Train:** Soft metric. Report per-turn score distribution. No rejection.
- **Val/Test Easy/Medium/Hard:** Hard gate. Run the best trained single-turn model (iter4_gru, retrained with BCEWithLogitsLoss fix as T3.2) on every individual turn. Reject + regenerate if ANY turn scores >= 0.5.
- **Val/Test Adversarial:** Hard gate with stricter threshold: reject if ANY turn scores >= 0.3.
- **Max regeneration:** 5 attempts per sequence. After 5 failures, keep with `gate_status: "failed"` flag. Report gate pass rates.
- **Validation report:** `data/synthetic_v2/validation_report.json` with per-turn score histograms, rejection rates per strategy/difficulty, distribution statistics.

### 3.5 Generation Infrastructure

**Prompt iteration phase (before batch):**
- Test generation prompts on 50 samples per strategy x 4 difficulty tiers = 800 test samples
- Manual review of 20 samples per category
- Iterate prompts until quality is acceptable
- Save final prompts as `data/synthetic_v2/generation_prompts/`

**Batch generation phase:**
- 5 async worker processes on RunPod CPU instance
- Workers 1-3: Sonnet 4.6 (Easy/Medium/Hard tiers)
- Worker 4: Template-based generation (no API)
- Worker 5: Opus 4.7 (Adversarial tier)
- Each worker writes to own output shard
- Merge + deduplicate after completion
- Temperature=0 for all API calls
- Save exact prompts + responses for reproducibility

**Reproducibility manifest:** `data/synthetic_v2/generation_manifest.json` records:
- API model version strings
- Temperature and sampling parameters
- SHA-256 of each prompt and response
- Generation timestamps
- Total API cost

## 4. Systematic Downstream Fixes

### 4.1 Code Fixes (No Retraining)

| ID | File | Fix | Details |
|----|------|-----|---------|
| F3 | `src/models/run_multi_turn.py` | Threshold tuning on val_loader | Accept both val_loader and test_loader. Sweep on val. Evaluate once on test. |
| F5 | All model files + `train.py` | BCELoss -> BCEWithLogitsLoss | Remove `torch.sigmoid()` from forward(). Use raw logits. Update train loop and eval to handle logits. |
| F6 | All `src/**/*.py` | Remove seed at import time | Delete `set_global_seed(42)` from top of every file. Set once in entry points only. |
| F7 | `src/models/multi_turn.py:67` | Remove debug print | Delete conditional print hack. |
| F8 | `PRD.md` | Update dataset count | Change "three datasets" to reflect actual 8 datasets used. |
| F9 | `src/training/train.py` | Log gradient norms | Log `clip_grad_norm_` return value to WandB. |
| F10 | Notebook cell 37 | Fix false claim | Update text to match actual threshold tuning methodology. |

### 4.2 Architecture Fixes (Require Retraining)

| ID | File | Fix | Details |
|----|------|-----|---------|
| F1 | `src/models/multi_turn.py` | Fix mask bug | Zero out turn encodings where mask==0 BEFORE feeding to sequence LSTM. `turn_encodings = turn_encodings * mask.unsqueeze(-1)` |
| F2 | `src/models/attention.py` | Fix mask bug | Same approach: mask turn encodings before LSTM. Attention mask stays as-is for softmax. |
| PM-1 | `src/models/transformer_multiturn.py` (new) | Multi-turn transformer baseline | DistilBERT processing concatenated turns with [SEP] tokens. Fine-tune classification head. |

### 4.3 Evaluation Additions (Require Clean Data)

| ID | Description | Details |
|----|-------------|---------|
| A1 | Mean pooling ablation | Average turn encodings (no LSTM), feed to classifier. Tests if ordering matters. |
| A2 | Shuffled turns ablation | Random turn order, same LSTM. Tests if sequence matters. |
| A3 | Last-turn-only ablation | Only final turn encoding -> classifier. Tests if model just reads last turn. |
| A4 | Random encoder ablation | Untrained encoder + sequence LSTM. Tests if encoder quality matters. |
| A5 | Turn-order sensitivity | For correctly classified attacks: shuffle, re-predict. Quantify ordering contribution. |
| A6 | Per-strategy F1 breakdown | Report F1/precision/recall per injection strategy. |
| A7 | Per-difficulty F1 breakdown | Report F1/precision/recall per difficulty tier. |
| A8 | Three-subset evaluation | Full / Hard (no turn >= 0.5) / Easy (some turn >= 0.5). Per PM-2. |
| A9 | Bootstrap confidence intervals | 1000 resamples, 95% CIs on all key metrics. Paired tests for main comparisons. |

### 4.4 Human Validation (Paper-Specific)

- Sample 200 sequences from test set (50 per difficulty tier)
- 2-3 annotators (NOT including the author for primary annotation)
- Per-sequence ratings: realism (1-5 Likert), turn-level attack identification, binary attack judgment
- Report: inter-annotator agreement (Cohen's kappa), average realism score, human vs. model detection accuracy
- Annotation guidelines document: `docs/annotation_guidelines.md`

## 5. RunPod Infrastructure

### 5.1 GPU Allocation

**Largest available GPU** (preferably A100 80GB or H100). Up to 5 instances.

**Wave 1 — Data generation (CPU-only, 1 instance):**
- 5 async workers generating data in parallel
- Estimated time: 4-8 hours depending on API throughput
- Output: `data/synthetic_v2/` with all shards + manifests

**Wave 2 — Training (5 GPU instances, parallel):**

| GPU | Task | Est. Time |
|-----|------|-----------|
| 1 | Retrain single-turn GRU (on 8-dataset data, for validation gate) | 20 min |
| 2 | Retrain iter5 multi-turn base (mask-fixed, new data) | 30 min |
| 3 | Retrain iter6 attention (mask-fixed, new data) | 30 min |
| 4 | Train PM-1 DistilBERT multi-turn transformer | 45 min |
| 5 | Template-based iter5 retrain (for comparison) | 30 min |

**Wave 3 — Ablations (5 GPU instances, parallel):**

| GPU | Task | Est. Time |
|-----|------|-----------|
| 1 | Ablation A1: mean pooling | 15 min |
| 2 | Ablation A2: shuffled turns | 15 min |
| 3 | Ablation A3: last-turn-only | 10 min |
| 4 | Ablation A4: random encoder | 15 min |
| 5 | iter7 threshold tuning (fixed, on val) | 10 min |

**Wave 4 — Evaluation (1-2 GPU instances):**

| Task | Est. Time |
|------|-----------|
| A5: Turn-order sensitivity analysis | 20 min |
| A6-A8: Per-strategy, per-difficulty, three-subset evaluation | 30 min |
| A9: Bootstrap confidence intervals | 45 min |

### 5.2 WandB Configuration

- Project: `multiturn-injection-detection-v2`
- Entity: (user's WandB org)
- Groups: `data-gen`, `training`, `ablation`, `evaluation`
- Tags per run: `{model_type}`, `{data_tier}`, `{ablation_type}`
- Artifacts: model checkpoints, dataset shards, partition manifests
- Custom panels: F1 comparison bar chart, per-strategy heatmap, training curves overlay

### 5.3 Bootstrap Script

Each RunPod instance is bootstrapped with:
1. Clone repo from GitHub
2. Install dependencies from `requirements.txt`
3. Download data artifacts from WandB (or sync from generation instance)
4. Set WandB API key from `pass` credential store
5. Run assigned training/ablation script
6. Upload results to WandB

## 6. Compute Cost Estimate

| Item | Estimated Cost |
|------|---------------|
| Sonnet 4.6 API (27K sequences) | $70-120 |
| Opus 4.7 API (3.5K sequences) | $50-80 |
| RunPod GPU (5x A100, ~4 hours total) | $30-60 |
| WandB | Free tier |
| **Total** | **$150-260** |

## 7. Success Criteria

The fix is successful if:
1. Zero cross-pool text overlap in partition manifest
2. Val/test hard gate pass rate > 70% (sequences where no turn >= 0.5)
3. At least one ablation (shuffled turns OR mean pooling) shows statistically significant degradation vs. the full temporal model (p < 0.05), confirming temporal reasoning
4. Multi-turn model outperforms per-turn baseline on the Hard subset by >= 5pp F1
5. All reported metrics have bootstrap 95% CIs
6. Human validation shows average realism score >= 3.5/5 and kappa >= 0.6
7. WandB project has complete run history for all experiments

## 8. Atomic Task Breakdown (for Parallel Subagent Development)

### Phase 0: Code Fixes (no data dependency, parallelizable)
- T0.1: Fix mask bug in multi_turn.py
- T0.2: Fix mask bug in attention.py
- T0.3: Fix threshold tuning in run_multi_turn.py
- T0.4: BCELoss -> BCEWithLogitsLoss in all model files + train.py
- T0.5: Remove seed-at-import from all src files
- T0.6: Remove debug prints from forward passes
- T0.7: Add gradient norm logging to train.py
- T0.8: Update PRD.md dataset count
- T0.9: Fix notebook false claim (cell 37)
- T0.10: Add WandB integration to train.py
- T0.11: Create multi-turn transformer baseline model (transformer_multiturn.py)
- T0.12: Create ablation model variants (mean_pooling.py, shuffled.py, last_turn_only.py)

### Phase 1: Data Generation Infrastructure (sequential, then parallel)
- T1.1: Create source text partitioner with manifest generation
- T1.2: Design + test LLM generation prompts (50 samples per strategy x 4 tiers)
- T1.3: Build async batch generation pipeline (5 workers)
- T1.4: Redesign template-based synthetic.py (fragment-all, no raw injection)
- T1.5: Build validation gate (single-turn classifier on every turn)
- T1.6: Build generation manifest + reproducibility tracking

### Phase 2: Data Generation Execution (parallel workers)
- T2.1: Run Sonnet Easy tier generation
- T2.2: Run Sonnet Medium tier generation
- T2.3: Run Sonnet Hard tier generation (full-dialogue)
- T2.4: Run Opus Adversarial tier generation (full-dialogue)
- T2.5: Run template-based baseline generation
- T2.6: Merge shards + run validation gate + generate reports
- T2.7: Generate partition manifest with SHA-256 verification

### Phase 3: RunPod Training (parallel GPUs)
- T3.1: Bootstrap RunPod instances (5x)
- T3.2: Retrain single-turn GRU (for validation gate)
- T3.3: Retrain iter5 multi-turn base (mask-fixed)
- T3.4: Retrain iter6 attention (mask-fixed)
- T3.5: Train DistilBERT multi-turn transformer
- T3.6: Retrain template-based iter5 (comparison)

### Phase 4: Ablations (parallel GPUs)
- T4.1: Run mean pooling ablation
- T4.2: Run shuffled turns ablation
- T4.3: Run last-turn-only ablation
- T4.4: Run random encoder ablation
- T4.5: Run threshold tuning (fixed, on val)

### Phase 5: Evaluation (sequential with some parallelism)
- T5.1: Turn-order sensitivity analysis
- T5.2: Per-strategy F1 breakdown
- T5.3: Per-difficulty F1 breakdown
- T5.4: Three-subset evaluation (Full/Hard/Easy)
- T5.5: Bootstrap confidence intervals + paired tests
- T5.6: Compile all results into evaluation summary

### Phase 6: Human Validation & Paper Updates
- T6.1: Sample 200 test sequences for annotation
- T6.2: Write annotation guidelines
- T6.3: Conduct annotation (2-3 annotators)
- T6.4: Compute inter-annotator agreement
- T6.5: Update report/final_report.tex with new results
- T6.6: Update notebook with new experiments
- T6.7: Update presentation

## 9. Dependency Graph

```
Phase 0 (code fixes) ─────────────────────────────────────────────────────┐
                                                                          │
Phase 1 (data infra) → Phase 2a (LLM generation, no gate) ───────────────┤
                                                                          │
Phase 0 + Phase 1 → T3.2 (retrain single-turn GRU) → Phase 2b (gate) ───┤
                                                                          │
Phase 2a + Phase 2b → Phase 3 (remaining training) ──────────────────────┘
                                                                          
Phase 3 → Phase 4 (ablations) → Phase 5 (evaluation) → Phase 6 (paper)

NOTES:
- Phase 0 tasks T0.1-T0.12 are all parallelizable (no interdependencies)
- Phase 1 tasks T1.1-T1.6 are mostly parallelizable (T1.2 depends on T1.1)
- Phase 2a (raw LLM generation) can run in parallel with Phase 0
- Phase 2b (validation gate) requires the retrained single-turn GRU from T3.2
- T3.2 (single-turn GRU retrain) requires Phase 0 code fixes (BCEWithLogitsLoss)
- Phase 3 tasks T3.3-T3.6 are all parallelizable (need Phase 2b gated data)
- Phase 4 tasks T4.1-T4.5 are all parallelizable
- Phase 5 tasks T5.1-T5.5 are mostly parallelizable (T5.6 depends on all)
```
