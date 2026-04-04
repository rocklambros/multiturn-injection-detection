# 10: sklearn Baselines (Iteration 0)

**Description:** TF-IDF + Logistic Regression and TF-IDF + Random Forest baselines.

**GitHub Issues:** #11 — [Models] Iteration 0: sklearn baselines

**Prerequisites:** Prompts 03 (cleaned data), 08 (evaluation framework)

**Expected Outputs:**
- `src/models/baselines.py`
- `results/iter0_baseline_lr/` and `results/iter0_baseline_rf/` with metrics and plots

---

## Role

You are an ML engineer establishing performance baselines.

## Grounding

Use **superpowers** to read:
1. `PRD.md` Section 2.4 Iteration 0 for exact pipeline configurations.
2. `data/processed/single_turn_train.csv` (first 10 rows) and `data/processed/single_turn_test.csv` (first 10 rows).

Use **goodmem** to read `data.train_samples`, `data.test_samples`, `foundation.eval_ready`.

## Task

Implement two sklearn pipelines and evaluate on both single-turn and multi-turn test sets.

**Completion criteria:**
- pipeline_lr: TfidfVectorizer(max_features=10000, ngram_range=(1,2)) + LogisticRegression(max_iter=1000, random_state=42)
- pipeline_rf: same vectorizer + RandomForestClassifier(n_estimators=100, random_state=42)
- Evaluate on single-turn test set (expect ~90% F1)
- Evaluate on multi-turn test set (concatenate turns as single string — expect poor performance)
- Save metrics via evaluation framework
- Document WHY baselines fail on multi-turn (no temporal awareness)

## Plugin Usage

**superpowers:** Run training and evaluation.

**ralph-loop:**
1. Generate `src/models/baselines.py`
2. Execute: Train both pipelines, evaluate, save metrics
3. Review: Single-turn F1 > 80%? Multi-turn F1 notably lower? Metrics JSON files exist?
4. Fix any issues
5. Confirm

**goodmem:** After completion, persist:
- `baselines.lr_f1_single = <val>`
- `baselines.rf_f1_single = <val>`
- `baselines.lr_f1_multiturn = <val>`
- `baselines.rf_f1_multiturn = <val>`

**serena:** Checkpoint — baselines are the performance floor all DL models must beat.

## Verification

```bash
cat results/iter0_baseline_lr/metrics.json
cat results/iter0_baseline_rf/metrics.json
```

**Execution:** `claude --prompt prompts/10_baselines.md --ultrathink`
