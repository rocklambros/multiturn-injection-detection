# 10: sklearn Baselines (Iteration 0)

**Description:** TF-IDF + Logistic Regression and TF-IDF + Random Forest baselines.

**GitHub Issues:** #11 — [Models] Iteration 0: sklearn baselines

**Prerequisites:** Prompts 03 (cleaned data), 08 (evaluation framework)

**Expected Outputs:**
- `src/models/baselines.py`
- `results/iter0_baseline_lr/` and `results/iter0_baseline_rf/` with metrics and plots

---

## Prompt

You are an ML engineer establishing performance baselines.

<investigate_before_answering>
1. Read `PRD.md` Section 2.4 Iteration 0 for exact pipeline configurations.
2. Read `data/processed/single_turn_train.csv` and `data/processed/single_turn_test.csv`.
3. Read `src/evaluation/metrics.py` for the evaluation interface.
</investigate_before_answering>

### Task

Implement two sklearn pipelines and evaluate on both single-turn and multi-turn test sets.

**Completion criteria:**
- pipeline_lr: TfidfVectorizer(max_features=10000, ngram_range=(1,2)) + LogisticRegression(max_iter=1000, random_state=42)
- pipeline_rf: same vectorizer + RandomForestClassifier(n_estimators=100, random_state=42)
- Evaluate on single-turn test set (expect ~90% F1)
- Evaluate on multi-turn test set (concatenate turns, expect failure)
- Save metrics via evaluation framework
- Document WHY baselines fail on multi-turn

### Tool Guidance

- **Goodman plugin:** Persist baseline F1 scores to agent memory. All downstream models compare against these.
- **Ralph loops:** Train, evaluate, verify metrics JSON files exist.

**Execution:** `claude --prompt prompts/10_baselines.md --ultrathink`
