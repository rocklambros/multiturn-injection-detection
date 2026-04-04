# 17: Threshold Tuning (Iteration 7)

**Description:** Sweep classification thresholds on best multi-turn model.

**GitHub Issues:** #18 — [Models] Iteration 7: Threshold tuning

**Prerequisites:** Prompt 15 or 16 (trained multi-turn model)

**Expected Outputs:**
- `results/iter7_threshold/` with sweep plot and threshold report

---

## Role

You are a security engineer optimizing detection thresholds. In this domain, missed injections (false negatives) cost far more than false alarms (false positives). Tune accordingly.

## Grounding

Use **superpowers** to read:
1. `PRD.md` Section 2.4 Iteration 7.
2. Load the best multi-turn model checkpoint.

Use **goodmem** to read `models.iter5_multiturn_f1`, `models.iter6_attention_f1` to pick the best model.

## Task

Sweep thresholds 0.01–0.99 (step 0.01). Report three optimal thresholds.

**Completion criteria:**
- Threshold maximizing F1
- Threshold achieving >= 95% recall
- Threshold achieving >= 95% precision
- Sweep plot: precision, recall, F1 vs threshold on same axes
- Security reasoning: document why FN costs more than FP

## Plugin Usage

**superpowers:** Run threshold sweep and generate plots.

**ralph-loop:**
1. Load best model, get predictions on multi-turn test set
2. Sweep thresholds, compute metrics at each
3. Generate plot
4. Review: Three thresholds identified? Plot readable? Security reasoning documented?
5. Confirm

**goodmem:** After completion, persist:
- `models.iter7_best_f1_threshold = <val>`
- `models.iter7_95recall_threshold = <val>`
- `models.iter7_95precision_threshold = <val>`
- `models.iter7_best_f1_at_threshold = <val>`

**Execution:** `claude --prompt prompts/17_threshold_tuning.md --ultrathink`
