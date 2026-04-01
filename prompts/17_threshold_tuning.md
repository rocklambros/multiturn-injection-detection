# 17: Threshold Tuning (Iteration 7)

**Description:** Sweep classification thresholds on best multi-turn model.

**GitHub Issues:** #18 — [Models] Iteration 7: Threshold tuning

**Prerequisites:** Prompt 15 or 16 (trained multi-turn model)

**Expected Outputs:**
- `results/iter7_threshold/` with sweep plot and threshold report

---

## Prompt

You are a security engineer optimizing detection thresholds for an injection detector.

<investigate_before_answering>
1. Read `PRD.md` Section 2.4 Iteration 7.
2. Load the best multi-turn model from models/.
3. Read `src/evaluation/metrics.py` for metric computation.
</investigate_before_answering>

### Task

Sweep thresholds 0.01-0.99 (step 0.01). Report three optimal thresholds.

**Completion criteria:**
- Threshold maximizing F1
- Threshold achieving >= 95% recall
- Threshold achieving >= 95% precision
- Sweep plot: precision, recall, F1 vs threshold on same axes
- Security reasoning: why FN costs more than FP

### Tool Guidance

- **Sequential-thinking MCP:** Think about the security cost model. A missed injection (FN) means an attack succeeds. A false alarm (FP) means a benign conversation gets flagged for review. Which is worse in production?

**Execution:** `claude --prompt prompts/17_threshold_tuning.md --ultrathink`
