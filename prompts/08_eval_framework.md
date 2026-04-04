# 08: Evaluation Framework (Metrics, Analysis, Visualization)

**Description:** Build metrics computation, error analysis, and visualization modules used by all iterations.

**GitHub Issues:** #9 — [Evaluation] Metrics, error analysis, and visualization modules

**Prerequisites:** Prompt 01 complete (project skeleton)

**Expected Outputs:**
- `src/evaluation/metrics.py`
- `src/evaluation/analysis.py`
- `src/evaluation/visualization.py`

---

## Role

You are an ML engineer building a reusable evaluation pipeline.

## Grounding

Use **superpowers** to read:
1. `PRD.md` Sections 4.2 (Metrics), 4.3 (Error Analysis), 4.4 (Visualization).
2. `src/utils/config.py` for iteration names.

## Task

Build three evaluation modules.

**metrics.py:** `compute_metrics(y_true, y_pred, y_prob) -> dict` with F1 (primary), precision, recall, ROC-AUC, PR-AUC, confusion matrix. `save_metrics(metrics_dict, iteration_name)` to JSON.

**analysis.py:** Confusion matrix PNG, top 10 FP/FN with confidence, confidence histogram, attention heatmaps for multi-turn.

**visualization.py:** Training curves, ROC curve, PR curve, cross-iteration F1 bar chart, attention heatmaps (seaborn).

**Completion criteria:**
- All plots saved as PNG to `results/{iteration_name}/`
- `compute_metrics()` with dummy data produces valid JSON
- F1 clearly marked as primary metric

## Plugin Usage

**Dispatch subagents** — these three modules are independent:
- Subagent A: Build `metrics.py`
- Subagent B: Build `analysis.py`
- Subagent C: Build `visualization.py`

**ralph-loop:** For each module:
1. Generate the code
2. Call with dummy arrays: `compute_metrics(np.array([0,1,1,0]), np.array([0,1,0,0]), np.array([0.1,0.9,0.4,0.2]))`
3. Verify output JSON structure and that plot functions don't crash
4. Fix matplotlib/seaborn backend issues if any
5. Confirm

**goodmem:** After completion, persist:
- `foundation.eval_ready = true`
- `foundation.primary_metric = F1`

## Verification

```bash
python -c "
from src.evaluation.metrics import compute_metrics
import numpy as np
m = compute_metrics(np.array([0,1,1,0]), np.array([0,1,0,0]), np.array([0.1,0.9,0.4,0.2]))
print(m)
"
```

**Execution:** `claude --prompt prompts/08_eval_framework.md --ultrathink`
