# 08: Evaluation Framework (Metrics, Analysis, Visualization)

**Description:** Build metrics computation, error analysis, and visualization modules used by all iterations.

**GitHub Issues:** #9 — [Evaluation] Metrics, error analysis, and visualization modules

**Prerequisites:** Prompt 01 (project skeleton)

**Expected Outputs:**
- `src/evaluation/metrics.py`
- `src/evaluation/analysis.py`
- `src/evaluation/visualization.py`

---

## Prompt

You are an ML engineer building the evaluation pipeline.

<investigate_before_answering>
1. Read `PRD.md` Sections 4.2 (Metrics), 4.3 (Error Analysis), 4.4 (Visualization).
2. Read `src/utils/config.py` for iteration names used in results/ paths.
</investigate_before_answering>

### Task

Build three evaluation modules:

**metrics.py:** `compute_metrics(y_true, y_pred, y_prob) -> dict` with F1 (primary), precision, recall, ROC-AUC, PR-AUC, confusion matrix. `save_metrics()` to JSON.

**analysis.py:** Confusion matrix PNG, top 10 FP/FN with confidence, confidence histogram, attention heatmaps for multi-turn.

**visualization.py:** Training curves, ROC curve, PR curve, cross-iteration F1 bar chart, attention heatmaps (seaborn).

**Completion criteria:**
- All plots saved as PNG to `results/{iteration_name}/`
- `compute_metrics()` with dummy data produces valid JSON output
- F1 is clearly marked as primary metric everywhere

### Tool Guidance

- **Ralph loops:** Call compute_metrics with dummy arrays. Verify output JSON structure. Generate a dummy plot. Fix any matplotlib/seaborn issues.

### Verification

```bash
python -c "
from src.evaluation.metrics import compute_metrics
import numpy as np
metrics = compute_metrics(np.array([0,1,1,0]), np.array([0,1,0,0]), np.array([0.1,0.9,0.4,0.2]))
print(metrics)
"
```

**Execution:** `claude --prompt prompts/08_eval_framework.md --ultrathink`
