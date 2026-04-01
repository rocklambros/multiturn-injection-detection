# 09: PyTorch Training Loop

**Description:** Implement reusable training loop with early stopping, LR scheduling, checkpointing, and history logging.

**GitHub Issues:** #10 — [Training] PyTorch training loop with early stopping and checkpointing

**Prerequisites:** Prompt 01 (project skeleton)

**Expected Outputs:**
- `src/training/train.py`

---

## Prompt

You are a PyTorch engineer building the training infrastructure.

<investigate_before_answering>
1. Read `PRD.md` Section 4.1 (Training Infrastructure) for the function signature and requirements.
2. Read `src/utils/config.py` for IterationConfig fields (epochs, patience, learning_rate).
3. Read `src/evaluation/metrics.py` for the metrics interface.
</investigate_before_answering>

### Task

Write `src/training/train.py` with `train_model()` function.

**Completion criteria:**
- Each epoch: forward pass, val loss, early stopping check
- Early stopping restores best weights
- LR scheduler: ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)
- Gradient clipping: max_norm=1.0
- Saves: training_history.json, model_summary.txt, {iteration_name}.pt
- GPU support with .to(device)
- Progress bars via tqdm
- Training 1 epoch on dummy data works

### Tool Guidance

- **Context7 plugin:** Look up PyTorch ReduceLROnPlateau and clip_grad_norm_ API.
- **Ralph loops:** Create a tiny model, train 1 epoch on random data, verify all save files are created.

### Verification

```bash
python -c "
import torch
import torch.nn as nn
from src.training.train import train_model
# Minimal smoke test with dummy model
model = nn.Linear(10, 1)
print('Training loop imports OK')
"
```

**Execution:** `claude --prompt prompts/09_training_loop.md --ultrathink`
