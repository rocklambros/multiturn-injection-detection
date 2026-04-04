# 09: PyTorch Training Loop

**Description:** Implement reusable training loop with early stopping, LR scheduling, checkpointing, and history logging.

**GitHub Issues:** #10 — [Training] PyTorch training loop with early stopping and checkpointing

**Prerequisites:** Prompt 01 complete (project skeleton)

**Expected Outputs:**
- `src/training/train.py`

---

## Role

You are a PyTorch engineer building training infrastructure.

## Grounding

Use **superpowers** to read:
1. `PRD.md` Section 4.1 (Training Infrastructure) for function signature and requirements.
2. `src/utils/config.py` for IterationConfig fields.

## Task

Write `src/training/train.py` with:
```python
def train_model(model, train_loader, val_loader, epochs, iteration_name,
                optimizer, criterion, device, patience=3):
```

**Completion criteria:**
- Each epoch: forward pass batches, compute val loss, check early stopping
- Early stopping restores best weights when patience exhausted
- LR scheduler: ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)
- Gradient clipping: clip_grad_norm_(max_norm=1.0)
- Saves: training_history.json, model_summary.txt, {iteration_name}.pt
- GPU support with .to(device)
- Progress bars via tqdm

## Plugin Usage

**context7:** Look up `torch.optim.lr_scheduler.ReduceLROnPlateau` and `torch.nn.utils.clip_grad_norm_` to verify API signatures.

**superpowers:** Run smoke test with dummy model and random data.

**ralph-loop:**
1. Generate `src/training/train.py`
2. Create a tiny `nn.Linear(10, 1)`, random DataLoader, train 2 epochs
3. Review: Does training_history.json get created? Does .pt checkpoint save? Does early stopping logic work?
4. Fix any issues
5. Confirm

**goodmem:** After completion, persist:
- `foundation.training_ready = true`
- `foundation.training_function = src.training.train.train_model`

## Verification

```bash
python -c "
from src.training.train import train_model
print('Training loop imports OK')
"
```

**Execution:** `claude --prompt prompts/09_training_loop.md --ultrathink`
