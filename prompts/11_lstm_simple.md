# 11: Simple LSTM (Iteration 1)

**Description:** First deep learning model — SingleTurnLSTM with random embeddings.

**GitHub Issues:** #12 — [Models] Iteration 1: Simple LSTM with random embeddings

**Prerequisites:** Prompts 06 (DataLoader), 09 (training loop)

**Expected Outputs:**
- `src/models/single_turn.py` with `SingleTurnLSTM` class
- `results/iter1_lstm/` with metrics, curves, confusion matrix
- `models/iter1_lstm.pt`

---

## Role

You are a PyTorch engineer building the first deep learning model for prompt injection detection.

## Grounding

Use **superpowers** to read:
1. `PRD.md` Section 2.4 Iteration 1 for exact architecture code.
2. `src/training/train.py` for the training interface.
3. `src/evaluation/metrics.py` for evaluation interface.

Use **goodmem** to read `data.vocab_size`, `foundation.training_ready`, `foundation.eval_ready`, `baselines.lr_f1_single` (target to beat).

## Task

Implement `SingleTurnLSTM(nn.Module)` per PRD.

**Completion criteria:**
- Architecture: Embedding(vocab_size, 128) → LSTM(128, 64) → Linear(64, 32) → ReLU → Linear(32, 1) → sigmoid
- `encode(x)` method returns pre-sigmoid hidden representation (batch, 32) for multi-turn downstream
- Training: epochs=20, batch_size=64, Adam(lr=0.001), BCELoss, patience=3
- Print parameter count
- Save training curves, confusion matrix, metrics.json
- Compare F1 against baselines

## Plugin Usage

**context7:** Verify `nn.LSTM` API for `batch_first=True` and hidden state extraction.

**superpowers:** Train the model, run evaluation.

**ralph-loop:**
1. Generate `src/models/single_turn.py` with `SingleTurnLSTM`
2. Build model, verify forward pass shape: input (batch, 256) → output (batch, 1)
3. Verify `encode()` returns (batch, 32)
4. Full training run
5. Review: F1 competitive with baselines? Training curves show convergence? No NaN losses?
6. Fix any issues, confirm

**goodmem:** After completion, persist:
- `models.iter1_f1 = <val>`
- `models.iter1_params = <count>`
- `models.iter1_path = models/iter1_lstm.pt`

## Verification

```bash
cat results/iter1_lstm/metrics.json
ls results/iter1_lstm/*.png
```

**Execution:** `claude --prompt prompts/11_lstm_simple.md --ultrathink`
