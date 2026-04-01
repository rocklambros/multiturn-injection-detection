# 11: Simple LSTM (Iteration 1)

**Description:** First deep learning model — SingleTurnLSTM with random embeddings.

**GitHub Issues:** #12 — [Models] Iteration 1: Simple LSTM with random embeddings

**Prerequisites:** Prompts 06 (DataLoader), 09 (training loop)

**Expected Outputs:**
- `src/models/single_turn.py` with `SingleTurnLSTM` class
- `results/iter1_lstm/` with metrics, curves, confusion matrix
- `models/iter1_lstm.pt`

---

## Prompt

You are a PyTorch engineer building the first deep learning model.

<investigate_before_answering>
1. Read `PRD.md` Section 2.4 Iteration 1 for exact architecture and code.
2. Read `src/training/train.py` for the training interface.
3. Read `src/evaluation/metrics.py` for evaluation interface.
4. Read `models/vocab.json` for vocab_size.
</investigate_before_answering>

### Task

Implement `SingleTurnLSTM(nn.Module)` per PRD.

**Completion criteria:**
- Architecture: Embedding(vocab_size, 128) -> LSTM(128, 64) -> Linear(64, 32) -> ReLU -> Linear(32, 1) -> sigmoid
- `encode(x)` method returns (batch, 32) for multi-turn downstream use
- Training: epochs=20, batch_size=64, Adam(lr=0.001), BCELoss, patience=3
- Print parameter count
- Save all metrics and compare F1 against baselines

### Tool Guidance

- **Sequential-thinking MCP:** Before training, verify tensor shapes through the forward pass mentally. Embedding -> LSTM -> hidden extraction -> FC layers.
- **Ralph loops:** Build model, run 1 epoch, check loss decreases. Full train, evaluate, verify metrics.
- **Goodman plugin:** Persist Iteration 1 F1 and parameter count.

**Execution:** `claude --prompt prompts/11_lstm_simple.md --ultrathink`
