# 14: GRU Comparison (Iteration 4)

**Description:** Replace LSTM with GRU, compare, select encoder for multi-turn classifier.

**GitHub Issues:** #15 — [Models] Iteration 4: GRU comparison and encoder selection

**Prerequisites:** Prompt 13 (Iteration 3 provides comparison target)

**Expected Outputs:**
- GRU variant in `src/models/single_turn.py`
- `results/iter4_gru/` with metrics
- Documented decision: LSTM or GRU for turn encoder

---

## Prompt

You are a deep learning engineer making an architecture selection decision.

<investigate_before_answering>
1. Read `PRD.md` Section 2.4 Iteration 4.
2. Read `src/models/single_turn.py` for BiLSTMClassifier to base GRU variant on.
3. Review Iteration 3 results in `results/iter3_bilstm_*/metrics.json`.
</investigate_before_answering>

### Task

Replace nn.LSTM with nn.GRU. Same architecture otherwise. Compare parameter count, training time, F1.

**Completion criteria:**
- Parameter count comparison printed
- Training time per epoch comparison
- F1 comparison
- **Decision documented:** which encoder (LSTM or GRU) to use in Iteration 5 multi-turn model, and why

### Tool Guidance

- **Sequential-thinking MCP:** Think through LSTM vs GRU tradeoffs before deciding. GRU has fewer parameters (no separate cell state), trains faster, but LSTM has richer memory. For security context that needs to persist across many turns, which matters more?
- **Goodman plugin:** Persist the encoder decision to agent memory. Iteration 5 depends on this choice.

**Execution:** `claude --prompt prompts/14_gru_comparison.md --ultrathink`
