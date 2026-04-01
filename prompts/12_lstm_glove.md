# 12: LSTM with GloVe Embeddings (Iteration 2)

**Description:** Replace random embeddings with pretrained GloVe 6B 100d.

**GitHub Issues:** #13 — [Models] Iteration 2: LSTM with pretrained GloVe embeddings

**Prerequisites:** Prompts 07 (GloVe matrix), 11 (Iteration 1 model code)

**Expected Outputs:**
- Updated `src/models/single_turn.py` supporting pretrained embeddings
- `results/iter2_glove/` with metrics and curves
- `models/iter2_glove.pt`

---

## Prompt

You are an NLP engineer comparing random vs. pretrained embeddings.

<investigate_before_answering>
1. Read `PRD.md` Section 2.4 Iteration 2.
2. Read `src/models/single_turn.py` for current SingleTurnLSTM implementation.
3. Read `data/embeddings/embedding_matrix.npy` shape.
</investigate_before_answering>

### Task

Modify SingleTurnLSTM to accept pretrained GloVe embeddings via `nn.Embedding.from_pretrained(matrix, freeze=True)`.

**Completion criteria:**
- Embedding dim changes to 100 (GloVe). LSTM input_size adjusted.
- Same training config as Iteration 1
- Compare: convergence speed, final F1, training curves side-by-side
- Report security vocabulary coverage in GloVe

### Tool Guidance

- **Ralph loops:** Train, compare F1 to Iteration 1, verify GloVe embeddings are frozen (no grad updates).
- **Goodman plugin:** Persist Iteration 2 F1 and GloVe coverage stats.

**Execution:** `claude --prompt prompts/12_lstm_glove.md --ultrathink`
