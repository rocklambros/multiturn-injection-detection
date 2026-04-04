# 12: LSTM with GloVe Embeddings (Iteration 2)

**Description:** Replace random embeddings with pretrained GloVe 6B 100d.

**GitHub Issues:** #13 — [Models] Iteration 2: LSTM with pretrained GloVe embeddings

**Prerequisites:** Prompts 07 (GloVe matrix), 11 (Iteration 1 model code)

**Expected Outputs:**
- Updated `src/models/single_turn.py` supporting pretrained embeddings
- `results/iter2_glove/` with metrics and curves
- `models/iter2_glove.pt`

---

## Role

You are an NLP engineer comparing random vs. pretrained embeddings for security text classification.

## Grounding

Use **superpowers** to read:
1. `PRD.md` Section 2.4 Iteration 2.
2. `src/models/single_turn.py` for current SingleTurnLSTM.
3. `data/embeddings/embedding_matrix.npy` shape.

Use **goodmem** to read `models.iter1_f1`, `data.glove_coverage`, `data.glove_missing_security_terms`, `data.embedding_dim`.

## Task

Modify SingleTurnLSTM to accept pretrained GloVe via `nn.Embedding.from_pretrained(matrix, freeze=True)`.

**Completion criteria:**
- Embedding dim = 100 (GloVe), LSTM input_size adjusted
- Same training config as Iteration 1
- Compare: convergence speed, final F1, training curves side-by-side
- Report security vocabulary coverage

## Plugin Usage

**superpowers:** Train and evaluate.

**ralph-loop:**
1. Modify model to accept pretrained embeddings
2. Train with GloVe
3. Review: F1 vs Iteration 1? Convergence faster or slower? Embeddings actually frozen?
4. Fix if needed
5. Confirm

**goodmem:** After completion, persist:
- `models.iter2_f1 = <val>`
- `models.iter2_convergence_epochs = <N>`
- `models.glove_vs_random = <better/worse/comparable>`

**Execution:** `claude --prompt prompts/12_lstm_glove.md --ultrathink`
