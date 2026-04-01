# 13: Bidirectional LSTM with Dropout (Iteration 3)

**Description:** BiLSTM with two dropout experiments (0.3 and 0.5).

**GitHub Issues:** #14 — [Models] Iteration 3: BiLSTM with dropout comparison

**Prerequisites:** Prompt 11 (Iteration 1 training integration)

**Expected Outputs:**
- `BiLSTMClassifier` in `src/models/single_turn.py`
- `results/iter3_bilstm_d03/` and `results/iter3_bilstm_d05/`
- `models/iter3_bilstm_d03.pt` and `models/iter3_bilstm_d05.pt`

---

## Prompt

You are a deep learning engineer running regularization experiments.

<investigate_before_answering>
1. Read `PRD.md` Section 2.4 Iteration 3 for BiLSTM architecture code.
2. Read `src/models/single_turn.py` for existing model patterns.
3. Read course lecture W5 material on dropout regularization.
</investigate_before_answering>

### Task

Implement `BiLSTMClassifier` per PRD. Train twice with dropout=0.3 and dropout=0.5.

**Completion criteria:**
- Bidirectional LSTM, concatenate forward/backward hidden: hidden_dim * 2
- Dropout after FC1 and before FC2
- epochs=30, patience=5
- Compare overfitting (train/val gap) between dropout rates
- Compare F1 against Iterations 1-2

### Tool Guidance

- **Sequential-thinking MCP:** Think about why bidirectional helps for injection detection. Forward context captures prefix patterns, backward captures suffix patterns.
- **Ralph loops:** Train both variants, compare, select better dropout rate.
- **Goodman plugin:** Persist best Iteration 3 config and F1.

**Execution:** `claude --prompt prompts/13_bilstm_dropout.md --ultrathink`
