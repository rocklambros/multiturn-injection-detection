# 13: Bidirectional LSTM with Dropout (Iteration 3)

**Description:** BiLSTM with two dropout experiments (0.3 and 0.5).

**GitHub Issues:** #14 — [Models] Iteration 3: BiLSTM with dropout comparison

**Prerequisites:** Prompt 11 (Iteration 1 training integration)

**Expected Outputs:**
- `BiLSTMClassifier` in `src/models/single_turn.py`
- `results/iter3_bilstm_d03/` and `results/iter3_bilstm_d05/`
- `models/iter3_bilstm_d03.pt` and `models/iter3_bilstm_d05.pt`

---

## Role

You are a deep learning engineer running regularization experiments.

## Grounding

Use **superpowers** to read:
1. `PRD.md` Section 2.4 Iteration 3 for architecture.
2. `src/models/single_turn.py` for existing patterns.

Use **goodmem** to read `models.iter1_f1`, `models.iter2_f1`.

## Task

Implement `BiLSTMClassifier` per PRD. Train twice: dropout=0.3 and dropout=0.5.

**Completion criteria:**
- Bidirectional LSTM, concat forward/backward: hidden_dim * 2
- Dropout after FC1 and before FC2
- epochs=30, patience=5
- Compare overfitting (train/val gap) between dropout rates
- Compare F1 against Iterations 1-2

## Plugin Usage

**Dispatch subagents** for the two parallel training runs:
- Subagent A: Train with dropout=0.3
- Subagent B: Train with dropout=0.5
- Then compare results.

**superpowers:** Execute training runs.

**ralph-loop:** For each variant:
1. Build model, verify forward pass shape
2. Full training
3. Review: train/val gap (overfitting signal), F1, convergence
4. Fix if needed
5. Confirm

**goodmem:** After completion, persist:
- `models.iter3_f1_d03 = <val>`
- `models.iter3_f1_d05 = <val>`
- `models.iter3_best_dropout = <0.3 or 0.5>`
- `models.iter3_overfitting_analysis = <summary>`

**Execution:** `claude --prompt prompts/13_bilstm_dropout.md --ultrathink`
