# 14: GRU Comparison (Iteration 4)

**Description:** Replace LSTM with GRU, compare, select encoder for multi-turn classifier.

**GitHub Issues:** #15 — [Models] Iteration 4: GRU comparison and encoder selection

**Prerequisites:** Prompt 13 (Iteration 3 provides comparison target)

**Expected Outputs:**
- GRU variant in `src/models/single_turn.py`
- `results/iter4_gru/` with metrics
- Documented encoder decision for Iteration 5

---

## Role

You are a deep learning engineer making a critical architecture selection that affects the rest of the project.

## Grounding

Use **superpowers** to read:
1. `PRD.md` Section 2.4 Iteration 4.
2. `src/models/single_turn.py` for BiLSTMClassifier base.
3. `results/iter3_bilstm_*/metrics.json` for comparison targets.

Use **goodmem** to read `models.iter3_best_dropout`, `models.iter3_f1_d03`, `models.iter3_f1_d05`.

## Task

Replace nn.LSTM with nn.GRU. Compare parameter count, training time, F1. Make the encoder decision.

**Completion criteria:**
- Parameter count comparison printed
- Training time per epoch comparison
- F1 comparison
- **Decision documented:** LSTM or GRU for Iteration 5 multi-turn model, with reasoning

## Plugin Usage

**superpowers:** Train GRU variant, measure wall-clock time.

**ralph-loop:**
1. Build GRU model
2. Train, measure time and F1
3. Review: Compare against best Iteration 3 variant on params, speed, F1
4. Make encoder decision based on evidence
5. Confirm

**goodmem:** CRITICAL — downstream prompts depend on this. Persist:
- `models.iter4_f1 = <val>`
- `models.iter4_params = <count>`
- `models.encoder_decision = <LSTM or GRU>`
- `models.encoder_decision_reasoning = <text>`
- `models.best_single_turn_iteration = <1, 2, 3, or 4>`
- `models.best_single_turn_f1 = <val>`
- `models.best_single_turn_path = models/iter<N>_*.pt`

**serena:** Checkpoint — the encoder decision is the single most important architectural choice for Phase E.

**Execution:** `claude --prompt prompts/14_gru_comparison.md --ultrathink`
