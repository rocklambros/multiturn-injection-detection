# 15: Multi-Turn Sequence Classifier (Iteration 5 — NOVEL)

**Description:** Build the dual-encoder multi-turn classifier. This is the novel contribution of the project.

**GitHub Issues:** #16 — [Models] Iteration 5: Multi-turn sequence classifier (NOVEL)

**Prerequisites:** Prompts 05 (synthetic data), 06 (MultiTurnDataset), 14 (encoder decision)

**Expected Outputs:**
- `src/models/multi_turn.py` with `MultiTurnClassifier`
- `results/iter5_multiturn/` with metrics
- `models/iter5_multiturn.pt`

---

## Role

You are a security ML researcher building a novel multi-turn injection detector. This is the centerpiece of the project: a dual-encoder architecture where a frozen single-turn encoder produces per-turn embeddings, and a sequence-level LSTM processes the turn sequence to detect distributed attacks.

## Grounding

Use **superpowers** to read:
1. `PRD.md` Section 2.4 Iteration 5 — full architecture code and rationale.
2. `PRD.md` Section 1.3 — temporal justification (gate mapping).
3. `src/models/single_turn.py` for the `encode()` method.
4. `src/data/loader.py` for MultiTurnDataset interface.

Use **goodmem** to read:
- `models.encoder_decision` — LSTM or GRU?
- `models.best_single_turn_path` — which checkpoint to load as turn encoder?
- `data.multiturn_train`, `data.multiturn_val`, `data.multiturn_test`
- `baselines.lr_f1_multiturn` — the floor to beat

## Task

Implement `MultiTurnClassifier(nn.Module)` per PRD.

**Completion criteria:**
- Frozen turn encoder from best single-turn model
- Architecture: encode each turn → stack → sequence_lstm(32, 64) → dropout(0.3) → FC(64,32) → FC(32,1) → sigmoid
- Input: (batch, 10, 256) with mask (batch, 10)
- Training: epochs=30, batch_size=32, patience=5, synthetic multi-turn data
- Evaluate on multi-turn test AND compare against baselines and single-turn applied turn-by-turn
- **The F1 gap is the core finding. Document it prominently.**

## Plugin Usage

**superpowers:** Load pretrained encoder, build multi-turn model, train, evaluate.

**ralph-loop:**
1. Generate `src/models/multi_turn.py`
2. Build model. Verify forward pass: (batch, 10, 256) + mask → (batch, 1)
3. Verify turn encoder is frozen: `all(not p.requires_grad for p in model.turn_encoder.parameters())`
4. Full training run
5. Evaluate on multi-turn test set. Compare F1 vs baselines and single-turn-per-turn.
6. Review: Is there a meaningful F1 gap? If yes, document it. If no, document why.
7. Confirm

**goodmem:** CRITICAL — this is the headline result. Persist:
- `models.iter5_multiturn_f1 = <val>`
- `models.iter5_multiturn_precision = <val>`
- `models.iter5_multiturn_recall = <val>`
- `models.core_finding_f1_gap = <multiturn F1 minus best single-turn-per-turn F1>`
- `models.core_finding_summary = <1-2 sentence description>`

**serena:** Checkpoint with full results — this is the most important single step in the project.

**Execution:** `claude --prompt prompts/15_multiturn_classifier.md --ultrathink`
