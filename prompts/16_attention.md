# 16: Attention Mechanism with Heatmaps (Iteration 6)

**Description:** Add attention to multi-turn classifier. Visualize which turns the model focuses on.

**GitHub Issues:** #17 — [Models] Iteration 6: Attention mechanism with heatmap visualization

**Prerequisites:** Prompt 15 (multi-turn classifier)

**Expected Outputs:**
- `src/models/attention.py` with `TurnAttention`
- `results/iter6_attention/` with metrics and heatmap PNGs
- `models/iter6_attention.pt`

---

## Role

You are a deep learning researcher adding interpretability to the multi-turn detector. The attention weights should tell a story: for correctly detected attacks, which turns did the model focus on?

## Grounding

Use **superpowers** to read:
1. `PRD.md` Section 2.4 Iteration 6 for attention architecture.
2. `src/models/multi_turn.py` for current architecture.
3. `src/evaluation/visualization.py` for heatmap functions.

Use **goodmem** to read `models.iter5_multiturn_f1` as comparison target.

## Task

Implement `TurnAttention` and integrate with multi-turn classifier.

**Completion criteria:**
- Attention: W(hidden, 32) → tanh → V(32, 1) → masked softmax
- Returns context vector AND attention_weights
- Modified multi-turn model uses full lstm_out with attention pooling
- Heatmaps for correctly classified attacks and misclassified sequences
- Compare F1 vs Iteration 5

## Plugin Usage

**superpowers:** Train and generate heatmaps.

**ralph-loop:**
1. Generate `src/models/attention.py`
2. Integrate with multi-turn classifier
3. Train, evaluate
4. Generate heatmaps for 10 correctly classified attacks, 5 misclassified
5. Review: Do attention weights concentrate on escalation turns? Do heatmaps tell a coherent story?
6. Fix if needed
7. Confirm

**goodmem:** After completion, persist:
- `models.iter6_attention_f1 = <val>`
- `models.attention_pattern = <description: e.g., "concentrates on turns 3-4 in escalation sequences">`
- `models.attention_vs_iter5 = <better/worse/comparable>`

**Execution:** `claude --prompt prompts/16_attention.md --ultrathink`
