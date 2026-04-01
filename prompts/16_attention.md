# 16: Attention Mechanism with Heatmaps (Iteration 6)

**Description:** Add attention to multi-turn classifier. Visualize which turns the model focuses on.

**GitHub Issues:** #17 — [Models] Iteration 6: Attention mechanism with heatmap visualization

**Prerequisites:** Prompt 15 (multi-turn classifier)

**Expected Outputs:**
- `src/models/attention.py` with `TurnAttention`
- `results/iter6_attention/` with metrics and heatmap PNGs
- `models/iter6_attention.pt`

---

## Prompt

You are a deep learning researcher adding interpretability to the multi-turn detector.

<investigate_before_answering>
1. Read `PRD.md` Section 2.4 Iteration 6 for attention architecture code.
2. Read `src/models/multi_turn.py` for current architecture.
3. Read `src/evaluation/visualization.py` for heatmap functions.
</investigate_before_answering>

### Task

Implement `TurnAttention` and integrate with multi-turn classifier.

**Completion criteria:**
- Attention: W(hidden, 32) -> tanh -> V(32, 1) -> masked softmax
- Returns context vector AND attention_weights
- Modified multi-turn model uses full lstm_out with attention pooling
- Heatmaps for correctly classified attacks and misclassified sequences
- Compare F1 vs Iteration 5

### Tool Guidance

- **Sequential-thinking MCP:** Think about what attention should reveal. For fragment-distributed attacks, weights should concentrate on turns containing fragments. For escalation, weights should increase toward later turns.
- **Ralph loops:** Build, train, generate heatmaps, verify they show meaningful patterns.

**Execution:** `claude --prompt prompts/16_attention.md --ultrathink`
