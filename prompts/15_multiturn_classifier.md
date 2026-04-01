# 15: Multi-Turn Sequence Classifier (Iteration 5 — NOVEL)

**Description:** Build the dual-encoder multi-turn classifier. This is the novel contribution.

**GitHub Issues:** #16 — [Models] Iteration 5: Multi-turn sequence classifier (NOVEL)

**Prerequisites:** Prompts 05 (synthetic data), 06 (MultiTurnDataset), 14 (encoder selection)

**Expected Outputs:**
- `src/models/multi_turn.py` with `MultiTurnClassifier`
- `results/iter5_multiturn/` with metrics
- `models/iter5_multiturn.pt`

---

## Prompt

You are a security ML researcher building a novel multi-turn injection detector.

<investigate_before_answering>
1. Read `PRD.md` Section 2.4 Iteration 5 for full architecture code and rationale.
2. Read `PRD.md` Section 1.3 for temporal justification (forget/update/output gate mapping).
3. Read the encoder decision from agent memory (Iteration 4 result).
4. Read `src/models/single_turn.py` for the encode() method.
5. Read `src/data/loader.py` for MultiTurnDataset interface.
</investigate_before_answering>

### Task

Implement `MultiTurnClassifier(nn.Module)` per PRD.

**Completion criteria:**
- Frozen turn encoder from best single-turn model
- Architecture: encode each turn -> stack -> sequence_lstm -> dropout -> FC -> sigmoid
- Input: (batch, 10, 256) with mask (batch, 10)
- Training: epochs=30, batch_size=32, patience=5, synthetic multi-turn data
- Evaluate on multi-turn test AND compare against baselines and single-turn applied turn-by-turn
- The F1 gap is the core finding. Document it.

### Tool Guidance

- **Sequential-thinking MCP:** This is the most complex model. Think through data flow: batch of conversations -> encode each turn independently -> stack turn embeddings -> sequence LSTM processes turn vectors over time -> classify. Verify shapes at each step.
- **Serena (session memory):** This may span multiple turns. Checkpoint after: (1) model builds without error, (2) forward pass produces correct output shape, (3) training completes, (4) evaluation done.
- **Ralph loops:** Build, forward pass shape check, train, evaluate. At each stage, verify before proceeding.
- **Goodman plugin:** Persist multi-turn F1 and the gap vs single-turn. This is the headline result.

**Execution:** `claude --prompt prompts/15_multiturn_classifier.md --ultrathink`
