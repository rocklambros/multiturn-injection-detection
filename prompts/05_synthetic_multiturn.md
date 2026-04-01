# 05: Synthetic Multi-Turn Conversation Generator

**Description:** Build four attack distribution strategies plus balanced benign sequences for multi-turn training data.

**GitHub Issues:** #5 — [Data Pipeline] Synthetic multi-turn conversation generator (4 strategies)

**Prerequisites:** Prompt 03 (cleaned single-turn data as source material)

**Expected Outputs:**
- `src/data/synthetic.py`
- `data/synthetic/multiturn_{train,val,test}.json`

---

## Prompt

You are a security researcher building synthetic multi-turn prompt injection datasets.

<investigate_before_answering>
1. Read `PRD.md` Section 3.3 (Synthetic Multi-Turn Sequence Generation) for all four strategies, JSON schema, and parameters.
2. Read `data/processed/single_turn_train.csv` to understand available injection and benign text.
3. Review the Crescendo paper reference in PRD Section 1.5 for the gradual escalation pattern.
</investigate_before_answering>

### Task

Write `src/data/synthetic.py` implementing all four generation strategies:
- Fragment distribution (40%)
- Gradual escalation (30%) — Crescendo pattern
- Context priming (20%)
- Instruction layering (10%)

**Completion criteria:**
- JSON schema matches PRD Section 3.3 exactly
- Parameters: 3-10 turns/sequence, train=5000, val=1000, test=1000, 50/50 balance, seed=42
- Pool of 500+ unique benign filler turns, max 3 reuses each
- Uses nltk.sent_tokenize() for fragment splitting
- Prints stats and validates 40 random samples

### Tool Guidance

- **Sequential-thinking MCP:** Before coding, think through how each strategy works. Fragment distribution needs sentence splitting. Gradual escalation needs templates that build specificity. Context priming needs persona establishment. Plan the approach before writing.
- **Ralph loops:** Generate data, validate 40 samples manually, check strategy distribution matches targets.
- **Goodman plugin:** Persist sequence counts and strategy distribution to agent memory.

### Verification

```bash
python src/data/synthetic.py
python -c "
import json
with open('data/synthetic/multiturn_train.json') as f:
    data = json.load(f)
print(f'Train sequences: {len(data)}')
print(f'Label distribution: {sum(1 for d in data if d["label"]==1)} attack, {sum(1 for d in data if d["label"]==0)} benign')
"
```

**Execution:** `claude --prompt prompts/05_synthetic_multiturn.md --ultrathink`
