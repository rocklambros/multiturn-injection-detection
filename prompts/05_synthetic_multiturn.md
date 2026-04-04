# 05: Synthetic Multi-Turn Conversation Generator

**Description:** Build four attack distribution strategies plus balanced benign sequences for multi-turn training data.

**GitHub Issues:** #5 — [Data Pipeline] Synthetic multi-turn conversation generator (4 strategies)

**Prerequisites:** Prompt 03 complete (cleaned single-turn data as source material)

**Expected Outputs:**
- `src/data/synthetic.py`
- `data/synthetic/multiturn_{train,val,test}.json`

---

## Role

You are a security researcher building synthetic multi-turn prompt injection datasets. You understand the Crescendo attack pattern (gradual escalation across turns) and the Foot-in-the-Door technique (establishing compliance before the real ask).

## Grounding

Use **superpowers** to read:
1. `PRD.md` Section 3.3 — all four strategies, JSON schema, parameters.
2. `data/processed/single_turn_train.csv` — sample injection and benign texts to use as source material.

Use **goodmem** to read `data.train_samples` and `data.class_balance_train`.

## Task

Write `src/data/synthetic.py` implementing all four generation strategies:
- Fragment distribution (40%): split injection into 3-5 fragments, interleave with benign filler
- Gradual escalation (30%): Crescendo pattern, each turn adds specificity
- Context priming (20%): establish persona in early turns, exploit later
- Instruction layering (10%): each turn adds one constraint, cumulative override

**Completion criteria:**
- JSON schema matches PRD Section 3.3 exactly
- Parameters: 3-10 turns/sequence, train=5000, val=1000, test=1000, 50/50 balance, seed=42
- Pool of 500+ unique benign filler turns, max 3 reuses each
- Uses nltk.sent_tokenize() for fragment splitting
- Prints stats and validates 40 random samples

## Plugin Usage

**Dispatch subagents** for the two independent tracks:
- Subagent A: Build the benign filler pool (sample and deduplicate 500+ turns from benign training data)
- Subagent B: Build the four strategy generators
- Then combine and generate the full dataset.

**superpowers:** Run the generation script. Verify output JSON files.

**ralph-loop:**
1. Generate `src/data/synthetic.py`
2. Execute: `python src/data/synthetic.py`
3. Review: Check strategy distribution matches targets (±5%), turn count distribution, class balance, 40 sample validation output
4. Fix any imbalances or schema violations
5. Confirm all criteria pass

**goodmem:** After completion, persist:
- `data.multiturn_train = 5000`
- `data.multiturn_val = 1000`
- `data.multiturn_test = 1000`
- `data.strategy_distribution = {fragment: 40, escalation: 30, priming: 20, layering: 10}`
- `data.synthetic_path = data/synthetic/`

**serena:** Checkpoint — synthetic data is a critical dependency for Phase E (multi-turn models).

## Verification

```bash
python src/data/synthetic.py
python -c "
import json
with open('data/synthetic/multiturn_train.json') as f:
    data = json.load(f)
print(f'Train sequences: {len(data)}')
print(f'Attack: {sum(1 for d in data if d["label"]==1)}, Benign: {sum(1 for d in data if d["label"]==0)}')
"
```

**Execution:** `claude --prompt prompts/05_synthetic_multiturn.md --ultrathink`
