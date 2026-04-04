# 06: PyTorch Dataset and DataLoader Classes

**Description:** Implement Dataset and DataLoader for single-turn and multi-turn data pipelines.

**GitHub Issues:** #7 — [Data Pipeline] PyTorch Dataset and DataLoader classes

**Prerequisites:** Prompt 04 complete (tokenizer exists)

**Expected Outputs:**
- `src/data/loader.py`

---

## Role

You are a PyTorch engineer building the data loading pipeline.

## Grounding

Use **superpowers** to read:
1. `PRD.md` Section 3.5 (Data Loader) for class signatures.
2. `PRD.md` Section 2.3 for tensor shapes.
3. `src/utils/tokenizer.py` for encode function signatures.

Use **goodmem** to read `data.vocab_size`, `data.max_sequence_length`.

## Task

Write `src/data/loader.py` with:
- `SingleTurnDataset(Dataset)` returning (token_ids, label)
- `MultiTurnDataset(Dataset)` returning (turn_token_ids, mask, label)
- `create_single_turn_loaders(...)` and `create_multi_turn_loaders(...)` factory functions

**Completion criteria:**
- Multi-turn mask: 1=real turn, 0=padding
- Train: shuffle=True. Val/Test: shuffle=False
- All: num_workers=2, pin_memory=True
- Print batch shapes after first iteration

## Plugin Usage

**context7:** Verify PyTorch `Dataset`, `DataLoader` API for `pin_memory` and `num_workers`.

**superpowers:** Run shape verification.

**ralph-loop:**
1. Generate `src/data/loader.py`
2. Create a DataLoader with dummy data, iterate one batch
3. Review: Shapes match PRD Section 2.3? Single-turn: (batch, 256). Multi-turn: (batch, 10, 256) with mask (batch, 10).
4. Fix any shape mismatches
5. Confirm pass

**goodmem:** After completion, persist:
- `data.loader_ready = true`
- `data.single_turn_shape = (batch, 256)`
- `data.multi_turn_shape = (batch, 10, 256)`

## Verification

```bash
python -c "
import torch
from src.data.loader import SingleTurnDataset, MultiTurnDataset
ds = SingleTurnDataset(torch.randint(0, 100, (50, 256)), torch.randint(0, 2, (50,)))
print(f'SingleTurn sample: {ds[0][0].shape}, {ds[0][1].shape}')
"
```

**Execution:** `claude --prompt prompts/06_dataset_dataloader.md --ultrathink`
