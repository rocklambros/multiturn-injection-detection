# 06: PyTorch Dataset and DataLoader Classes

**Description:** Implement Dataset and DataLoader for single-turn and multi-turn data pipelines.

**GitHub Issues:** #7 — [Data Pipeline] PyTorch Dataset and DataLoader classes

**Prerequisites:** Prompt 04 (tokenizer must exist)

**Expected Outputs:**
- `src/data/loader.py`

---

## Prompt

You are a PyTorch engineer building the data loading pipeline.

<investigate_before_answering>
1. Read `PRD.md` Section 3.5 (Data Loader) for class signatures and requirements.
2. Read `PRD.md` Section 2.3 for tensor shapes.
3. Read `src/utils/tokenizer.py` for encode functions.
4. Read `src/utils/config.py` for batch sizes.
</investigate_before_answering>

### Task

Write `src/data/loader.py` with:
- `SingleTurnDataset(Dataset)` returning (token_ids, label)
- `MultiTurnDataset(Dataset)` returning (turn_token_ids, mask, label)
- Factory functions for DataLoaders

**Completion criteria:**
- Multi-turn mask: 1=real turn, 0=padding
- Train: shuffle=True. Val/Test: shuffle=False
- All: num_workers=2, pin_memory=True
- Print batch shapes after first iteration

### Tool Guidance

- **Context7 plugin:** Verify PyTorch Dataset/DataLoader API for pin_memory and num_workers usage.
- **Ralph loops:** Create a DataLoader, iterate one batch, verify shapes match PRD Section 2.3.

### Verification

```bash
python -c "
from src.data.loader import SingleTurnDataset
import torch
# Quick shape check with dummy data
ds = SingleTurnDataset(torch.randint(0, 100, (50, 256)), torch.randint(0, 2, (50,)))
print(f'Dataset length: {len(ds)}')
print(f'Sample shapes: {ds[0][0].shape}, {ds[0][1].shape}')
"
```

**Execution:** `claude --prompt prompts/06_dataset_dataloader.md --ultrathink`
