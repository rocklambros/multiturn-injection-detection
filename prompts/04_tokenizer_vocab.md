# 04: Build Vocabulary and Tokenizer

**Description:** Build custom vocabulary from training data, implement encode functions for single-turn and multi-turn.

**GitHub Issues:** #6 — [Data Pipeline] Build vocabulary and tokenizer

**Prerequisites:** Prompt 03 (cleaned training data must exist)

**Expected Outputs:**
- `src/utils/tokenizer.py`
- `models/vocab.json`

---

## Prompt

You are an NLP engineer building the tokenization pipeline.

<investigate_before_answering>
1. Read `PRD.md` Section 3.4 (Tokenization) for config and function signatures.
2. Read `data/processed/single_turn_train.csv` to understand text format.
3. Read `src/utils/config.py` for sequence length and vocab size constants.
</investigate_before_answering>

### Task

Write `src/utils/tokenizer.py` with:
- `build_vocab(train_texts, max_vocab_size=20000) -> dict`
- `encode_texts(vocab, texts, max_len=256) -> torch.LongTensor`
- `encode_multiturn(vocab, turns_list, max_turns=10, max_len=256) -> torch.LongTensor`

**Completion criteria:**
- Special tokens: PAD at index 0, OOV at index 1
- Vocab fitted on training data ONLY
- Saved to `models/vocab.json`
- Print: vocab size, OOV rate on val set, sample encoded sequence

### Tool Guidance

- **Goodman plugin:** Persist vocab size to agent memory. Downstream model prompts need this.
- **Ralph loops:** Build vocab, encode a sample batch, verify shapes match PRD Section 2.3.

### Verification

```bash
python -c "
from src.utils.tokenizer import build_vocab, encode_texts
import pandas as pd
train = pd.read_csv('data/processed/single_turn_train.csv')
vocab = build_vocab(train['text'].tolist())
print(f'Vocab size: {len(vocab)}')
encoded = encode_texts(vocab, train['text'][:5].tolist())
print(f'Encoded shape: {encoded.shape}')
"
```

**Execution:** `claude --prompt prompts/04_tokenizer_vocab.md --ultrathink`
