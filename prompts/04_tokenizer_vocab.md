# 04: Build Vocabulary and Tokenizer

**Description:** Build custom vocabulary from training data, implement encode functions for single-turn and multi-turn.

**GitHub Issues:** #6 — [Data Pipeline] Build vocabulary and tokenizer

**Prerequisites:** Prompt 03 complete (cleaned training data exists)

**Expected Outputs:**
- `src/utils/tokenizer.py`
- `models/vocab.json`

---

## Role

You are an NLP engineer building the tokenization pipeline.

## Grounding

Use **superpowers** to read:
1. `PRD.md` Section 3.4 (Tokenization) for config and function signatures.
2. First 10 rows of `data/processed/single_turn_train.csv` to understand text format.

Use **goodmem** to read `data.train_samples` to confirm data exists.

## Task

Write `src/utils/tokenizer.py` with:
- `build_vocab(train_texts, max_vocab_size=20000) -> dict`
- `encode_texts(vocab, texts, max_len=256) -> torch.LongTensor`
- `encode_multiturn(vocab, turns_list, max_turns=10, max_len=256) -> torch.LongTensor`
- `save_vocab(vocab, path)` and `load_vocab(path)` for JSON serialization

**Completion criteria:**
- Special tokens: PAD at index 0, OOV at index 1
- Vocabulary fitted on training data ONLY (no data leakage)
- Saved to `models/vocab.json`
- Print: vocab size, OOV rate on val set, sample encoded sequence

## Plugin Usage

**context7:** Look up `torch.LongTensor` and padding conventions to confirm API.

**superpowers:** Run tokenizer build and verify output files.

**ralph-loop:**
1. Generate `src/utils/tokenizer.py`
2. Execute build_vocab on training data
3. Review: Vocab size reasonable? OOV rate < 5%? Encoded shapes correct?
4. Fix any issues
5. Confirm vocab.json saved and loadable

**goodmem:** After completion, persist:
- `data.vocab_size = <N>`
- `data.oov_rate_val = <pct>`
- `data.vocab_path = models/vocab.json`
- `data.max_sequence_length = 256`

## Verification

```bash
python -c "
from src.utils.tokenizer import build_vocab, encode_texts, load_vocab
import pandas as pd
train = pd.read_csv('data/processed/single_turn_train.csv')
vocab = build_vocab(train['text'].tolist())
print(f'Vocab size: {len(vocab)}')
encoded = encode_texts(vocab, train['text'][:5].tolist())
print(f'Encoded shape: {encoded.shape}')
"
```

**Execution:** `claude --prompt prompts/04_tokenizer_vocab.md --ultrathink`
