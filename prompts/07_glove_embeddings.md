# 07: GloVe Download and Embedding Matrix

**Description:** Download GloVe 6B 100d and build embedding matrix aligned with project vocabulary.

**GitHub Issues:** #8 — [Data Pipeline] GloVe download and embedding matrix

**Prerequisites:** Prompt 04 complete (vocabulary exists at models/vocab.json)

**Expected Outputs:**
- GloVe download script (extend `src/data/download.py` or create `src/data/download_glove.py`)
- `data/embeddings/glove.6B.100d.txt`
- `data/embeddings/embedding_matrix.npy`

---

## Role

You are an NLP engineer building pretrained embedding integration.

## Grounding

Use **superpowers** to read:
1. `PRD.md` Section 2.4 Iteration 2 for GloVe requirements.
2. `models/vocab.json` to get vocab size and word-to-index mapping.
3. Verify `.gitignore` covers `data/embeddings/`.

Use **goodmem** to read `data.vocab_size`.

## Task

Download GloVe 6B, extract 100d vectors, build aligned embedding matrix.

**Completion criteria:**
- Matrix shape: (vocab_size, 100)
- Row 0 (PAD) = zeros, Row 1 (OOV) = mean of all GloVe vectors
- Print coverage stats and notable missing security terms ("ignore", "override", "jailbreak", "bypass", "inject", "prompt")
- Saved to `data/embeddings/embedding_matrix.npy`

## Plugin Usage

**superpowers:** Download GloVe zip, extract, build matrix, save.

**ralph-loop:**
1. Generate the download and matrix-building code
2. Execute it
3. Review: Matrix shape correct? PAD row zeros? Coverage > 50%? Security terms checked?
4. Fix any issues
5. Confirm

**goodmem:** After completion, persist:
- `data.glove_coverage = <pct>`
- `data.glove_missing_security_terms = [<list>]`
- `data.embedding_matrix_path = data/embeddings/embedding_matrix.npy`
- `data.embedding_dim = 100`

## Verification

```bash
python -c "
import numpy as np
matrix = np.load('data/embeddings/embedding_matrix.npy')
print(f'Shape: {matrix.shape}')
print(f'PAD row sum (should be 0): {matrix[0].sum()}')
"
```

**Execution:** `claude --prompt prompts/07_glove_embeddings.md --ultrathink`
