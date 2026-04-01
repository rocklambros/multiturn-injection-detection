# 07: GloVe Download and Embedding Matrix

**Description:** Download GloVe 6B 100d and build embedding matrix aligned with project vocabulary.

**GitHub Issues:** #8 — [Data Pipeline] GloVe download and embedding matrix

**Prerequisites:** Prompt 04 (vocabulary must exist at models/vocab.json)

**Expected Outputs:**
- `src/data/download_glove.py` (or extension of download.py)
- `data/embeddings/glove.6B.100d.txt`
- `data/embeddings/embedding_matrix.npy`

---

## Prompt

You are an NLP engineer building pretrained embedding integration.

<investigate_before_answering>
1. Read `PRD.md` Section 2.4 Iteration 2 for GloVe usage requirements.
2. Read `models/vocab.json` to understand vocab size and word-to-index mapping.
3. Check `.gitignore` covers data/embeddings/.
</investigate_before_answering>

### Task

Download GloVe 6B, extract 100d vectors, build embedding matrix.

**Completion criteria:**
- Matrix shape: (vocab_size, 100)
- Row 0 (PAD) = zeros, Row 1 (OOV) = mean of all GloVe vectors
- Print coverage stats and notable missing security terms
- Saved to `data/embeddings/embedding_matrix.npy`

### Tool Guidance

- **Ralph loops:** Download, build matrix, verify shape, check coverage of security terms ("ignore", "override", "jailbreak", "bypass").

### Verification

```bash
python -c "
import numpy as np
matrix = np.load('data/embeddings/embedding_matrix.npy')
print(f'Embedding matrix shape: {matrix.shape}')
print(f'PAD row (should be zeros): {matrix[0].sum()}')
"
```

**Execution:** `claude --prompt prompts/07_glove_embeddings.md --ultrathink`
