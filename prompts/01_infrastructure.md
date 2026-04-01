# 01: Project Infrastructure Setup

**Description:** Create the full directory structure, requirements.txt, seed module, and config module.

**GitHub Issues:** #2 — [Infrastructure] Create project skeleton, requirements, seed, config

**Prerequisites:** None. This is the foundation.

**Expected Outputs:**
- All directories from PRD Section 7 exist
- All `__init__.py` files in every src/ subdirectory
- `requirements.txt` with pinned PyTorch deps
- `src/utils/seed.py` with `set_global_seed(42)`
- `src/utils/config.py` with `IterationConfig` dataclass and `ITERATIONS` dict
- `.gitignore` covering data/raw/, data/embeddings/, models/, results/, __pycache__/, .ipynb_checkpoints/

---

## Prompt

You are a Python infrastructure engineer setting up a PyTorch deep learning project.

<investigate_before_answering>
Before writing any code:
1. Read `PRD.md` completely — it is the source of truth for all file paths, configurations, and architecture decisions.
2. Read PRD Section 7 (Repository Structure) for exact directory layout.
3. Read PRD Section 8 (Configuration System) for the `IterationConfig` dataclass and full `ITERATIONS` dict.
4. Read PRD Section 9 (Reproducibility) for the `set_global_seed()` implementation.
5. Read PRD Section 10 (Dependencies) for `requirements.txt` contents.
</investigate_before_answering>

### Task

Create the complete project skeleton for the multi-turn injection detection project.

**Completion criteria:**
- `python -c "from src.utils.seed import set_global_seed; set_global_seed(42)"` runs without error
- `python -c "from src.utils.config import ITERATIONS; print(len(ITERATIONS))"` prints 10
- Every directory in PRD Section 7 exists
- `.gitignore` is correct

**Constraints:**
- PyTorch, not Keras/TensorFlow
- Python 3.10+
- Pin versions in requirements.txt

### Tool Guidance

- **Goodman plugin (agent memory):** Persist the directory structure and config schema to agent memory. Downstream prompts will reference these.
- **Ralph loops:** After creating all files, verify each import works. Fix any issues found before committing.

### Verification

```bash
python -c "from src.utils.seed import set_global_seed; set_global_seed(42); print('Seed OK')"
python -c "from src.utils.config import ITERATIONS; print(f'Iterations: {len(ITERATIONS)}')"
find src -name '__init__.py' | wc -l  # Should be >= 6
ls data/raw data/processed data/synthetic data/embeddings models results notebooks report prompts
```

**Execution:** `claude --prompt prompts/01_infrastructure.md --ultrathink`
