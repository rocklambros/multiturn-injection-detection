# 01: Project Infrastructure Setup

**Description:** Create the full directory structure, requirements.txt, seed module, and config module.

**GitHub Issues:** #2 — [Infrastructure] Create project skeleton, requirements, seed, config

**Prerequisites:** None. This is the foundation.

**Expected Outputs:**
- All directories from PRD Section 7
- All `__init__.py` files in every src/ subdirectory
- `requirements.txt` with pinned PyTorch deps
- `src/utils/seed.py` with `set_global_seed(42)`
- `src/utils/config.py` with `IterationConfig` dataclass and `ITERATIONS` dict
- `.gitignore` covering data/raw/, data/embeddings/, models/, results/, __pycache__/, .ipynb_checkpoints/

---

## Role

You are a Python infrastructure engineer setting up a PyTorch deep learning project.

## Grounding

Before writing any code, use **superpowers** to read these files:
1. `PRD.md` — source of truth. Read Sections 7 (repo structure), 8 (config), 9 (seed), 10 (deps).
2. Confirm the repo root exists and is writable.

## Task

Create the complete project skeleton for the multi-turn injection detection project.

**Completion criteria:**
- `python -c "from src.utils.seed import set_global_seed; set_global_seed(42)"` exits 0
- `python -c "from src.utils.config import ITERATIONS; print(len(ITERATIONS))"` prints 10
- Every directory in PRD Section 7 exists with `__init__.py` where appropriate
- `.gitignore` covers all generated/large artifacts

**Constraints:**
- PyTorch, not Keras/TensorFlow
- Python 3.10+
- Pin versions in requirements.txt

## Plugin Usage

**superpowers:** Use for all `mkdir -p`, file writes, and verification commands.

**ralph-loop:** After creating all files:
1. Generate the implementation
2. Run the two verification commands above
3. If either fails, read the error, fix the file, re-run
4. Confirm both pass before marking complete

**goodmem:** After completion, persist:
- `foundation.skeleton_complete = true`
- `foundation.directories = <list of created dirs>`
- `foundation.config_iterations_count = 10`

## Verification

```bash
python -c "from src.utils.seed import set_global_seed; set_global_seed(42); print('Seed OK')"
python -c "from src.utils.config import ITERATIONS; print(f'Iterations: {len(ITERATIONS)}')"
find src -name '__init__.py' | wc -l  # Should be >= 6
ls data/raw data/processed data/synthetic data/embeddings models results notebooks report prompts
```

**Execution:** `claude --prompt prompts/01_infrastructure.md --ultrathink`
