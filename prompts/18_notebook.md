# 18: Master Jupyter Notebook

**Description:** Build the 13-section notebook that calls src modules and survives Restart & Run All.

**GitHub Issues:** #19 — [Deliverable] Jupyter notebook

**Prerequisites:** All model prompts (10–17) complete

**Expected Outputs:**
- `notebooks/Final_Project.ipynb`
- HTML export of complete run

---

## Role

You are an ML engineer assembling the final deliverable notebook. Every markdown cell must explain WHY, not just WHAT. The rubric gives zero credit for steps without reasoning.

## Grounding

Use **superpowers** to read:
1. `PRD.md` Section 5.3 for the 13-section structure.
2. `PRD.md` Section 5.2 for rubric mapping.
3. All `results/*/metrics.json` files for cross-iteration data.

Use **goodmem** to read ALL stored metrics and decisions:
- All `baselines.*` keys
- All `models.*` keys (iter1 through iter7, encoder decision, core finding)
- `data.*` keys (sample counts, vocab size)

This is the prompt where goodmem pays off: every decision made in prior prompts feeds the narrative.

## Task

Build `notebooks/Final_Project.ipynb` with 13 sections per PRD.

**Completion criteria:**
- Calls src modules (no 200-line inline cells)
- Every markdown cell explains WHY
- Kernel > Restart & Run All completes without error
- Cross-iteration comparison bar chart
- Total execution under 2 hours on Jetson
- Export to HTML

**Critical rubric mapping:**
- Problem Statement: 10%
- Problem Setup: 20%
- Problem Exploration: 20%
- NN Implementation: 25%
- Refining Models: 25%

## Plugin Usage

**superpowers:** Create notebook JSON, run nbconvert for execution test.

**serena:** This is a long task. Checkpoint after every 3 sections:
- Checkpoint 1: Sections 1-3 (Problem, Data, Synthetic)
- Checkpoint 2: Sections 4-7 (Baselines, Iterations 1-3)
- Checkpoint 3: Sections 8-10 (Iterations 4-6)
- Checkpoint 4: Sections 11-13 (Iteration 7, Comparison, Conclusions)

**ralph-loop:** After building all 13 sections:
1. Run Kernel > Restart & Run All via `jupyter nbconvert --execute`
2. Review: Any cell failures? Missing imports? Shape errors?
3. Fix failing cells
4. Re-run until clean
5. Export to HTML
6. Confirm

**goodmem:** After completion, persist:
- `deliverables.notebook_path = notebooks/Final_Project.ipynb`
- `deliverables.notebook_html = notebooks/Final_Project.html`
- `deliverables.notebook_execution_time = <minutes>`

**Execution:** `claude --prompt prompts/18_notebook.md --ultrathink`
