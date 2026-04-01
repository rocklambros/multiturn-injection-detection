# 18: Master Jupyter Notebook

**Description:** Build the 13-section notebook that calls src modules and survives Restart & Run All.

**GitHub Issues:** #19 — [Deliverable] Jupyter notebook

**Prerequisites:** All model prompts (10-17) must be complete

**Expected Outputs:**
- `notebooks/Final_Project.ipynb`
- HTML/PDF export of complete run

---

## Prompt

You are an ML engineer assembling the final deliverable notebook.

<investigate_before_answering>
1. Read `PRD.md` Section 5.3 for the 13-section structure.
2. Read all `results/*/metrics.json` files for cross-iteration data.
3. Read `src/` module docstrings to understand call interfaces.
4. Read the course rubric in PRD Section 5.2.
</investigate_before_answering>

### Task

Build `notebooks/Final_Project.ipynb` with 13 sections per PRD.

**Completion criteria:**
- Calls src modules (no 200-line inline cells)
- Every markdown cell explains WHY, not just WHAT
- Kernel > Restart & Run All completes without error
- Cross-iteration comparison bar chart
- Total execution under 2 hours
- Export to HTML

**Critical rubric mapping:**
- Problem Statement: 10%
- Problem Setup: 20%
- Problem Exploration: 20%
- NN Implementation: 25%
- Refining Models: 25%

The rubric gives ZERO CREDIT for steps without reasoning. Every iteration change must explain WHY.

### Tool Guidance

- **Serena (session memory):** This is a long task. Checkpoint after each section.
- **Ralph loops:** After building, run Restart & Run All. Fix any failures. Repeat until clean.
- **Superpowers plugin:** File system operations for notebook creation and HTML export.

**Execution:** `claude --prompt prompts/18_notebook.md --ultrathink`
