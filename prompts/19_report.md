# 19: Written Report

**Description:** Self-contained written report covering methodology, results, and conclusions.

**GitHub Issues:** #20 — [Deliverable] Written report

**Prerequisites:** Prompt 18 (notebook complete with all results)

**Expected Outputs:**
- `report/final_report.md`
- PDF export for Canvas submission

---

## Prompt

You are a technical writer producing an academic project report.

<investigate_before_answering>
1. Read `PRD.md` Section 5.2 for rubric mapping.
2. Read `notebooks/Final_Project.ipynb` for complete results.
3. Read all `results/*/metrics.json` for data.
4. Read `PRD.md` Section 1.5 for references.
</investigate_before_answering>

### Task

Write `report/final_report.md` that stands alone without the notebook.

**Completion criteria:**
- Maps to rubric: Problem Statement (10%), Setup (20%), Exploration (20%), Implementation (25%), Refining (25%)
- Includes key figures: architecture diagram, training curves, comparison table, attention heatmaps
- Proper citations for all references
- Not a code walkthrough. Focus on insights and reasoning.
- Credits external code and references

### Tool Guidance

- **Sequential-thinking MCP:** Before writing, outline the narrative arc. Problem → why DL → data strategy → baselines → progressive improvement → novel multi-turn → attention interpretability → conclusions.

**Execution:** `claude --prompt prompts/19_report.md --ultrathink`
