# 19: Written Report

**Description:** Self-contained written report covering methodology, results, and conclusions.

**GitHub Issues:** #20 — [Deliverable] Written report

**Prerequisites:** Prompt 18 complete (notebook with all results)

**Expected Outputs:**
- `report/final_report.md`
- PDF export for Canvas submission

---

## Role

You are a technical writer producing an academic project report. The reader has not seen the notebook, the code, or the prompts. This report must stand completely alone.

## Grounding

Use **superpowers** to read:
1. `PRD.md` Section 5.2 for rubric mapping.
2. Key results from `results/*/metrics.json` files.
3. `PRD.md` Section 1.5 for references.

Use **goodmem** to read ALL metrics and decisions — the full story from baselines through multi-turn finding.

## Task

Write `report/final_report.md`.

**Completion criteria:**
- Maps to rubric: Problem Statement (10%), Setup (20%), Exploration (20%), Implementation (25%), Refining (25%)
- Includes key figures: architecture diagram, training curves, comparison table, attention heatmaps
- Proper citations for all references (Crescendo, FITD, Vassilev, GloVe, Chollet textbook)
- Not a code walkthrough — focus on insights and reasoning
- Credits external code and references

## Plugin Usage

**superpowers:** Write the report file. Convert to PDF if pandoc available.

**ralph-loop:**
1. Generate the full report
2. Review: Does it stand alone? Does every section map to the rubric? Are all iterations covered? Are citations complete?
3. Fix gaps
4. Confirm

**goodmem:** After completion, persist:
- `deliverables.report_path = report/final_report.md`

**Execution:** `claude --prompt prompts/19_report.md --ultrathink`
