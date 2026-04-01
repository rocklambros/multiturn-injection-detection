# 20: Presentation Deck

**Description:** 10-minute presentation for Week 10 class.

**GitHub Issues:** #21 — [Deliverable] 10-minute presentation deck

**Prerequisites:** Prompt 19 (report provides narrative)

**Expected Outputs:**
- Slide deck (PPTX or PDF)

---

## Prompt

You are preparing a 10-minute technical presentation for a graduate deep learning class.

<investigate_before_answering>
1. Read `report/final_report.md` for narrative and results.
2. Read `PRD.md` Section 5.2 for rubric context.
3. Gather key figures from `results/` directories.
</investigate_before_answering>

### Task

Create ~10-12 slides covering:
- Problem and motivation
- Architecture overview
- Data strategy
- Key results (cross-iteration F1 chart)
- Attention heatmap examples
- Conclusions and future work

**Completion criteria:**
- 10-minute time limit
- Anticipate Q&A: why not transformers? How realistic is synthetic data?
- Key visuals: architecture diagram, comparison bar chart, attention heatmap

### Tool Guidance

- **Superpowers plugin:** File system operations for slide creation.

**Execution:** `claude --prompt prompts/20_presentation.md --ultrathink`
