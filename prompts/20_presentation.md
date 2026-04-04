# 20: Presentation Deck

**Description:** 10-minute presentation for Week 10 class.

**GitHub Issues:** #21 — [Deliverable] 10-minute presentation deck

**Prerequisites:** Prompt 19 complete (report provides narrative)

**Expected Outputs:**
- Slide deck (PPTX or PDF) in `report/`

---

## Role

You are preparing a 10-minute technical presentation for a graduate deep learning class. The audience knows ML fundamentals but not your specific problem domain (prompt injection, multi-turn attacks). Lead with the problem, not the solution.

## Grounding

Use **superpowers** to read:
1. `report/final_report.md` for narrative and results.
2. Key figures from `results/` directories.

Use **goodmem** to read:
- `models.core_finding_f1_gap` — the headline number
- `models.attention_pattern` — the interpretability story
- `models.encoder_decision` and `models.encoder_decision_reasoning`

## Task

Create ~10-12 slides.

**Completion criteria:**
- Slide 1: Title
- Slides 2-3: Problem and motivation (what are multi-turn attacks, why current detectors fail)
- Slide 4: Architecture overview (dual-encoder diagram)
- Slide 5: Data strategy (3 datasets + 4 synthetic strategies)
- Slides 6-8: Key results (cross-iteration F1 chart, multi-turn vs single-turn gap, attention heatmap)
- Slide 9: Security implications (threshold tuning, FN vs FP cost)
- Slide 10: Conclusions and future work
- Anticipate Q&A: why not transformers? How realistic is synthetic data?

## Plugin Usage

**superpowers:** Create slide deck file.

**ralph-loop:**
1. Generate deck
2. Review: 10-minute pacing? Key visuals included? Q&A prep adequate?
3. Fix
4. Confirm

**goodmem:** After completion, persist:
- `deliverables.presentation_path = report/presentation.*`
- `deliverables.build_complete = true`

**Execution:** `claude --prompt prompts/20_presentation.md --ultrathink`
