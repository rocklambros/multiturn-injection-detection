# Human Validation Annotation Protocol

## Overview

- 300 multi-turn conversation sequences sampled from the v3 test set
- 3 annotators recruited from COMP 4531 peers (non-authors, not involved in model development or data generation)
- Each sequence reviewed by 2 annotators (200 unique + 100 overlap for inter-annotator agreement)

## Sampling Strategy

Stratified sample from the test set:
- 75 sequences per difficulty tier (easy, medium, hard, adversarial)
- Within each tier: 50% attack, 50% benign (matching the dataset balance)
- Overlap set (100 sequences): 25 per tier, balanced labels

## Task Description

For each conversation, annotators answer three questions:

1. **Binary classification:** "Does this conversation contain a multi-turn prompt injection attempt?" (Yes / No)
2. **Turn localization (if Yes):** "At which turn does the attack begin?" (turn number)
3. **Confidence:** (High / Medium / Low)

## Annotation Interface

- Conversations displayed one turn at a time with a "show next turn" button
- Annotators see the full conversation context and make per-turn judgments
- No metadata (tier, strategy, generation method) is visible to annotators
- Turn numbers displayed for localization task
- Order of sequences randomized per annotator

## Inter-Annotator Agreement

- Compute Krippendorff's alpha on the 100 overlap sequences (binary classification task)
- Minimum acceptable alpha: 0.60 (moderate agreement)
- If alpha < 0.60: review disagreements, refine protocol, re-annotate disagreement cases
- Report agreement separately for each tier to identify systematic difficulty patterns

## Annotator Qualifications

- Familiarity with prompt injection concepts (brief 15-minute training session provided)
- Not involved in model development or data generation for this project
- Compensated at $0.50 per sequence ($150 total per annotator)

## Training Session

Before annotation begins, each annotator receives:
1. A 15-minute walkthrough of prompt injection attacks (single-turn and multi-turn examples)
2. 10 practice sequences with ground-truth labels and explanations
3. Written guidelines with edge-case examples

## Timeline

| Phase | Activity | Duration |
|-------|----------|----------|
| Phase 0 | Recruitment and IRB (if required) | Week 1 |
| Phase 2 | Training session (after data generation) | 1 day |
| Phase 5-6 | Annotation period | 2-3 days |
| Phase 6 | Agreement analysis and adjudication | 1 day |

## Output Format

Each annotator produces a JSON file:
```json
{
  "annotator_id": "A1",
  "annotations": [
    {
      "sequence_id": "test_0042",
      "contains_attack": true,
      "attack_start_turn": 3,
      "confidence": "high"
    }
  ]
}
```

## Analysis Plan

1. Compute Krippendorff's alpha on the 100-sequence overlap set
2. Report per-tier agreement rates
3. Compare annotator labels against model predictions to identify systematic model errors
4. Use disagreement cases to surface ambiguous attack patterns for future data generation
