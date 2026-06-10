---
gated: manual
language:
- en
license: cc-by-nc-4.0
task_categories:
- text-classification
tags:
- security
- prompt-injection
- multi-turn
- llm-safety
- adversarial
size_categories:
- 10K<n<100K
extra_gated_prompt: >-
  This dataset contains synthetic multi-turn prompt injection attack
  conversations designed for security research. By requesting access,
  you agree to:

  1. Use this data only for defensive security research, detection system
     development, or academic study

  2. Not use this data to develop, improve, or deploy offensive prompt
     injection tools against production AI systems

  3. Cite the associated paper in any published work that uses this dataset

  4. Report any misuse you become aware of to the dataset authors

  Please describe your intended use case below.
extra_gated_fields:
  Intended use case: text
  Affiliation: text
  I agree to the responsible use terms: checkbox
---

# Multi-Turn Distributed Prompt Injection Detection Dataset

## Dataset Description

27,180 synthetic multi-turn conversations (18,754 train / 3,296 val / 5,130 test) designed for training and evaluating temporal prompt injection detectors. Each conversation consists of 6-9 user turns with assistant responses.

### Shared-Prefix Design

Every attack conversation is paired with a benign conversation that shares identical opening turns. A conversational prefix of 3-5 user turns establishes a natural topic. From this shared prefix, two continuations branch: one benign (natural topic continuation) and one attack (distributed injection). This design eliminates vocabulary-level confounds in the opening turns --- a first-turn-only classifier achieves F1 = 0.35 (chance level).

### Attack Strategies

| Strategy | Proportion | Description |
|----------|-----------|-------------|
| Fragment distribution | 45% | Injection payload split across 3-5 turns with on-topic filler |
| Gradual escalation | 25% | Crescendo pattern: incremental escalation toward exploit |
| Context priming | 15% | Establish persona/authority, then exploit trust |
| Instruction layering | 15% | Cumulative behavioral constraints override safety |

### Difficulty Tiers

| Tier | Test n | Description |
|------|--------|-------------|
| Easy | 1,462 | Shorter prefixes, less camouflage, more direct language |
| Medium | 1,414 | Moderate prefix, some topic-relevant camouflage |
| Hard | 1,394 | Longer prefixes, strong camouflage, subtle escalation |
| Adversarial | 860 | Maximum camouflage, attack indistinguishable from topic drift |

All tiers are balanced 50/50 attack/benign.

### Data Format

Each JSON file contains a list of conversation objects:

```json
{
  "conversation_id": "sp_easy_train_0001",
  "label": 1,
  "difficulty": "easy",
  "strategy": "fragment_distributed",
  "prefix_length": 3,
  "turns": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ]
}
```

### Generation

All conversations were generated using Claude Sonnet 4.6 via the Anthropic API. A pre-trained single-turn GRU classifier validated that no individual turn exceeds the single-turn detection threshold.

## Intended Use

This dataset is intended for:
- Training multi-turn prompt injection detectors
- Benchmarking temporal detection architectures
- Studying distributed attack patterns in conversational AI

## Ethical Considerations

This dataset contains synthetic attack conversations that demonstrate techniques for bypassing AI safety measures. While the attack patterns are drawn from published research (Crescendo, Foot-in-the-Door), the dataset provides ready-made training examples that could potentially be misused. Access is gated to mitigate this risk. Users must agree to responsible use terms before downloading.

## Citation

If you use this dataset, please cite:

```bibtex
@misc{anonymous2026multiturn,
  title={Temporal Detection of Distributed Prompt Injection Attacks in Multi-Turn Conversations},
  author={{Anonymous (under review)}},
  year={2026},
  note={Anonymized for review}
}
```

## Associated Resources

- [Model weights](../../models/) (included in the review repository)
- [Repository](https://anonymous.4open.science/r/multiturn-injection-detection-73E6)
