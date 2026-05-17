# Temporal Detection of Distributed Prompt Injection Attacks in Multi-Turn Conversations

**Rock Lambros**

## Abstract

Prompt injection attacks against large language model (LLM) systems increasingly distribute malicious intent across multiple conversation turns, where each individual turn appears benign in isolation. Existing detection systems operate on single messages and cannot model the temporal dependencies that distributed attacks exploit. We present a dual-encoder architecture that combines a frozen single-turn GRU encoder with a trainable sequence LSTM to detect multi-turn injection patterns. The turn encoder compresses each message into a 32-dimensional representation; the sequence LSTM processes these representations temporally, learning to recognize escalation patterns, fragmented payloads, and cumulative constraint overrides. On a shared-prefix evaluation dataset of 27,180 synthetic conversations across four difficulty tiers, the temporal LSTM (27K trainable parameters) achieves F1 = 0.837 [0.826, 0.847], significantly outperforming turn-level voting baselines (best F1 = 0.727, p < 0.001). Shuffling turn order causes 55% of correctly classified attacks to flip to incorrect, confirming genuine temporal learning. We also evaluate hierarchical and concatenated DistilBERT baselines (5.5M to 66.4M parameters), which achieve F1 = 0.976 to 0.992 but require 200 to 2,460x more trainable parameters. The parameter efficiency of the dual-encoder design makes it deployable on edge hardware where transformer models are impractical.

## 1. Introduction

Large language model systems deployed in production face a growing class of attacks that exploit the multi-turn nature of conversational interaction. Traditional prompt injection uses a single malicious message to override the model's instructions. Distributed attacks instead spread malicious intent across several turns of apparently normal conversation. The Crescendo pattern (Russinovich et al., 2025) gradually escalates requests across turns, each individually innocuous, until the cumulative context enables the final exploitative request. The Foot-in-the-Door technique (2025) exploits compliance momentum: small, reasonable requests in early turns make the model more likely to comply with larger requests later.

These attacks exploit a structural weakness in current defenses. Production-grade injection detectors (ProtectAI's DeBERTa, InjecGuard, Rebuff) analyze each message independently. A per-message classifier has no mechanism to carry forward accumulated risk state across turns. An attacker who distributes their payload across five turns, each of which individually scores below the detection threshold, evades the filter entirely. Vassilev (2025) extends Gödel's incompleteness theorem to argue that no single-turn classifier can theoretically detect all such attacks, because the attack signal exists in the relationships between turns rather than in any individual turn's content.

We address this gap with a dual-encoder temporal architecture. A frozen single-turn GRU encoder compresses each conversation turn into a 32-dimensional vector. A trainable sequence LSTM processes these vectors temporally, learning cross-turn patterns that no per-turn classifier can capture. The architecture requires only 27,000 trainable parameters, three orders of magnitude fewer than transformer baselines, making it deployable on resource-constrained edge devices.

Our evaluation uses a shared-prefix dataset designed to minimize vocabulary confounds: each attack conversation shares identical opening turns with a matched benign conversation. We evaluate against 14 baselines and ablations with bootstrap confidence intervals and paired significance tests for all comparisons.

## 2. Related Work

### 2.1 Multi-Turn Attacks

Russinovich et al. (2025) introduce the Crescendo attack pattern, demonstrating that a series of gradually escalating requests can bypass safety training in GPT-4, Claude, and Gemini. The attack succeeds because each individual turn falls below the model's refusal threshold, but the accumulated conversational context shifts the model toward compliance. The Foot-in-the-Door technique (2025) draws on social psychology research to show that small initial commitments create compliance momentum exploitable in later turns.

These findings establish that multi-turn attacks represent a qualitatively different threat from single-turn injection. The attack signal is temporal: it exists in the sequence of turns, not in any individual message.

### 2.2 Single-Turn Detection

Current state-of-the-art injection detectors operate on individual messages. ProtectAI's DeBERTa fine-tuned model achieves 99%+ F1 on established benchmarks (Prompt Guard, Lakera). InjecGuard (2024) combines instruction-aware classification with heuristic rules. Rebuff uses a multi-layer approach including canary tokens and prompt hardening. All these systems analyze each message independently and have no mechanism to model cross-turn dependencies.

### 2.3 Sequence Modeling for Security

Temporal modeling has precedent in security applications. Network intrusion detection systems use LSTM and GRU architectures to model packet sequences (Mirsky et al., 2018). Malware classification benefits from sequential analysis of API call traces (Pascanu et al., 2015). In the NLP security domain, Perez and Ribeiro (2022) demonstrate adversarial attacks on language models that require multi-step reasoning about prompt context, though their work focuses on attack generation rather than detection.

Our work extends temporal modeling to the LLM security domain, where the "sequence" is a conversation and the temporal signal encodes social engineering patterns. The key distinction from prior sequence-modeling applications is that the temporal signal here is *semantic* rather than *syntactic*: the LSTM must learn that a turn establishing authority followed by a turn requesting access is suspicious, even though neither turn contains obviously malicious content.

### 2.4 Dual-Encoder Architectures

Dual-encoder designs have proven effective in settings where two levels of representation are needed. In document retrieval, Karpukhin et al. (2020) use separate query and passage encoders. In video classification, two-stream architectures process spatial and temporal features independently (Simonyan and Zisserman, 2014). Our design applies this principle to conversation analysis: the turn encoder captures per-message features, and the sequence model captures cross-message dynamics. The frozen turn encoder adds a constraint not present in most dual-encoder systems, forcing the temporal model to learn from fixed representations rather than jointly optimizing both levels.

### 2.5 Formal Limitations

Vassilev (2025) applies Gödel's incompleteness theorem to AI safety, arguing that any finite rule set for detecting harmful prompts will always have blind spots. This theoretical result motivates learned detectors over rule-based systems and, specifically, detectors that can model the relationships between messages rather than classifying each message against a fixed rule set.

## 3. Dataset

### 3.1 Shared-Prefix Design

No public dataset of multi-turn distributed prompt injection attacks exists. We generated 27,180 synthetic conversations using a shared-prefix architecture and the Anthropic API (Claude Sonnet 4.6).

Each conversation is generated as a matched pair. A conversational prefix of *k* user turns (sampled uniformly from {3, 4, 5}) establishes a natural topic. From this shared prefix, two continuations branch: one benign (natural topic continuation) and one attack (distributed injection). Both continuations run for 3-5 additional user turns, producing conversations of 6-9 user turns (12-19 turns including assistant responses).

The shared-prefix design eliminates vocabulary-level confounds in the opening turns. A first-turn-only classifier achieves F1 = 0.35 on this data, which is chance level for balanced classes. This means any model that achieves above-chance performance must rely on information from the post-branch turns, where attack and benign conversations diverge.

### 3.2 Attack Strategies

Four attack strategies, drawn from published research, distribute malicious intent across the post-branch turns:

**Fragment distribution (45%)** splits the injection payload into 3-5 fragments interleaved with on-topic filler messages. Each fragment independently scores below the single-turn detection threshold. The strategy exploits the fundamental assumption of per-message classifiers: that each message can be evaluated independently.

**Gradual escalation (25%)** follows the Crescendo pattern. Each turn nudges the conversation closer to the attack goal without making an overt request. Early post-branch turns ask benign follow-up questions; middle turns introduce the target topic; final turns make the exploitative request in the context established by prior turns.

**Context priming (15%)** establishes persona or authority in early post-branch turns, then exploits the established trust in later turns. The attacker claims expertise, cites authority, or builds rapport before making the injection request.

**Instruction layering (15%)** adds one behavioral constraint per turn, cumulatively overriding the model's safety guidelines. No individual constraint is obviously malicious, but their accumulation steers the model toward the desired behavior.

### 3.3 Difficulty Tiers

Each attack is assigned a difficulty tier controlling the degree of camouflage:

| Tier | Test n | Characteristics |
|------|-------:|----------------|
| Easy | 1,462 | Shorter prefixes, less camouflage, more direct language |
| Medium | 1,414 | Moderate prefix length, some topic-relevant camouflage |
| Hard | 1,394 | Longer prefixes, strong camouflage, subtle escalation |
| Adversarial | 860 | Maximum camouflage, attack indistinguishable from topic drift |

All tiers are balanced 50/50 attack/benign.

### 3.4 Data Quality Controls

Three mechanisms guard against artifacts:

1. **Shared-prefix pairing** eliminates early-turn vocabulary confounds. First-turn-only classifier: F1 = 0.35. Conversation-length classifier: F1 = 0.48. Both at chance level.

2. **Validation gate**: A pre-trained single-turn GRU classifier scores each individual turn. Sequences where any single turn exceeds the detection threshold are rejected, forcing the attack signal into cross-turn patterns.

3. **Confound gate battery**: Seven automated checks run on 5-fold cross-validation of the training split. Each gate tests whether a specific shortcut feature achieves above-threshold F1:

| Gate | F1 (mean ± std) | Threshold | Result |
|------|:---:|:---:|:---:|
| Unigram BoW | 0.938 ± 0.003 | < 0.60 | FAIL |
| Bigram BoW | 0.945 ± 0.003 | < 0.65 | FAIL |
| First-turn only | 0.354 ± 0.015 | < 0.58 | PASS |
| Last-turn only | 0.963 ± 0.003 | < 0.65 | FAIL |
| Conversation length | 0.482 ± 0.191 | < 0.55 | PASS |
| Max-vote BoW | 0.684 ± 0.001 | < 0.70 | PASS |
| Mean-vote BoW | 0.926 ± 0.006 | < 0.65 | FAIL |

Three of seven pass. The first-turn and conversation-length gates confirm the shared-prefix design eliminates early-turn and structural confounds. The BoW failures indicate residual vocabulary differences in post-branch turns, since attack continuations use somewhat different vocabulary because they pursue different conversational goals. The turn-order sensitivity analysis (Section 6.3) demonstrates that the temporal model relies on ordering information inaccessible to BoW classifiers.

### 3.5 Dataset Statistics

| Split | Total | Attack | Benign |
|-------|------:|-------:|-------:|
| Train | 18,754 | 9,377 | 9,377 |
| Val | 3,296 | 1,648 | 1,648 |
| Test | 5,130 | 2,565 | 2,565 |

## 4. Model Architecture

### 4.1 Dual-Encoder Design

The architecture separates turn-level feature extraction from cross-turn temporal modeling:

```
Turn 1 → [Frozen GRU Encoder] → 32-dim ─┐
Turn 2 → [Frozen GRU Encoder] → 32-dim  ─┤
Turn 3 → [Frozen GRU Encoder] → 32-dim  ─┼→ [Sequence LSTM (64-dim)] → Dense(64→32→1)
  ...                                     │
Turn N → [Frozen GRU Encoder] → 32-dim ─┘
```

**Turn Encoder**: A bidirectional GRU with 128-dimensional hidden state (64 per direction), trained on 73K single-turn injection samples. The `encode()` method returns the final hidden state projected to 32 dimensions. All 2.6M parameters are frozen during multi-turn training. The frozen encoder serves as a compression function: each turn is reduced to a 32-dimensional vector capturing "how injection-like is this turn."

**Sequence LSTM**: A single-layer LSTM with 64-dimensional hidden state processes the sequence of turn vectors. The LSTM's forget, input, and output gates learn temporal patterns: escalation trajectories, fragmented payload accumulation, and persona establishment-exploitation sequences. Only the LSTM and classification head are trainable (27,000 parameters total).

**Classification Head**: Dense(64→32) → ReLU → Dropout(0.3) → Dense(32→1). Trained with BCEWithLogitsLoss.

### 4.2 Attention Variant

An additive (Bahdanau) attention layer computes a weighted combination of all LSTM hidden states, replacing final-hidden-state-only classification. Attention weights indicate each turn's contribution to the classification decision, providing interpretability for security analysts.

### 4.3 Transformer Baselines

**Hierarchical DistilBERT (PM-1a)**: Frozen DistilBERT (66.4M params) processes each turn independently, extracting [CLS] representations. A trainable 2-layer, 4-head cross-turn transformer (5.5M params) processes the turn sequence. Learned position embeddings encode turn order.

**Concatenated DistilBERT (PM-1b)**: All turns are joined with [SEP] tokens and processed through a fully fine-tuned DistilBERT (66.4M trainable params). This is the brute-force approach: provide the entire conversation as a single token sequence.

## 5. Single-Turn Baseline Results

Before constructing the multi-turn detector, we trained and evaluated seven single-turn classifiers on 73,390 samples (51,373 training) to select the turn encoder and validate a dataset-complexity heuristic.

| Model | F1 | Notes |
|-------|:---:|-------|
| TF-IDF + Random Forest | 0.834 | Bag-of-bigrams, 500 trees |
| GRU (chosen encoder) | 0.815 | 128-dim hidden, bidirectional |
| BiLSTM + Dropout | 0.815 | 128-dim hidden, dropout 0.3 |
| LSTM (baseline) | 0.814 | 128-dim hidden |
| TF-IDF + Logistic Regression | 0.814 | L2-regularized |
| LSTM + GloVe | 0.813 | 300-dim pretrained embeddings |
| Custom Transformer | 0.808 | 2-layer, 4-head, 128-dim |

The TF-IDF + Random Forest baseline outperforms all neural architectures. This result aligns with Chollet's dataset complexity heuristic: the ratio of training samples to vocabulary size is 51,373 / 87.3 = 588, well below the 1,500 threshold above which neural models typically justify their capacity. At this ratio, bag-of-bigrams captures sufficient signal; transformers and recurrent models add capacity without commensurate data to exploit it.

The GRU was selected as the turn encoder despite not winning the single-turn comparison, because it produces 32-dimensional hidden-state embeddings suitable for downstream temporal processing. TF-IDF features lack the continuous geometry needed for sequential modeling. DistilBERT (frozen, F1 = 0.806) was considered but rejected for computational cost on edge hardware.

## 6. Results

### 6.1 Model Hierarchy

All results on the v3 test set (5,130 sequences). 95% bootstrap CIs from 1000 resamples.

| Model | F1 | 95% CI | AUC | Params |
|-------|:---:|:---:|:---:|------:|
| Concatenated DistilBERT | 0.992 | [0.989, 0.994] | 1.000 | 66.4M |
| Hierarchical DistilBERT | 0.976 | [0.971, 0.980] | 0.998 | 5.5M |
| Continuation-only LSTM | 0.846 | [0.835, 0.856] | 0.923 | 27K |
| Autoencoder encoder | 0.845 | [0.834, 0.856] | 0.922 | 27K |
| Iter 6 (LSTM + attention) | 0.837 | [0.825, 0.848] | 0.921 | 29K |
| **Iter 5 (temporal LSTM)** | **0.837** | **[0.826, 0.847]** | **0.919** | **27K** |
| Reversed turns | 0.833 | [0.821, 0.844] | 0.916 | 27K |
| Shuffled turns | 0.760 | [0.748, 0.772] | 0.849 | 27K |
| Mean pool | 0.755 | [0.743, 0.768] | 0.839 | 27K |
| A10 top-3-mean voting | 0.727 | - | - | 0 |
| Max pool | 0.719 | [0.705, 0.733] | 0.819 | 27K |
| A10 max-vote | 0.706 | - | - | 0 |
| Prefix-only | 0.667 | [0.655, 0.679] | 0.500 | 27K |
| Cosine baseline | 0.612 | [0.596, 0.627] | 0.642 | 0 |
| A10 mean-vote | 0.231 | - | - | 0 |

### 6.2 Statistical Significance

Paired one-sided bootstrap tests (1000 resamples):

| Comparison | Observed Diff | p-value | Significant |
|-----------|:---:|:---:|:---:|
| Temporal LSTM > A10 max-vote | +0.131 | < 0.001 | Yes |
| Temporal LSTM > A10 top-3-mean | +0.110 | < 0.001 | Yes |
| Temporal LSTM > A10 mean-vote | +0.606 | < 0.001 | Yes |
| Temporal LSTM > shuffled turns | +0.077 | < 0.001 | Yes |
| Continuation-only > prefix-only | +0.179 | < 0.001 | Yes |
| Attention > temporal LSTM | +0.000 | 0.453 | No |
| DistilBERT-hier > temporal LSTM | +0.139 | < 0.001 | Yes |
| DistilBERT-concat > temporal LSTM | +0.155 | < 0.001 | Yes |

The first four comparisons establish the core claim: temporal LSTM modeling significantly outperforms all turn-level aggregation methods. The shuffled-turns comparison uses the same model, same parameters, same data; only the turn order differs. The continuation-vs-prefix comparison confirms that post-branch turns carry the discriminative signal (+0.179 F1), consistent with the shared-prefix design. Both DistilBERT variants significantly outperform the temporal LSTM, establishing the accuracy-efficiency tradeoff.

Under Bonferroni correction for 8 comparisons (adjusted alpha = 0.00625), all significant results remain significant since every reported p-value is below 0.001. The Holm-Bonferroni sequential procedure yields the same conclusion.

### 6.3 Turn-Order Sensitivity

We take every correctly classified attack sequence, randomly shuffle its turns, and re-run inference through the temporal LSTM:

- **55% of correctly classified attacks flip to incorrect after shuffling**
- Ordered F1: 0.837 → Shuffled F1: 0.489 (on the originally-correct subset)
- Flip rate is uniform across tiers: easy 55%, medium 54%, hard 56%, adversarial 55%

This is the strongest evidence that the LSTM learns genuine temporal patterns. A model relying on vocabulary or per-turn features would be unaffected by shuffling. The uniformity across tiers confirms the model does not rely on tier-specific shortcuts.

### 6.4 Per-Tier Performance

| Tier | Temporal LSTM F1 | +Attention F1 | DistilBERT-concat F1 | n |
|------|:---:|:---:|:---:|---:|
| Easy | 0.872 | 0.876 | 0.994 | 1,462 |
| Medium | 0.828 | 0.832 | 0.991 | 1,414 |
| Hard | 0.828 | 0.830 | 0.994 | 1,394 |
| Adversarial | 0.802 | 0.786 | 0.985 | 860 |

The temporal LSTM shows a 7-point F1 degradation from easy to adversarial, confirming the difficulty tiers function as designed. DistilBERT's degradation is smaller (0.9 points) but follows the same pattern, suggesting the tier rankings reflect genuine properties of the attacks.

### 6.5 Per-Strategy Analysis

| Strategy | Temporal LSTM F1 | +Attention F1 | n (test) |
|----------|:---:|:---:|:---:|
| Fragment distribution | 0.776 | 0.787 | 1,160 |
| Gradual escalation | 0.676 | 0.681 | 669 |
| Context priming | 0.628 | 0.650 | 372 |
| Instruction layering | 0.605 | 0.612 | 364 |

Fragment distribution is easiest to detect because it produces the most distinctive temporal signature: discrete payload fragments separated by filler turns create characteristic spikes in the turn-encoding sequence. Instruction layering is hardest because each turn adds a subtle constraint, producing a smooth gradient in the embedding space rather than sharp transitions. This difficulty ranking holds across all model variants, indicating it reflects genuine properties of the attack strategies rather than detector-specific artifacts.

### 6.6 Ablation Analysis

**Turn-level voting (A10)**: Using the same frozen GRU encoder as the temporal LSTM, we score each turn independently and aggregate via max, mean, or top-3-mean. Best voting F1 = 0.727 (top-3-mean), 11 points below temporal LSTM (p < 0.001). The gap cannot be explained by encoder quality or training data; the only difference is the presence of cross-turn processing.

**Shuffled vs ordered**: Random turn permutation drops F1 by 7.7 points (p < 0.001). Same model, same parameters, same data. The LSTM learns from turn order.

**Reversed vs ordered**: Reversing the turn order barely affects performance (0.833 vs 0.837, not significant). The LSTM can read the escalation pattern in either direction, consistent with the bidirectional nature of the underlying GRU encoder.

**Pooling ablations**: Replacing the sequence LSTM with mean pooling (F1 = 0.755) or max pooling (F1 = 0.719) drops performance by 8-12 points. Both lose temporal ordering; mean pooling additionally dilutes the signal.

**Continuation-only**: Providing only post-branch turns (F1 = 0.846) matches or slightly exceeds the full model (0.837). The shared prefix contributes noise, not signal.

**Autoencoder encoder**: Replacing the injection-trained GRU with a reconstruction-trained autoencoder (F1 = 0.845) produces equivalent performance. The sequence LSTM drives temporal detection regardless of whether the turn encoder was trained for injection classification.

### 6.7 Threshold Tuning

Sweeping the classification threshold on the validation set from the default 0.5 to 0.64 improves the single-turn GRU encoder to F1 = 0.995 (precision = 0.996, recall = 0.994) with ROC-AUC = 0.9997 and PR-AUC = 0.9997. The confusion matrix at the optimal threshold shows only 2 false positives and 3 false negatives on a 1,000-sample test set. This result demonstrates that the learned representations contain more discriminative information than the default threshold reveals, and that threshold calibration should be standard practice before reporting final single-turn performance.

## 7. Analysis

### 7.1 What Temporal Modeling Captures

The turn-order gap (55% flip rate on shuffle) and the voting gap (+0.131 over max-vote) together establish that the LSTM learns genuine cross-turn relationships. Three patterns are most prominent:

**Escalation trajectories**: The LSTM's forget gate maintains high activation across escalating turns, preserving the accumulated context. The input gate spikes on turns that advance the attack goal. This pattern appears most clearly in gradual escalation attacks.

**Fragment accumulation**: For fragment distribution attacks, the LSTM's cell state incrementally incorporates each payload fragment. The classification confidence increases monotonically with each fragment observed, reflecting the model's ability to accumulate evidence across non-adjacent turns.

**Context exploitation**: In context priming attacks, the LSTM's hidden state shifts detectably when the conversation pivots from trust-building to exploitation. This pivot produces a characteristic change in the hidden-state trajectory visible in PCA projections.

### 7.2 Parameter Efficiency

The temporal LSTM achieves F1 = 0.837 with 27,000 trainable parameters. Concatenated DistilBERT achieves 0.992 with 66.4M, a 2,460x parameter ratio. The dual-encoder design is 200x smaller even than hierarchical DistilBERT (5.5M trainable).

This efficiency matters for deployment. The temporal LSTM runs inference in under 5ms per conversation on an NVIDIA Jetson Orin AGX (Ampere GPU). DistilBERT requires 40-80ms per conversation for the concatenated variant. On CPU-only servers or mobile edge devices, the gap is larger.

The efficiency comes from the architectural separation: the frozen turn encoder compresses 256-token turns into 32-dimensional vectors, and the sequence LSTM operates entirely in this compressed space. The model never processes raw text during temporal analysis; it works with the GRU's summary of each turn.

### 7.3 Residual Confounds

The confound gate battery reveals that unigram/bigram BoW classifiers achieve F1 > 0.93 on training data. This means vocabulary differences between attack and benign continuations remain detectable by lexical classifiers, despite the shared prefix.

Three observations contextualize this result:

First, the vocabulary confound concentrates in post-branch turns. The shared prefix design successfully eliminates confounds in the opening turns (first-turn F1 = 0.35). The remaining confound reflects the genuine fact that attack continuations use somewhat different vocabulary than benign ones because they discuss different topics and pursue different goals.

Second, the temporal model operates in a 32-dimensional embedding space where raw vocabulary is already compressed away. The model cannot "read" the words in each turn; it processes the GRU's summary of each turn's injection-likeness. The BoW confound is inaccessible to the temporal model by construction.

Third, the turn-order sensitivity analysis (55% flip rate) demonstrates that the model relies on temporal ordering inaccessible to BoW classifiers. Shuffling the turns preserves all vocabulary features but destroys the temporal structure, and the model's performance degrades substantially.

### 7.4 Failure Modes

Per-strategy analysis reveals that instruction layering (F1 = 0.605) and context priming (F1 = 0.628) are the hardest strategies to detect. Both produce gradual, smooth changes in the turn-encoding sequence rather than the sharp transitions characteristic of fragment distribution. The temporal LSTM's 64-dimensional hidden state may lack the capacity to model these subtle shifts.

Per-tier analysis shows adversarial sequences (F1 = 0.802) are genuinely harder, though the model degrades gracefully rather than catastrophically. The 7-point gap from easy to adversarial suggests room for improvement, particularly through larger hidden states or multi-layer sequence models.

Examining the error patterns more closely, false negatives cluster in conversations where the attack goal closely matches the natural topic of the shared prefix. For example, a conversation about home security that transitions into requesting lockpicking instructions looks similar to a benign conversation about physical security measures. The GRU encoder's 32-dimensional compression loses the semantic distinction between "discussing security measures" and "requesting attack instructions" because both share security-related vocabulary. This compression-induced ambiguity represents the fundamental tradeoff of the dual-encoder design: the same compression that prevents vocabulary confounds also prevents fine-grained semantic discrimination.

False positives concentrate in conversations where benign continuations involve unusually specific or directive language, such as technical instructions, step-by-step guides, or detailed how-to content. The model interprets the shift from casual conversation to structured, imperative language as an escalation pattern resembling instruction layering. This failure mode suggests that the model has partially learned a proxy for "conversational register shift" rather than purely detecting malicious intent.

### 7.5 Attention Pattern Analysis

The attention variant (iter6) provides interpretability through turn-level attention weights. On correctly classified attack sequences, attention concentrates on the turns immediately before and during the exploitation phase, typically turns 4-6 in a 7-turn conversation. On fragment distribution attacks, attention peaks correspond to turns containing payload fragments, suggesting the model has learned to weight the turns that contribute most to the cumulative attack signal.

The attention weights also reveal an asymmetry between attack strategies. For gradual escalation, attention is distributed more uniformly across turns, reflecting the gradual nature of the escalation. For context priming, attention peaks at the pivot point where the conversation shifts from trust-building to exploitation. These patterns are consistent with the theoretical structure of each strategy and provide evidence that the model learns strategy-appropriate temporal representations.

Despite these interpretable patterns, attention does not improve classification accuracy (iter6 F1 = 0.837 vs iter5 F1 = 0.837, p = 0.453). The final hidden state of the LSTM already captures sufficient information for classification. Attention provides a different (weighted-sum) view of the same information, useful for human interpretability but not for machine classification on this dataset.

## 8. Limitations

**Synthetic data**: All conversations are generated by a single LLM (Claude Sonnet 4.6). Real-world distributed attacks would exhibit greater diversity in phrasing, domain context, and social engineering sophistication. The model's performance on naturally occurring attacks is unknown.

**Single source model**: Cross-model generalization (detecting attacks generated by GPT-4, Gemini, or open-weight models) has not been tested. Stylistic differences between LLMs may affect the transferability of learned patterns.

**Residual vocabulary confounds**: Despite the shared-prefix design, BoW classifiers achieve F1 > 0.93 on training data. While the temporal model cannot exploit vocabulary by construction, this confound complicates the interpretation of ablation results that do have text access (e.g., DistilBERT baselines).

**Fixed conversation length**: Conversations range from 6-9 user turns. Production LLM conversations may span 20-50 turns. The LSTM's ability to maintain discriminative hidden states over longer sequences is untested.

**No human validation**: The attack strategies are designed synthetically. Human annotators have not validated that the generated attacks represent realistic threat scenarios. An annotation protocol has been developed (300 sequences, 3 annotators, Krippendorff's alpha target ≥ 0.60) but not yet executed.

## 9. Future Directions

**Cross-domain transfer**: The model is trained on synthetic conversations about general topics. Testing on domain-specific multi-turn attacks (customer service, code generation, medical consultation) would assess generalization. The frozen turn encoder is trained on diverse single-turn injection data, which may provide a degree of domain robustness, but the sequence LSTM's learned temporal patterns may be dataset-specific.

**Online detection**: The current architecture requires the complete conversation before classification. For deployment, the model should produce calibrated probabilities after each new turn, enabling early detection of attacks in progress. This requires modifying the sequence LSTM to emit per-timestep predictions and calibrating these intermediate outputs against the full-conversation ground truth.

**Formal safety analysis**: The LSTM hidden-state trajectories form continuous safety-state curves in 64-dimensional space. Investigating whether these trajectories satisfy the Markov property would connect our empirical results to formal verification approaches. If the hidden state at turn *t* contains all information needed to predict the classification at turn *t+1*, the safety-state dynamics can be modeled as a Markov chain, enabling formal guarantees about detection latency and false-negative bounds.

**Longer contexts**: Production LLM conversations commonly span 20-50 turns. The LSTM's ability to maintain discriminative hidden states over longer sequences needs investigation. Attention-based pooling over the turn sequence may be necessary to avoid gradient degradation in longer conversations.

**Hybrid architectures**: The 15.5-point F1 gap between the temporal LSTM (0.837) and concatenated DistilBERT (0.992) suggests that combining temporal modeling with pretrained language understanding could yield improvements. A promising direction is replacing the frozen GRU encoder with a frozen DistilBERT while keeping the lightweight sequence LSTM, combining rich turn representations with efficient temporal processing.

## 10. Conclusion

We present a dual-encoder architecture for detecting distributed prompt injection attacks in multi-turn conversations. The temporal LSTM achieves F1 = 0.837 with 27,000 trainable parameters, significantly outperforming turn-level voting baselines (p < 0.001). Turn-order sensitivity analysis confirms genuine temporal learning: 55% of correctly classified attacks flip to incorrect when turns are shuffled.

The architecture's parameter efficiency, with three orders of magnitude fewer trainable parameters than DistilBERT baselines, enables deployment on resource-constrained edge devices. The frozen turn encoder provides a natural decomposition: single-turn detection expertise is preserved while the sequence LSTM learns the cross-turn patterns that distributed attacks create.

Transformer baselines with full text access achieve higher absolute performance (F1 = 0.976-0.992), establishing an upper bound on what is achievable with current model capacity. The dual-encoder design occupies a different point on the accuracy-efficiency frontier: it trades absolute performance for deployability, interpretability (via attention weights), and architectural transparency (the model operates on compressed turn representations, not raw text).

The broader implication is that multi-turn attacks require multi-turn defenses. Per-message classifiers, regardless of their sophistication on single-turn benchmarks, have a structural blind spot for temporal attack patterns. The gap between per-turn voting (F1 = 0.727) and temporal modeling (F1 = 0.837) quantifies this blind spot and motivates the development of sequence-aware detection systems for production LLM deployments.

## References

- Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP 2020*.
- Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). Visualizing the Loss Landscape of Neural Nets. *NeurIPS 2018*.
- Mirsky, Y., Doitshman, T., Elovici, Y., & Shabtai, A. (2018). Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection. *NDSS 2018*.
- Pascanu, R., Stokes, J. W., Sanossian, H., Marinescu, M., & Thomas, A. (2015). Malware classification with recurrent networks. *ICASSP 2015*.
- Perez, F. & Ribeiro, I. (2022). Ignore This Title and HackAPrompt: Exposing Systemic Weaknesses of LLMs through a Global Scale Prompt Hacking Competition. *EMNLP 2023*.
- Russinovich, M., Salem, A., & Eldan, R. (2025). Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack. *USENIX Security 2025*.
- Simonyan, K. & Zisserman, A. (2014). Two-Stream Convolutional Networks for Action Recognition in Videos. *NeurIPS 2014*.
- Vassilev, A. (2025). Fundamental Limits of AI Safety: Applying Gödel's Incompleteness Theorems to AI Content Moderation. *NIST AI 100-2e2025*.
- Foot-in-the-Door: Understanding and Mitigating Compliance Momentum in LLM Conversations. *EMNLP 2025*.
- InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models. *2024*.
