# Multi-Turn Distributed Prompt Injection Detection

**Author:** Rock Lambros  
**Date:** May 2026  
**Platform:** NVIDIA Jetson Orin AGX (64GB RAM, 2048-core Ampere GPU)

---

## 1. Problem Statement and Motivation

### 1.1 The Problem

Single-turn prompt injection detection achieves high accuracy on known attack patterns — ProtectAI's DeBERTa model reaches 99%+ F1 on published benchmarks — but this performance is limited to known attack distributions. Novel attacks, adversarially crafted inputs, and multi-turn strategies routinely evade these detectors (InjecGuard, 2024; Vassilev, 2025). Real-world attacks against AI agent systems increasingly distribute malicious intent across multiple conversation turns, where each individual turn appears benign in isolation.

This is the **Crescendo attack pattern** (Russinovich et al., USENIX Security 2025): an attacker gradually escalates across turns, establishing trust and context before making the exploitative request. The **Foot-in-the-Door technique** (EMNLP 2025) similarly leverages compliance momentum across turns. Vassilev (2025) extends Gödel's incompleteness theorem to argue that no single-turn classifier can theoretically detect all such attacks.

### 1.2 Why Deep Learning

The multi-turn attack signal is fundamentally temporal: earlier turns create context that later turns exploit. LSTM and GRU architectures are designed to model exactly this type of sequential dependency through their gating mechanisms. A bag-of-words or per-turn classifier has no mechanism to carry forward accumulated risk state.

### 1.3 Contribution

This project builds a multi-turn distributed prompt injection detection system using a dual-encoder architecture: a frozen single-turn encoder paired with a sequence-level LSTM that carries forward context across turns. We evaluate on a shared-prefix dataset of 27,180 synthetic conversations across four difficulty tiers, with bootstrap confidence intervals and paired significance tests for all comparisons.

---

## 2. Data and Preprocessing

### 2.1 Single-Turn Datasets

Eight HuggingFace datasets were merged:

| Dataset | Samples | Type |
|---------|---------|------|
| deepset/prompt-injections | 662 | Binary classification |
| xTRam1/safe-guard-prompt-injection | 10,296 | Binary classification |
| neuralchemy/Prompt-injection-dataset | 6,274 | Binary classification |
| imoxto/prompt_injection_cleaned_dataset-v2 | 40,000 (subsampled from 535K) | Binary with system prompt context |
| reshabhs/SPML_Chatbot_Prompt_Injection | 16,012 | Gandalf CTF attacks |
| TrustAIRLab/in-the-wild-jailbreak-prompts (jailbreak) | 1,405 | Real-world jailbreaks |
| TrustAIRLab/in-the-wild-jailbreak-prompts (regular) | 13,735 | Real-world benign prompts |
| jackhhao/jailbreak-classification | 1,306 | Curated binary |

After cleaning (deduplication, whitespace normalization, length filtering), 73,390 samples remained, split 70/15/15 stratified on label: 51,373 train / 11,008 val / 11,009 test. Class balance: ~64% benign / 36% injection.

### 2.2 Cleaning Pipeline

Nine cleaning steps applied in order: column normalization, label normalization (0=benign, 1=injection), whitespace stripping, internal whitespace collapse, exact deduplication, near-deduplication (lowercase + strip punctuation), short text removal (<3 tokens), long text removal (>2048 chars), and logging.

Removals: 167 exact duplicates, 40 near-duplicates, 132 empty/short, 489 too-long.

### 2.3 Synthetic Multi-Turn Data (v3 Shared-Prefix)

No public dataset of multi-turn distributed attacks exists. We generated **27,180 synthetic conversations** (18,754 train / 3,296 val / 5,130 test) using a shared-prefix architecture and the Anthropic API (Claude Sonnet 4.6).

**Shared-prefix design**: Each conversation is generated as a matched pair — one benign continuation and one attack continuation branching from an identical conversational prefix. The prefix length *k* is sampled uniformly from {3, 4, 5} user turns. Both continuations run for 3-5 additional turns, producing conversations of 6-9 user turns total (12-19 turns including assistant responses). This paired structure eliminates vocabulary-level confounds in the opening turns: a first-turn-only classifier achieves F1 = 0.35 (chance level).

**Attack strategies**:

| Strategy | % of Attacks | Pattern | Research Basis |
|----------|:---:|---------|----------------|
| Fragment distribution | 45% | Split injection across 3-5 turns, interleaved with on-topic filler | Evasion of per-message filters |
| Gradual escalation | 25% | Crescendo pattern — each turn nudges toward the attack goal | Russinovich et al. (USENIX Security 2025) |
| Context priming | 15% | Establish persona/authority early, exploit later | Foot-in-the-Door (EMNLP 2025) |
| Instruction layering | 15% | Each turn adds one constraint, cumulatively overriding safety | Incremental constraint injection |

**Difficulty tiers**: Each attack is assigned a tier (easy, medium, hard, adversarial) controlling how aggressively the injection signal is obscured. All splits are balanced 50/50 within each tier.

| Tier | Train | Val | Test |
|------|------:|----:|-----:|
| Easy | 5,812 | 1,002 | 1,462 |
| Medium | 5,684 | 1,000 | 1,414 |
| Hard | 5,590 | 976 | 1,394 |
| Adversarial | 1,668 | 318 | 860 |

**Data quality controls**: (1) Shared-prefix pairing eliminates early-turn confounds. (2) Validation gate: a pre-trained GRU classifier rejects sequences where any individual turn exceeds the detection threshold, forcing the attack signal into cross-turn patterns. (3) Confound gate battery: seven automated checks run on 5-fold cross-validation of training data.

### 2.4 Tokenization

Custom vocabulary of 20,000 tokens built from training data only. Max sequence length: 256 tokens. OOV rate: 0.87% on training, 1.19% on validation.

### 2.5 Chollet Heuristic Analysis

Following Chollet (Deep Learning with Python, Chapters 11/15), the ratio of training samples to mean words per sample (51,373 / 87.3 = 588, well below the 1,500 threshold) predicts that bag-of-bigrams models should outperform sequence and transformer models on single-turn data. This prediction is empirically validated.

---

## 3. Model Architecture

### 3.1 Iteration 0: Baselines

TF-IDF (max 10K features, bigrams) + Logistic Regression and Random Forest. No deep learning.

### 3.2 Iterations 1-4: Single-Turn Models

| Iter | Architecture | Key Feature |
|------|-------------|-------------|
| 1 | LSTM(128→64) | Random embeddings |
| 2 | LSTM(100→64) | GloVe 6B 100d (frozen) |
| 3 | BiLSTM(128→64) | Bidirectional + dropout (0.3, 0.5) |
| 4 | BiGRU(128→64) | GRU comparison |

All use: Adam optimizer, BCEWithLogitsLoss, early stopping, ReduceLROnPlateau, gradient clipping (max_norm=1.0).

### 3.3 Iterations 4b-4c: Transformer Comparison

| Iter | Architecture | Key Feature |
|------|-------------|-------------|
| 4b | Custom Transformer Encoder | 2-layer, 4-head self-attention, same vocab as LSTM |
| 4c | DistilBERT (frozen body) | Transfer learning, pretrained language model |

### 3.4 Iteration 5: Multi-Turn Classifier

Dual-encoder architecture:
1. **Turn encoder** (frozen GRU from Iter 4): encodes each turn into 32-dim vector
2. **Sequence LSTM** (64-dim hidden): processes turn vectors temporally
3. **Classification head**: Dense(64→32→1) with dropout

Only ~27,000 parameters trainable (the sequence LSTM and head). Turn encoder's 2.6M parameters are frozen.

### 3.5 Iteration 6: Attention

Additive (Bahdanau) attention over sequence LSTM hidden states replaces final-hidden-state-only classification. Each turn gets an attention weight indicating its importance. Provides interpretability at zero accuracy cost.

### 3.6 DistilBERT Baselines

Two transformer baselines test whether raw model capacity substitutes for architectural design:

- **PM-1a: Hierarchical DistilBERT** (71.9M total, 5.5M trainable): Frozen DistilBERT per turn → [CLS] representations → 2-layer cross-turn transformer → classification head.
- **PM-1b: Concatenated DistilBERT** (66.4M, all trainable): All turns concatenated with [SEP], fully fine-tuned DistilBERT.

### 3.7 Ablation Models

Five ablation experiments isolate what drives temporal detection:

| Ablation | What It Tests |
|----------|---------------|
| A10: Turn-level voting (max, mean, top-3) | Can per-turn scoring + aggregation match temporal modeling? |
| Shuffled turns | Is turn order informative? |
| Reversed turns | Is forward-vs-backward direction informative? |
| Mean/max pool | Does the LSTM's sequential processing add value beyond aggregation? |
| Continuation-only / prefix-only | Which turns carry the class signal? |
| Autoencoder encoder | Does the GRU's injection-detection training matter? |

---

## 4. Results

### 4.1 Single-Turn Results

| Model | F1 | Accuracy | ROC-AUC |
|-------|-----|----------|---------|
| Stratified random (chance) | 0.358 | 0.540 | 0.500 |
| TF-IDF + LR (Iter 0) | 0.814 | 0.878 | 0.939 |
| **TF-IDF + RF (Iter 0)** | **0.834** | **0.890** | **0.945** |
| LSTM (Iter 1) | 0.814 | 0.877 | 0.942 |
| GloVe LSTM (Iter 2) | 0.813 | 0.881 | 0.942 |
| BiLSTM d=0.3 (Iter 3) | 0.815 | 0.884 | 0.942 |
| GRU (Iter 4) | 0.815 | 0.885 | 0.946 |
| Custom Transformer (Iter 4b) | 0.808 | 0.880 | 0.944 |
| DistilBERT frozen (Iter 4c) | 0.806 | 0.873 | — |

**Encoder decision**: GRU — competitive F1 with fewer parameters than BiLSTM. Selected as the frozen turn encoder for all multi-turn experiments.

**Chollet heuristic validated**: TF-IDF + RF achieves the highest single-turn F1 at 0.834, confirming the prediction that at ratio 588, simpler models outperform sequence and transformer architectures.

### 4.2 Multi-Turn Results (v3 Shared-Prefix Dataset)

All results below are on the v3 test set (5,130 sequences, 4 difficulty tiers, balanced 50/50 labels). 95% bootstrap confidence intervals from 1000 resamples.

| Model | F1 | 95% CI | AUC | Trainable Params |
|-------|:---:|:---:|:---:|:---:|
| Concatenated DistilBERT | 0.992 | [0.989, 0.994] | 1.000 | 66.4M |
| Hierarchical DistilBERT | 0.976 | [0.971, 0.980] | 0.998 | 5.5M |
| Continuation-only LSTM | 0.846 | [0.835, 0.856] | 0.923 | 27K |
| Autoencoder encoder | 0.845 | [0.834, 0.856] | 0.922 | 27K |
| Iter 6 (+attention) | 0.837 | [0.825, 0.848] | 0.921 | 29K |
| **Iter 5 (temporal LSTM)** | **0.837** | **[0.826, 0.847]** | **0.919** | **27K** |
| Reversed turns | 0.833 | [0.821, 0.844] | 0.916 | 27K |
| Shuffled turns | 0.760 | [0.748, 0.772] | 0.849 | 27K |
| Mean pool | 0.755 | [0.743, 0.768] | 0.839 | 27K |
| A10 top-3-mean voting | 0.727 | — | — | 0 |
| Max pool | 0.719 | [0.705, 0.733] | 0.819 | 27K |
| A10 max-vote | 0.706 | — | — | 0 |
| Prefix-only | 0.667 | [0.655, 0.679] | 0.500 | 27K |
| Cosine baseline | 0.612 | [0.596, 0.627] | 0.642 | 0 |
| A10 mean-vote | 0.231 | — | — | 0 |

### 4.3 Per-Tier Breakdown

| Tier | iter5 F1 | iter6 F1 | DistilBERT-concat F1 | n |
|------|:---:|:---:|:---:|---:|
| Easy | 0.866 | 0.874 | 0.996 | 1,462 |
| Medium | 0.837 | 0.840 | 0.996 | 1,414 |
| Hard | 0.831 | 0.830 | 0.989 | 1,394 |
| Adversarial | 0.794 | 0.786 | 0.984 | 860 |

### 4.4 Statistical Significance

Paired one-sided bootstrap tests (1000 resamples):

| Comparison | F1 Diff | p-value |
|-----------|:---:|:---:|
| iter5 > A10 max-vote | +0.131 | < 0.001 |
| iter5 > A10 top-3-mean | +0.110 | < 0.001 |
| iter5 > shuffled turns | +0.077 | < 0.001 |
| iter6 > iter5 | +0.000 | 0.453 (n.s.) |
| DistilBERT-concat > iter5 | +0.155 | < 0.001 |

### 4.5 Turn-Order Sensitivity

Shuffling the turns of correctly classified attack sequences: 55% flip from correct to incorrect. Ordered F1 = 0.837 → shuffled F1 = 0.489 (on the originally-correct subset). Flip rate is uniform across tiers (54-56%).

### 4.6 Per-Strategy Breakdown

| Strategy | iter5 F1 | iter6 F1 | n (test) |
|----------|:---:|:---:|:---:|
| Fragment distribution | 0.776 | 0.787 | 1,160 |
| Gradual escalation | 0.676 | 0.681 | 669 |
| Context priming | 0.628 | 0.650 | 372 |
| Instruction layering | 0.605 | 0.612 | 364 |

### 4.7 Confound Gate Analysis

Seven gates run on 5-fold cross-validation of training data:

| Gate | F1 (5-fold mean) | Threshold | Result |
|------|:---:|:---:|:---:|
| First-turn only | 0.354 | < 0.58 | PASS |
| Conversation length | 0.482 | < 0.55 | PASS |
| Max-vote BoW | 0.684 | < 0.70 | PASS |
| Unigram BoW | 0.938 | < 0.60 | FAIL |
| Bigram BoW | 0.945 | < 0.65 | FAIL |
| Last-turn only | 0.963 | < 0.65 | FAIL |
| Mean-vote BoW | 0.930 | < 0.70 | FAIL |

The passing gates confirm the shared-prefix design eliminates early-turn and length confounds. The failing gates indicate that vocabulary differences persist in post-branch turns. The turn-order sensitivity analysis (55% flip rate) demonstrates that the temporal LSTM relies on ordering information that BoW classifiers cannot exploit.

---

## 5. Discussion

### 5.1 Why Temporal Modeling Works

The dual-encoder architecture succeeds because it separates two concerns:
1. **What does each turn say?** (turn encoder — frozen, compression to 32-dim)
2. **How do turns relate over time?** (sequence LSTM — learns temporal patterns)

The turn-level voting gap (+0.131 F1 over max-vote, p < 0.001) demonstrates that independent per-turn scoring cannot recover the cross-turn signal. The shuffled-turns gap (+0.077 F1, p < 0.001) confirms that turn order carries genuine information. These two results together establish that the LSTM learns temporal relationships, not bag-of-turns features.

### 5.2 The DistilBERT Question

Concatenated DistilBERT achieves F1 = 0.992 with 66.4M trainable parameters versus our 0.837 with 27K — a 2,460x parameter ratio. This gap has two interpretations:

The DistilBERT models have full text access and can exploit both temporal and vocabulary signals, including the residual vocabulary differences that the confound gates flagged. Our model operates in a 32-dimensional embedding space where vocabulary is compressed away.

The comparison that matters is not "does our model beat DistilBERT" (it does not) but "does temporal modeling add value beyond per-turn classification" (it does, significantly). The 27K-parameter temporal model is deployable on resource-constrained devices where DistilBERT is impractical.

### 5.3 Limitations

- **Synthetic data**: All conversations generated by Claude Sonnet 4.6. Performance on naturally occurring multi-turn attacks is unknown.
- **Residual vocabulary confounds**: BoW classifiers achieve F1 > 0.93 on training data. The temporal signal operates on top of this confound.
- **Single model family**: Cross-model generalization (attacks generated by GPT-4, Gemini, open-weight models) is untested.
- **Fixed conversation length**: 6-9 user turns. Behavior on 20+ turn conversations is untested.
- **English only**: Non-English injection patterns are not represented.

### 5.4 Future Work

1. **Cross-domain transfer**: Test on multi-turn attack data from different domains
2. **Real-world validation**: Deploy as a secondary filter behind production AI systems
3. **Online detection**: Classify after each new turn for streaming inference
4. **Formal safety analysis**: Investigate whether LSTM hidden-state trajectories satisfy the Markov property for connection to formal verification approaches
5. **Longer contexts**: Extend to 20-50 turn conversations common in production

---

## 6. Reproducibility

- **Seed**: 42 for all random operations (Python, NumPy, PyTorch, cuDNN)
- **Platform**: NVIDIA Jetson Orin AGX, PyTorch 2.8.0, Python 3.12
- **Data**: Single-turn from HuggingFace (Apache 2.0 / MIT). Multi-turn generated via Anthropic API.
- **Code**: Full source in `src/`, evaluation in `scripts/`, results in `results/v3_evaluation/`
- **Hardware**: 64GB RAM, Ampere GPU (sm_87), CUDA 12.6. Training run on RunPod A100.
- **Statistical methods**: 1000-sample bootstrap for CIs, paired one-sided bootstrap for significance tests.

All model weights, metrics, predictions (.npz), and plots saved to `models/` and `results/` directories.
