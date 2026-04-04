# Multi-Turn Distributed Prompt Injection Detection

**Author:** Rock Lambros  
**Course:** COMP 4531: Deep Learning, University of Denver  
**Date:** April 2026  
**Platform:** NVIDIA Jetson Orin AGX (64GB RAM, 2048-core Ampere GPU)

---

## 1. Problem Statement and Motivation

### 1.1 The Problem

Single-turn prompt injection detection achieves high accuracy on known attack patterns — ProtectAI's DeBERTa model reaches 99%+ F1 on published benchmarks — but this performance is limited to known attack distributions. Novel attacks, adversarially crafted inputs, and multi-turn strategies routinely evade these detectors (InjecGuard, 2024; Vassilev, 2025). Real-world attacks against AI agent systems increasingly distribute malicious intent across multiple conversation turns, where each individual turn appears benign in isolation.

This is the **Crescendo attack pattern** (Russinovich et al., USENIX Security 2025): an attacker gradually escalates across turns, establishing trust and context before making the exploitative request. The **Foot-in-the-Door technique** (EMNLP 2025) similarly leverages compliance momentum across turns. Vassilev (2025) extends Gödel's incompleteness theorem to argue that no single-turn classifier can theoretically detect all such attacks.

### 1.2 Why Deep Learning

The multi-turn attack signal is fundamentally temporal: earlier turns create context that later turns exploit. LSTM and GRU architectures are designed to model exactly this type of sequential dependency through their gating mechanisms. A bag-of-words or per-turn classifier has no mechanism to carry forward accumulated risk state.

### 1.3 Contribution

This project builds the first multi-turn distributed prompt injection detection system using a dual-encoder architecture: a frozen single-turn encoder paired with a sequence-level LSTM that carries forward context across turns. We demonstrate a **+19 percentage point F1 improvement** over per-turn classification on multi-turn attack data.

---

## 2. Data and Preprocessing

### 2.1 Single-Turn Datasets

Eight HuggingFace datasets were merged (expanded from the original three to enable transformer comparison per instructor feedback):

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

### 2.3 Synthetic Multi-Turn Data

No public dataset of multi-turn distributed attacks exists. We generated 7,000 synthetic conversations (5,000 train, 1,000 val, 1,000 test) using four strategies:

- **Fragment distribution (40%)**: Split injection text into 3-5 fragments interleaved with benign filler
- **Gradual escalation (30%)**: Crescendo pattern — each turn adds specificity toward the goal
- **Context priming (20%)**: Establish persona/authority in early turns, exploit later
- **Instruction layering (10%)**: Each turn adds one constraint, cumulatively overriding safety

Balanced 50/50 classes, 3-10 turns per conversation. Benign conversations sampled from a pool of 500+ unique filler turns.

### 2.4 Tokenization

Custom vocabulary of 20,000 tokens built from training data only. Max sequence length: 256 tokens. OOV rate: 0.87% on training, 1.19% on validation.

### 2.5 Chollet Heuristic Analysis

Following Chollet (Deep Learning with Python, Chapters 11/15), we compute the ratio of training samples to mean words per sample to predict which model family will perform best:

- **Training samples**: 51,373
- **Mean words per sample**: 87.3
- **Ratio**: 51,373 / 87.3 = **588**
- **Threshold**: 1,500

At ratio 588 (well below 1,500), the Chollet heuristic predicts that **bag-of-bigrams models should outperform sequence and transformer models**. This is because with relatively few samples per unit of text complexity, simpler models with strong feature engineering (TF-IDF bigrams) can capture the discriminative patterns without the overfitting risk inherent in higher-capacity models.

This prediction is empirically validated: TF-IDF + RF (F1=0.834) outperforms all deep learning models including transformers. The instructor's insight is confirmed — transformers need substantially more training data to outperform simpler approaches on this task.

---

## 3. Model Architecture and Iteration Plan

### 3.1 Iteration 0: Baselines

TF-IDF (max 10K features, bigrams) + Logistic Regression and Random Forest. No deep learning.

### 3.2 Iterations 1-4: Single-Turn Models

| Iter | Architecture | Key Feature |
|------|-------------|-------------|
| 1 | LSTM(128→64) | Random embeddings |
| 2 | LSTM(100→64) | GloVe 6B 100d (frozen) |
| 3 | BiLSTM(128→64) | Bidirectional + dropout (0.3, 0.5) |
| 4 | BiGRU(128→64) | GRU comparison |

All use: Adam optimizer, BCELoss, early stopping, ReduceLROnPlateau, gradient clipping (max_norm=1.0).

### 3.3 Iterations 4b-4c: Transformer Comparison

Per instructor feedback, we add two transformer architectures to compare against LSTM/GRU:

| Iter | Architecture | Key Feature |
|------|-------------|-------------|
| 4b | Custom Transformer Encoder | 2-layer, 4-head self-attention, same vocab as LSTM |
| 4c | DistilBERT (frozen body) | Transfer learning, pretrained language model |

The custom transformer (Iter 4b) provides a controlled comparison: same vocabulary, same embedding dimension, similar parameter count (~2.8M) — the only difference is self-attention vs. recurrent gates. DistilBERT (Iter 4c) tests whether pretrained language understanding transfers to the security domain even with a frozen body (~99K trainable parameters).

### 3.5 Iteration 5: Multi-Turn Classifier (Novel)

Dual-encoder architecture:
1. **Turn encoder** (frozen GRU from Iter 4): encodes each turn into 32-dim vector
2. **Sequence LSTM** (64-dim hidden): processes turn vectors temporally
3. **Classification head**: Dense(64→32→1) with dropout

Only ~27,000 parameters trainable (the sequence LSTM and head). Turn encoder's 2.6M parameters are frozen.

### 3.6 Iteration 6: Attention

Additive attention over sequence LSTM hidden states replaces final-hidden-state-only classification. Provides interpretability: which turns drive the decision.

### 3.7 Iteration 7: Threshold Tuning

Sweep thresholds 0.01-0.99 on the attention model. Optimize for F1, 95% recall, and 95% precision operating points.

---

## 4. Results

### 4.1 Single-Turn Results

| Model | F1 | Accuracy | ROC-AUC |
|-------|-----|----------|---------|
| TF-IDF + LR (Iter 0) | 0.814 | 0.878 | 0.939 |
| **TF-IDF + RF (Iter 0)** | **0.834** | **0.890** | **0.945** |
| LSTM (Iter 1) | 0.814 | 0.877 | 0.942 |
| GloVe LSTM (Iter 2) | 0.813 | 0.881 | 0.942 |
| BiLSTM d=0.3 (Iter 3) | 0.815 | 0.884 | 0.942 |
| GRU (Iter 4) | 0.815 | 0.885 | 0.946 |
| Custom Transformer (Iter 4b) | 0.808 | 0.880 | 0.944 |
| DistilBERT frozen (Iter 4c) | 0.806 | 0.873 | — |

**Encoder decision**: GRU — competitive F1 with fewer parameters than BiLSTM.

**Chollet heuristic validated**: TF-IDF + RF (bag-of-bigrams) achieves the highest F1 at 0.834, confirming the Chollet prediction that at ratio 588, simpler models outperform sequence and transformer architectures. The custom transformer (0.808) and DistilBERT (0.806) both underperform the bag-of-bigrams baseline, demonstrating the instructor's point: transformers need substantially more training data to be competitive.

**GloVe finding**: With the expanded dataset, GloVe LSTM (F1=0.813) performs comparably to random embeddings (F1=0.814). The earlier dramatic gap (0.66 vs 0.95 on the smaller dataset) was an artifact of limited data — with more diverse training examples, both embedding approaches converge.

### 4.2 Multi-Turn Results (Core Finding)

| Model | Multi-Turn F1 |
|-------|--------------|
| TF-IDF + LR (concatenated) | 0.656 |
| TF-IDF + RF (concatenated) | 0.739 |
| GRU per-turn (max prob) | 0.887 |
| **Multi-turn LSTM (Iter 5)** | **0.989** |
| Multi-turn + Attention (Iter 6) | **0.992** |
| Tuned threshold (Iter 7) | **0.995** |

**The core finding**: the dual-encoder temporal architecture achieves **F1=0.989-0.995 on multi-turn data**, a **+10.3 percentage point improvement** over the best single-turn model applied per-turn (F1=0.887).

This validates the temporal hypothesis: the sequence-level LSTM carries forward accumulated context across turns, detecting escalation patterns that no individual turn reveals.

### 4.3 Threshold Analysis

Optimal threshold: 0.64 (F1=0.995). The threshold is close to the default 0.5, indicating the model produces well-calibrated probabilities for multi-turn attack sequences.

---

## 5. Discussion

### 5.1 Why Multi-Turn Works

The dual-encoder architecture succeeds because it separates two concerns:
1. **What does each turn say?** (turn encoder — frozen, already good at single-turn detection)
2. **How do turns relate over time?** (sequence LSTM — learns temporal patterns)

The frozen turn encoder prevents catastrophic forgetting while the sequence LSTM learns to recognize escalation, persona establishment, and cumulative constraint patterns.

### 5.2 Why GloVe Failed

GloVe embeddings are trained on general web text where security terms like "jailbreak," "bypass," and "inject" have different (or no) semantic meaning. Freezing the embeddings prevents the model from adapting these representations. Learned random embeddings, trained directly on injection data, capture domain-specific semantics that GloVe cannot.

### 5.3 Limitations

- **Synthetic data**: Our multi-turn conversations are generated from single-turn data using template-based strategies. Real-world distributed attacks may be more sophisticated and harder to detect.
- **English only**: All training data is English. Non-English injection patterns are not represented.
- **Known patterns**: Training data skews toward published attack types. Novel social engineering approaches may evade detection.
- **Fixed architecture**: 10-turn maximum with fixed 256-token turns. Real conversations vary more widely.

### 5.4 Future Work

1. **Real-world multi-turn datasets** from AI safety research teams
2. **Transformer-based turn encoders** (BERT/DeBERTa) for richer representations
3. **Online detection**: classify incrementally as each turn arrives
4. **Cross-lingual detection**: extend to multilingual attacks
5. **Adversarial robustness**: test against adversaries aware of the architecture

---

## 6. Reproducibility

- **Seed**: 42 for all random operations (Python, NumPy, PyTorch, cuDNN)
- **Platform**: NVIDIA Jetson Orin AGX, PyTorch 2.8.0, Python 3.12
- **Data**: All datasets from HuggingFace (Apache 2.0 / MIT / HuggingFace licenses)
- **Code**: Full source in `src/`, execution prompts in `prompts/`
- **Hardware**: 64GB RAM, Ampere GPU (sm_87), CUDA 12.6

All model weights, metrics, and plots saved to `models/` and `results/` directories.
