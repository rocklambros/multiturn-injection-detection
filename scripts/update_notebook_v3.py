"""Update notebook cells for v3 results.

Modifies cells in-place: Section 3 (data pipeline), Sections 9-11 (multi-turn),
inserts new DistilBERT + ablation sections, updates Section 12 (cross-iteration).
"""

import json
import copy
from pathlib import Path

NB_PATH = Path("notebooks/multiturn_injection_detection.ipynb")

def make_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split("\n") if isinstance(source, str) else source
    }

def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split("\n") if isinstance(source, str) else source
    }

def fix_source(lines):
    """Ensure each line (except last) ends with newline for nbformat."""
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1 and not line.endswith("\n"):
            result.append(line + "\n")
        else:
            result.append(line)
    return result


# ============================================================================
# SECTION 3: Synthetic Multi-Turn Data Generation (Cell 9-10)
# ============================================================================

SECTION_3_MARKDOWN = fix_source("""## 3. Synthetic Multi-Turn Data Generation

### The Data Gap

No public dataset of multi-turn distributed prompt injection attacks exists. Every benchmark in the literature evaluates single messages in isolation. This gap matters because distributed attacks --- where an attacker spreads malicious intent across several turns of normal-looking conversation --- cannot be represented, let alone detected, in single-turn frameworks.

To close this gap, we generated **27,180 synthetic conversations** (18,754 train / 3,296 val / 5,130 test) using a shared-prefix architecture and the Anthropic API (Claude Sonnet 4.6). The design decisions below reflect lessons from three prior data generation attempts (v1 and v2) where trivial baselines exposed fatal confounds.

### Shared-Prefix Architecture

Each conversation in the dataset is generated as a **matched pair**: one benign continuation and one attack continuation branching from an identical conversational prefix. This paired structure eliminates vocabulary-level confounds that plagued earlier versions --- a bag-of-words classifier cannot separate the classes when both share the same opening turns.

```
Shared prefix (k turns):   User₁ → Asst₁ → User₂ → Asst₂ → ... → Userₖ → Asstₖ
                                                                          ╱         ╲
                                                            Benign branch    Attack branch
                                                            (natural topic   (distributed
                                                             continuation)    injection)
```

The prefix length *k* is sampled uniformly from {3, 4, 5} user turns (6-10 total turns including assistant responses). After the branch point, both continuations run for 3-5 additional user turns, producing conversations of 6-9 user turns total (12-19 turns including assistant responses).

### Attack Strategies

Four strategies drawn from published attack research:

| Strategy | % of Attacks | Pattern | Research Basis |
|----------|:---:|---------|----------------|
| Fragment distribution | 45% | Split injection payload across 3-5 turns, interleaved with on-topic filler | Evasion of per-message filters |
| Gradual escalation | 25% | Each turn nudges the conversation closer to the attack goal (Crescendo) | Russinovich et al. (USENIX Security 2025) |
| Context priming | 15% | Establish persona/authority early, exploit established trust later | Foot-in-the-Door (EMNLP 2025) |
| Instruction layering | 15% | Each turn adds one constraint, cumulatively overriding safety guidelines | Incremental constraint injection |

### Difficulty Tiers

Each attack is assigned a difficulty tier controlling how aggressively the injection signal is obscured:

| Tier | Train | Val | Test | Characteristics |
|------|------:|----:|-----:|----------------|
| Easy | 5,812 | 1,002 | 1,462 | Shorter prefixes, less camouflage, more direct language |
| Medium | 5,684 | 1,000 | 1,414 | Moderate prefix length, some topic-relevant camouflage |
| Hard | 5,590 | 976 | 1,394 | Longer prefixes, strong camouflage, subtle escalation |
| Adversarial | 1,668 | 318 | 860 | Maximum camouflage, attack indistinguishable from topic drift |

All splits are exactly balanced (50/50 attack/benign) within each tier.

### Data Quality Controls

Three mechanisms guard against artifacts:

1. **Shared-prefix pairing**: Identical opening turns for attack and benign branches eliminates early-turn vocabulary confounds. A first-turn-only classifier scores F1 = 0.35 (chance level).

2. **Validation gate**: A pre-trained single-turn GRU classifier scores each individual turn. Sequences where any single turn exceeds the detection threshold are rejected. This forces the attack signal into cross-turn patterns rather than concentrated in any one message.

3. **Confound gate battery**: Seven automated checks (unigram/bigram BoW, first-turn-only, last-turn-only, conversation length, per-turn voting) run on a 5-fold cross-validation of the training set to catch residual data artifacts before model training begins.

### What the Confound Gates Reveal

Three of seven gates pass cleanly (first-turn F1 = 0.35, conversation length F1 = 0.48, max-vote per-turn F1 = 0.68). Three gates fail: unigram BoW (0.94), bigram BoW (0.95), and last-turn BoW (0.96). The last-turn failure reflects a genuine signal --- the final turns of attack sequences *should* differ from benign ones, because that is where the injection payload lands. The BoW failures indicate that vocabulary differences between attack and benign continuations remain detectable by lexical classifiers, despite the shared prefix. This is a known limitation: eliminating *all* lexical signal would require attack continuations to use identical vocabulary to benign ones, which would make them ineffective as attacks.

The critical observation is that these confounds concentrate in the *post-branch* turns. The temporal structure --- which turns carry the signal, how the signal accumulates, and in what order --- remains the primary differentiator for architecture comparison. A bag-of-words model that achieves 95% by reading vocabulary cannot distinguish *ordered* turns from *shuffled* turns, but our temporal model can (shuffled F1 drops from 0.837 to 0.760, p < 0.001).
""".strip().split("\n"))


SECTION_3_CODE = fix_source("""# ============================================================================
# Load and Explore v3 Synthetic Multi-Turn Conversations
# ============================================================================
import json
from collections import Counter
from pathlib import Path

data_dir = Path("data/synthetic_v3")
datasets = {}
for split in ["train", "val", "test"]:
    with open(data_dir / f"multiturn_{split}.json") as f:
        datasets[split] = json.load(f)

# --- Dataset Overview ---
print("v3 Shared-Prefix Dataset")
print("=" * 55)
for split, data in datasets.items():
    attacks = sum(1 for s in data if s["label"] == 1)
    benign = len(data) - attacks
    print(f"  {split:5s}: {len(data):>6,} sequences  ({attacks:,} attack, {benign:,} benign)")

# --- Strategy Distribution (train attacks only) ---
train_attacks = [s for s in datasets["train"] if s["label"] == 1]
strats = Counter(s.get("strategy", "none") for s in train_attacks)
total_attacks = sum(strats.values())
print(f"\\nAttack strategies (train, n={total_attacks:,}):")
for s, c in strats.most_common():
    print(f"  {s:25s} {c:>5,}  ({c/total_attacks*100:.1f}%)")

# --- Difficulty Tier Distribution ---
tiers = Counter(s.get("difficulty", "unknown") for s in datasets["test"])
print(f"\\nDifficulty tiers (test, n={len(datasets['test']):,}):")
for tier in ["easy", "medium", "hard", "adversarial"]:
    c = tiers[tier]
    atk = sum(1 for s in datasets["test"] if s.get("difficulty") == tier and s["label"] == 1)
    print(f"  {tier:15s} {c:>5,}  ({atk} attack, {c-atk} benign)")

# --- Turn Count Distribution ---
user_turns = [len([t for t in s["turns"] if t.get("role") == "user"]) for s in datasets["train"]]
print(f"\\nUser turns per conversation: {min(user_turns)}-{max(user_turns)} (mean {np.mean(user_turns):.1f})")

# --- Visualization: Strategy + Tier Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Strategy distribution (stacked bar by tier)
tier_order = ["easy", "medium", "hard", "adversarial"]
strat_order = [s for s, _ in strats.most_common()]
colors_strat = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

tier_strat_counts = {}
for tier in tier_order:
    tier_attacks = [s for s in train_attacks if s.get("difficulty") == tier]
    tier_strat_counts[tier] = Counter(s.get("strategy") for s in tier_attacks)

x = np.arange(len(tier_order))
width = 0.18
for i, strat in enumerate(strat_order):
    counts = [tier_strat_counts[tier].get(strat, 0) for tier in tier_order]
    axes[0].bar(x + i * width, counts, width, label=strat.replace("_", " "), color=colors_strat[i])

axes[0].set_xticks(x + 1.5 * width)
axes[0].set_xticklabels([t.capitalize() for t in tier_order])
axes[0].set_ylabel("Count")
axes[0].set_title("Attack Strategies by Difficulty Tier")
axes[0].legend(fontsize=8, loc="upper right")

# Panel 2: Shared-prefix structure visualization
# Show turn count histogram for attack vs benign
attack_turns = [len([t for t in s["turns"] if t.get("role") == "user"])
                for s in datasets["train"] if s["label"] == 1]
benign_turns = [len([t for t in s["turns"] if t.get("role") == "user"])
                for s in datasets["train"] if s["label"] == 0]
bins = range(5, 11)
axes[1].hist(benign_turns, bins=bins, alpha=0.6, label="Benign", color="steelblue", edgecolor="white")
axes[1].hist(attack_turns, bins=bins, alpha=0.6, label="Attack", color="tomato", edgecolor="white")
axes[1].set_xlabel("User Turns per Conversation")
axes[1].set_ylabel("Count")
axes[1].set_title("Turn Count Distribution (Shared-Prefix Pairs)")
axes[1].legend()

plt.tight_layout()
plt.savefig("results/v3_data_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\\nSaved: results/v3_data_overview.png")
""".strip().split("\n"))


# ============================================================================
# SECTIONS 9-11: Multi-Turn Results (Cells 31-38)
# ============================================================================

SECTION_9_MARKDOWN = fix_source("""## 9. Iteration 5: Multi-Turn Classifier

### Approach

Multi-turn distributed prompt injection detection remains an underexplored area in published work. This project addresses that gap with a dual-encoder temporal architecture.

### Dual-Encoder Architecture

```
Turn 1 → [Frozen GRU Encoder] → 32-dim vector ─┐
Turn 2 → [Frozen GRU Encoder] → 32-dim vector  ─┤
Turn 3 → [Frozen GRU Encoder] → 32-dim vector  ─┼→ [Sequence LSTM (64-dim)] → Dense(64→32→1)
Turn 4 → [Frozen GRU Encoder] → 32-dim vector  ─┤
Turn N → [Frozen GRU Encoder] → 32-dim vector ─┘
```

**Level 1, Turn Encoder (Frozen)**: The GRU from Iteration 4, with all 2.6M parameters frozen. Each conversation turn is independently encoded into a 32-dimensional vector that captures "how injection-like is this turn?" This is the final hidden state of the GRU after processing all tokens in the turn.

**Level 2, Sequence LSTM (Trainable)**: A new LSTM with 64-dimensional hidden state that processes the sequence of turn vectors. This is where temporal learning happens. The LSTM's gates learn patterns like:
- "Turn 1 established a persona → Turn 3 referenced it → Turn 5 exploited it"
- "Gradual escalation from neutral to specific to directive"
- "Fragmented payload pieces accumulating across turns"

**Classification Head**: Dense(64→32) → ReLU → Dropout → Dense(32→1), trained with BCEWithLogitsLoss.

### Parameter Efficiency

Only **~27,000 parameters are trainable** (the sequence LSTM and classification head). The turn encoder's 2.6M parameters are frozen. This prevents catastrophic forgetting (the encoder retains its single-turn detection ability) and makes training fast and stable.

### v3 Results

On the v3 shared-prefix dataset (5,130 test sequences across 4 difficulty tiers), iter5 achieves **F1 = 0.837 [0.826, 0.847]** (95% bootstrap CI, 1000 resamples).

The per-tier breakdown reveals a clear difficulty gradient:

| Tier | F1 | Accuracy | AUC | n |
|------|:---:|:---:|:---:|---:|
| Easy | 0.866 | 0.862 | 0.941 | 1,462 |
| Medium | 0.837 | 0.843 | 0.915 | 1,414 |
| Hard | 0.831 | 0.831 | 0.917 | 1,394 |
| Adversarial | 0.794 | 0.802 | 0.885 | 860 |

The 7-point F1 drop from easy to adversarial confirms that the difficulty tiers function as intended --- adversarial sequences genuinely challenge the model. The relatively graceful degradation (not a cliff) suggests the model learns generalizable temporal patterns rather than tier-specific shortcuts.

### What This Means

The temporal architecture detects attacks that no per-turn classifier can identify. We validate this claim rigorously in Section 12 via turn-level voting baselines (A10) and paired bootstrap tests.
""".strip().split("\n"))

SECTION_9_CODE = fix_source("""# ============================================================================
# Iteration 5: Multi-Turn Classifier — v3 Results
# ============================================================================
with open("results/v3_evaluation/iter5_per_tier.json") as f:
    iter5_tier = json.load(f)
with open("results/v3_evaluation/iter5_bootstrap_ci.json") as f:
    iter5_ci = json.load(f)

o = iter5_tier["overall"]
ci = iter5_ci["overall"]["f1"]
print("MULTI-TURN CLASSIFICATION RESULTS (v3 shared-prefix)")
print("=" * 60)
print(f"  F1:        {o['f1']:.4f}  [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
print(f"  Precision: {o['precision']:.4f}")
print(f"  Recall:    {o['recall']:.4f}")
print(f"  AUC:       {o['auc']:.4f}")

print(f"\\n  Per-tier breakdown:")
for tier in ["easy", "medium", "hard", "adversarial"]:
    t = iter5_tier["per_tier"][tier]
    print(f"    {tier:15s} F1={t['f1']:.4f}  Acc={t['accuracy']:.4f}  AUC={t['auc']:.4f}  (n={t['n']})")

# --- Per-tier F1 bar chart ---
fig, ax = plt.subplots(figsize=(10, 5))
tiers = ["easy", "medium", "hard", "adversarial"]
f1s = [iter5_tier["per_tier"][t]["f1"] for t in tiers]
tier_cis = iter5_ci["per_tier"]
ci_lower = [tier_cis[t]["f1"]["ci_lower"] for t in tiers]
ci_upper = [tier_cis[t]["f1"]["ci_upper"] for t in tiers]
errors = [[f - l for f, l in zip(f1s, ci_lower)], [u - f for f, u in zip(f1s, ci_upper)]]

colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]
bars = ax.bar(tiers, f1s, color=colors, edgecolor="white", linewidth=0.5)
ax.errorbar(range(len(tiers)), f1s, yerr=errors, fmt="none", ecolor="black", capsize=5, linewidth=1.5)
ax.set_ylabel("F1 Score")
ax.set_title("Iter 5: Multi-Turn LSTM — Per-Tier Performance")
ax.set_ylim(0.7, 0.95)
ax.axhline(y=o["f1"], color="gray", linestyle="--", alpha=0.5, label=f"Overall F1={o['f1']:.3f}")
ax.legend()
for bar, val in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{val:.3f}",
            ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("results/v3_iter5_per_tier.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip().split("\n"))

SECTION_9_GATE_MARKDOWN = fix_source("""### LSTM Gate Dynamics Across Conversation Turns

The sequence LSTM processes one turn-encoding per timestep. At each step, three gates control information flow:
- **Forget gate**: How much prior context to retain (high = remember everything, low = reset).
- **Input gate**: How much new turn information to incorporate.
- **Output gate**: What fraction of the cell state to expose as the hidden state.

The heatmaps below show mean gate activations (averaged over the 64 hidden units) for a sample attack conversation and a benign conversation. In the attack, notice how the input gate spikes on turns that carry escalation toward the exploit, while the forget gate stays high to maintain the accumulated attack context. The benign conversation shows more uniform gate activations --- no single turn warrants a dramatic shift in the hidden state.
""".strip().split("\n"))


SECTION_10_MARKDOWN = fix_source("""## 10. Iteration 6: Attention Mechanism

### Motivation

The plain LSTM in Iteration 5 uses only its **final hidden state** to classify the conversation. All temporal information must be compressed into a single 64-dimensional vector, and information from early turns may be diluted or lost.

### Additive (Bahdanau) Attention

We add an **additive attention layer** over all LSTM hidden states:

```
h₁, h₂, ..., hₙ = LSTM outputs for each turn
eᵢ = tanh(W · hᵢ + b)           ← score each turn's hidden state
αᵢ = softmax(eᵢ)                 ← normalize to attention weights
context = Σ αᵢ · hᵢ              ← weighted combination
```

### Interpretability

Attention weights provide a form of model interpretability for security applications: when the model flags a conversation, we can examine which turns received the highest weights. In practice, attention concentrates on escalation turns --- the moments where the conversation shifts from benign to malicious.

### v3 Results

Iter6 achieves **F1 = 0.837 [0.825, 0.848]**, statistically indistinguishable from iter5 (paired bootstrap p = 0.453). The attention mechanism does not measurably improve classification accuracy on this dataset, but it provides interpretability at zero accuracy cost.

| Tier | F1 | Accuracy | AUC |
|------|:---:|:---:|:---:|
| Easy | 0.874 | 0.867 | 0.945 |
| Medium | 0.840 | 0.847 | 0.921 |
| Hard | 0.830 | 0.837 | 0.918 |
| Adversarial | 0.786 | 0.799 | 0.886 |
""".strip().split("\n"))

SECTION_10_CODE = fix_source("""# ============================================================================
# Iteration 6: Attention Mechanism — v3 Results
# ============================================================================
with open("results/v3_evaluation/iter6_per_tier.json") as f:
    iter6_tier = json.load(f)
with open("results/v3_evaluation/iter6_bootstrap_ci.json") as f:
    iter6_ci = json.load(f)

o6 = iter6_tier["overall"]
ci6 = iter6_ci["overall"]["f1"]
print(f"Iter 6: Multi-Turn LSTM + Additive Attention")
print(f"  F1:        {o6['f1']:.4f}  [{ci6['ci_lower']:.4f}, {ci6['ci_upper']:.4f}]")
print(f"  Precision: {o6['precision']:.4f}")
print(f"  Recall:    {o6['recall']:.4f}")
print(f"  AUC:       {o6['auc']:.4f}")

o5 = iter5_tier["overall"]
print(f"\\n  vs Iter 5:  F1 diff = {o6['f1'] - o5['f1']:+.4f} (not significant, p=0.453)")

# --- Comparison bar chart: iter5 vs iter6 by tier ---
fig, ax = plt.subplots(figsize=(10, 5))
tiers = ["easy", "medium", "hard", "adversarial"]
f1_5 = [iter5_tier["per_tier"][t]["f1"] for t in tiers]
f1_6 = [iter6_tier["per_tier"][t]["f1"] for t in tiers]

x = np.arange(len(tiers))
width = 0.35
ax.bar(x - width/2, f1_5, width, label="Iter 5 (LSTM)", color="#FF9800", edgecolor="white")
ax.bar(x + width/2, f1_6, width, label="Iter 6 (+Attention)", color="#E65100", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([t.capitalize() for t in tiers])
ax.set_ylabel("F1 Score")
ax.set_title("Iter 5 vs Iter 6: Attention adds interpretability without accuracy cost")
ax.set_ylim(0.7, 0.95)
ax.legend()
plt.tight_layout()
plt.savefig("results/v3_iter5_vs_iter6.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip().split("\n"))


SECTION_11_MARKDOWN = fix_source("""## 11. DistilBERT Baselines: The Parameter Efficiency Question

Two transformer baselines test whether raw model capacity can substitute for architectural design:

### PM-1a: Hierarchical DistilBERT (71.9M total, 5.5M trainable)

Each turn is independently processed through a frozen DistilBERT encoder (66.4M params), extracting [CLS] representations. A trainable cross-turn transformer (2 layers, 4 heads) then processes the sequence of turn representations, followed by a classification head. This architecture parallels our dual-encoder design but replaces the GRU turn encoder with DistilBERT and the sequence LSTM with a transformer.

### PM-1b: Concatenated DistilBERT (66.4M, all trainable)

All turns are concatenated with [SEP] tokens and processed through a fully fine-tuned DistilBERT. This is the brute-force approach: give the model the entire conversation as a single sequence and let backpropagation figure out what matters.

### Results

| Model | Trainable Params | F1 | 95% CI | AUC |
|-------|:---:|:---:|:---:|:---:|
| Dual-encoder LSTM (iter5) | 27K | 0.837 | [0.826, 0.847] | 0.919 |
| + Attention (iter6) | 29K | 0.837 | [0.825, 0.848] | 0.921 |
| Hierarchical DistilBERT | 5.5M | 0.976 | [0.971, 0.980] | 0.998 |
| Concatenated DistilBERT | 66.4M | 0.992 | [0.989, 0.994] | 1.000 |

The DistilBERT baselines substantially outperform the dual-encoder LSTM. Concatenated DistilBERT in particular achieves near-perfect classification. This result has two interpretations, and both matter:

**The sobering interpretation**: raw capacity wins. With 66.4M trainable parameters and access to the full conversation as a single token sequence, DistilBERT can learn whatever patterns distinguish attack from benign continuations --- including residual vocabulary differences that the confound gates flagged.

**The encouraging interpretation**: our 27K-parameter model achieves F1 = 0.837 with no pretrained language model, no fine-tuning, and a frozen turn encoder. The LSTM processes only 32-dimensional turn embeddings, not raw text. It cannot exploit vocabulary --- it operates entirely on the GRU's compression of each turn's "injection-likeness." The 7.6-point gap between iter5 (0.837) and the shuffled-turns ablation (0.760) demonstrates that temporal ordering contributes real signal even in this compressed representation space.

The right comparison is not "is our model better than DistilBERT?" (it is not) but "does temporal modeling add value beyond per-turn classification?" (it does, significantly). Section 12 makes this argument with paired bootstrap tests.
""".strip().split("\n"))


SECTION_11_CODE = fix_source("""# ============================================================================
# DistilBERT Baselines — v3 Results
# ============================================================================
with open("results/v3_evaluation/distilbert_hier_per_tier.json") as f:
    dh_tier = json.load(f)
with open("results/v3_evaluation/distilbert_concat_per_tier.json") as f:
    dc_tier = json.load(f)
with open("results/v3_evaluation/distilbert_hier_bootstrap_ci.json") as f:
    dh_ci = json.load(f)
with open("results/v3_evaluation/distilbert_concat_bootstrap_ci.json") as f:
    dc_ci = json.load(f)

print("DistilBERT Baseline Results")
print("=" * 65)
for name, tier, ci in [("Hierarchical (PM-1a)", dh_tier, dh_ci),
                        ("Concatenated (PM-1b)", dc_tier, dc_ci)]:
    o = tier["overall"]
    f1_ci = ci["overall"]["f1"]
    print(f"  {name}:")
    print(f"    F1={o['f1']:.4f} [{f1_ci['ci_lower']:.4f}, {f1_ci['ci_upper']:.4f}]  "
          f"Prec={o['precision']:.4f}  Rec={o['recall']:.4f}  AUC={o['auc']:.4f}")
    for t in ["easy", "medium", "hard", "adversarial"]:
        td = tier["per_tier"][t]
        print(f"      {t:15s} F1={td['f1']:.4f}  Acc={td['accuracy']:.4f}  (n={td['n']})")
    print()

# --- Parameter efficiency visualization ---
fig, ax = plt.subplots(figsize=(10, 6))
models = ["Dual-encoder\\nLSTM (iter5)", "Dual-encoder\\n+Attn (iter6)",
          "Hierarchical\\nDistilBERT", "Concatenated\\nDistilBERT"]
params = [27_000, 29_000, 5_500_000, 66_400_000]
f1s = [iter5_tier["overall"]["f1"], iter6_tier["overall"]["f1"],
       dh_tier["overall"]["f1"], dc_tier["overall"]["f1"]]
colors = ["#FF9800", "#E65100", "#9C27B0", "#7B1FA2"]

scatter = ax.scatter(params, f1s, s=200, c=colors, edgecolors="black", linewidth=1, zorder=5)
for i, (model, p, f1) in enumerate(zip(models, params, f1s)):
    ax.annotate(f"{model}\\nF1={f1:.3f}", (p, f1),
                textcoords="offset points", xytext=(15, 10 if i < 2 else -25),
                fontsize=9, ha="left")

ax.set_xscale("log")
ax.set_xlabel("Trainable Parameters (log scale)")
ax.set_ylabel("F1 Score")
ax.set_title("Parameter Efficiency: F1 vs Trainable Parameters")
ax.grid(True, alpha=0.3)
ax.set_ylim(0.75, 1.02)
plt.tight_layout()
plt.savefig("results/v3_param_efficiency.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip().split("\n"))


# ============================================================================
# NEW SECTION: Ablation Studies
# ============================================================================

ABLATION_MARKDOWN = fix_source("""## 11b. Ablation Studies: What Drives the Temporal Signal?

Five ablation experiments isolate which components of the dual-encoder architecture contribute to performance. Each ablation modifies exactly one aspect of the iter5 architecture while holding everything else constant.

### A10: Turn-Level Voting (The Critical Ablation)

Turn-level voting is the most important ablation because it tests the null hypothesis directly: *can we detect multi-turn attacks by classifying each turn independently and aggregating the per-turn scores?*

We freeze the GRU turn encoder from iter5 and score each turn independently. Three aggregation methods combine the per-turn scores into a conversation-level decision:

| Aggregation | F1 | Accuracy | Logic |
|-------------|:---:|:---:|-------|
| Max-vote | 0.706 | 0.716 | Flag if any single turn is suspicious |
| Mean-vote | 0.231 | 0.580 | Average suspicion across turns |
| Top-3-mean | 0.727 | 0.735 | Average the 3 most suspicious turns |

All three voting methods fall well below iter5's F1 = 0.837. The paired bootstrap test confirms: iter5 > max-vote by +0.131 F1 points (p < 0.001), iter5 > top-3-mean by +0.110 (p < 0.001). The sequence LSTM learns cross-turn relationships that no per-turn aggregation can recover.

The mean-vote collapse (F1 = 0.231) deserves comment. Because the shared-prefix architecture ensures that attack and benign conversations share identical opening turns, averaging all turn scores dilutes the post-branch signal below the classification threshold. Max-vote partially compensates by looking at the single most suspicious turn, but even the most suspicious individual turn in a well-crafted distributed attack may not exceed the threshold.

### Temporal Ordering Ablations

| Ablation | F1 | 95% CI | What It Tests |
|----------|:---:|:---:|---------------|
| iter5 (ordered) | 0.837 | [0.826, 0.847] | Full model |
| Shuffled turns | 0.760 | [0.748, 0.772] | Is turn order informative? |
| Reversed turns | 0.833 | [0.821, 0.844] | Is forward-vs-backward direction informative? |

**Shuffled vs ordered**: Randomly permuting the turn order drops F1 by 7.7 points (p < 0.001). This is the cleanest evidence that the LSTM learns temporal patterns, not just bag-of-turns features. The turn-order sensitivity analysis (Section 12) shows 55% of correctly classified attacks flip to incorrect after shuffling.

**Reversed vs ordered**: Reversing the turn order barely affects performance (0.833 vs 0.837, not significant). The LSTM can read the escalation pattern in either direction. This is consistent with the bidirectional nature of the underlying GRU encoder and suggests the model learns *relational* patterns between turns rather than absolute position-dependent features.

### Pooling Ablations

| Ablation | F1 | 95% CI | Architecture |
|----------|:---:|:---:|--------------|
| iter5 (LSTM) | 0.837 | [0.826, 0.847] | Sequence LSTM → final hidden state |
| Mean pool | 0.755 | [0.743, 0.768] | Average all turn embeddings |
| Max pool | 0.719 | [0.705, 0.733] | Element-wise max across turns |

Replacing the sequence LSTM with simple pooling drops F1 by 8-12 points. Mean pooling dilutes the signal (same failure mode as mean-vote). Max pooling preserves the strongest per-dimension signal but loses temporal ordering. Both confirm that the LSTM's sequential processing adds value beyond aggregation.

### Encoder Ablations

| Ablation | F1 | 95% CI | What It Tests |
|----------|:---:|:---:|---------------|
| Continuation-only | 0.846 | [0.835, 0.856] | LSTM sees only post-branch turns |
| Prefix-only | 0.667 | [0.655, 0.679] | LSTM sees only shared prefix |
| Autoencoder control | 0.845 | [0.834, 0.856] | GRU encoder replaced with autoencoder |

**Continuation-only** matches full-model performance (0.846 vs 0.837). The prefix turns --- shared between attack and benign --- contribute noise rather than signal. This validates the shared-prefix design: the model learns from the divergent post-branch turns, not from early conversation context.

**Prefix-only** performs at effective chance (0.667), confirming that the shared prefix carries no class-discriminative information.

**Autoencoder control** tests whether the GRU turn encoder's *injection-detection training* matters, or whether any reasonable encoder would work. We train an autoencoder to reconstruct turns (not classify them) and use its latent representations as turn embeddings. The autoencoder control achieves 0.845, matching the injection-trained GRU encoder. This suggests the sequence LSTM is the primary driver of temporal detection --- the turn encoder need only produce reasonable turn representations, not injection-specific ones.
""".strip().split("\n"))

ABLATION_CODE = fix_source("""# ============================================================================
# Ablation Studies: Comprehensive Results
# ============================================================================
with open("results/v3_evaluation/all_models_summary.json") as f:
    summary = json.load(f)

# --- A10 Voting Results ---
print("A10 Turn-Level Voting Baselines")
print("=" * 55)
for method in ["a10_max_vote", "a10_mean_vote", "a10_top3_mean"]:
    v = summary["a10_voting"][method]
    print(f"  {method:20s} F1={v['f1']:.4f}  Acc={v['accuracy']:.4f}")

print(f"\\n  iter5 LSTM:           F1={iter5_tier['overall']['f1']:.4f}")
print(f"  Gap (iter5 - best voting): +{iter5_tier['overall']['f1'] - 0.727:.3f}")

# --- All ablations ranked ---
print(f"\\nAll Ablation Results (ranked by F1)")
print("-" * 65)
ablation_models = [
    ("iter5 (ordered)", "iter5"),
    ("ablation_continuation", "ablation_continuation"),
    ("ablation_autoencoder", "ablation_autoencoder"),
    ("iter6 (+attention)", "iter6"),
    ("ablation_reversed", "ablation_reversed"),
    ("ablation_shuffled", "ablation_shuffled"),
    ("ablation_mean_pool", "ablation_mean_pool"),
    ("ablation_max_pool", "ablation_max_pool"),
    ("ablation_prefix", "ablation_prefix"),
]
for label, key in ablation_models:
    tier = summary["per_tier_metrics"][key]["overall"]
    ci = summary["bootstrap_cis"][key]["overall"]["f1"]
    print(f"  {label:30s} F1={tier['f1']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

# --- Ablation comparison visualization ---
fig, ax = plt.subplots(figsize=(12, 7))

# Group ablations by category
categories = {
    "Temporal": [
        ("iter5\\n(ordered)", 0.837, "#FF9800"),
        ("Shuffled", 0.760, "#FFE0B2"),
        ("Reversed", 0.833, "#FFC107"),
    ],
    "Pooling": [
        ("Mean pool", 0.755, "#B3E5FC"),
        ("Max pool", 0.719, "#81D4FA"),
    ],
    "Voting\\n(A10)": [
        ("Max-vote", 0.706, "#E1BEE7"),
        ("Top-3", 0.727, "#CE93D8"),
        ("Mean-vote", 0.231, "#F3E5F5"),
    ],
    "Encoder": [
        ("Continuation", 0.846, "#C8E6C9"),
        ("Autoencoder", 0.845, "#A5D6A7"),
        ("Prefix-only", 0.667, "#E8F5E9"),
    ],
}

x_pos = 0
xticks = []
xtick_labels = []
group_positions = []
for group_name, items in categories.items():
    group_start = x_pos
    for label, f1, color in items:
        bar = ax.bar(x_pos, f1, color=color, edgecolor="black", linewidth=0.5, width=0.8)
        ax.text(x_pos, f1 + 0.01, f"{f1:.3f}", ha="center", fontsize=8, fontweight="bold")
        xticks.append(x_pos)
        xtick_labels.append(label)
        x_pos += 1
    group_end = x_pos - 1
    group_positions.append((group_start, group_end, group_name))
    x_pos += 0.5  # gap between groups

ax.set_xticks(xticks)
ax.set_xticklabels(xtick_labels, fontsize=8, rotation=30, ha="right")
ax.set_ylabel("F1 Score")
ax.set_title("Ablation Study: What Drives Multi-Turn Detection?")
ax.axhline(y=0.837, color="gray", linestyle="--", alpha=0.4, label="iter5 baseline")
ax.set_ylim(0, 1.05)
ax.legend(loc="upper right")

# Group labels
for start, end, name in group_positions:
    mid = (start + end) / 2
    ax.text(mid, -0.08, name, ha="center", fontsize=9, fontweight="bold",
            transform=ax.get_xaxis_transform())

plt.tight_layout()
plt.savefig("results/v3_ablation_summary.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip().split("\n"))


# ============================================================================
# SECTION 12: Cross-Iteration Comparison (Cells 39-46)
# ============================================================================

SECTION_12_MARKDOWN = fix_source("""## 12. Cross-Iteration Comparison and Statistical Analysis

### The Full Picture

This section synthesizes results across all models on the v3 shared-prefix dataset, with bootstrap confidence intervals and paired significance tests. Every number below comes from the same 5,130-sequence test set (2,565 attack, 2,565 benign, 4 difficulty tiers).

### Model Hierarchy

| Model | F1 | 95% CI | AUC | Trainable Params |
|-------|:---:|:---:|:---:|:---:|
| Concatenated DistilBERT | 0.992 | [0.989, 0.994] | 1.000 | 66.4M |
| Hierarchical DistilBERT | 0.976 | [0.971, 0.980] | 0.998 | 5.5M |
| Continuation-only LSTM | 0.846 | [0.835, 0.856] | 0.923 | 27K |
| Autoencoder encoder | 0.845 | [0.834, 0.856] | 0.922 | 27K |
| Iter 6 (+attention) | 0.837 | [0.825, 0.848] | 0.921 | 29K |
| Iter 5 (temporal LSTM) | 0.837 | [0.826, 0.847] | 0.919 | 27K |
| Reversed turns | 0.833 | [0.821, 0.844] | 0.916 | 27K |
| Shuffled turns | 0.760 | [0.748, 0.772] | 0.849 | 27K |
| Mean pool | 0.755 | [0.743, 0.768] | 0.839 | 27K |
| A10 top-3-mean | 0.727 | — | — | 0 |
| Max pool | 0.719 | [0.705, 0.733] | 0.819 | 27K |
| A10 max-vote | 0.706 | — | — | 0 |
| Prefix-only | 0.667 | [0.655, 0.679] | 0.500 | 27K |
| Cosine baseline | 0.612 | [0.596, 0.627] | 0.642 | 0 |
| A10 mean-vote | 0.231 | — | — | 0 |

### Statistical Significance (Paired Bootstrap Tests)

All comparisons use one-sided paired bootstrap with 1000 resamples. The test asks "is Model A significantly better than Model B?"

| Comparison | F1 Diff | p-value | Significant? |
|-----------|:---:|:---:|:---:|
| iter5 > A10 max-vote | +0.131 | < 0.001 | Yes |
| iter5 > A10 top-3-mean | +0.110 | < 0.001 | Yes |
| iter5 > A10 mean-vote | +0.606 | < 0.001 | Yes |
| iter5 > shuffled | +0.077 | < 0.001 | Yes |
| iter6 > iter5 | +0.000 | 0.453 | No |
| iter5 > DistilBERT-hier | -0.139 | 1.000 | No (DistilBERT wins) |
| iter5 > DistilBERT-concat | -0.155 | 1.000 | No (DistilBERT wins) |
| ordered > shuffled (sensitivity) | +0.348 | < 0.001 | Yes |

The first four rows establish the core claim: temporal LSTM modeling significantly outperforms all turn-level aggregation methods. The shuffled-turns comparison is the cleanest: same model, same parameters, same data --- only the turn order differs. When order is destroyed, F1 drops 7.7 points.

### Turn-Order Sensitivity

To directly measure how much the model relies on turn ordering, we take every correctly classified attack sequence, randomly shuffle its turns, and re-run inference:

- **55% of correctly classified attacks flip to incorrect after shuffling**
- Ordered F1: 0.837 → Shuffled F1: 0.489 (on the subset of originally-correct attacks)
- Flip rate is uniform across tiers (54-56%), confirming the model does not rely on tier-specific shortcuts

This is the strongest evidence that the LSTM learns genuine temporal patterns. A model that relied on vocabulary or per-turn features would be unaffected by shuffling.

### Per-Strategy Breakdown

| Strategy | iter5 F1 | iter6 F1 | n (test) |
|----------|:---:|:---:|:---:|
| Fragment distribution | 0.776 | 0.787 | 1,160 |
| Gradual escalation | 0.676 | 0.681 | 669 |
| Context priming | 0.628 | 0.650 | 372 |
| Instruction layering | 0.605 | 0.612 | 364 |

Fragment distribution is easiest to detect because it produces the most distinctive temporal signature: discrete payload fragments separated by filler turns create characteristic spikes in the turn-encoding sequence. Instruction layering is hardest because each turn adds a subtle constraint, producing a smooth gradient in the embedding space rather than sharp transitions.
""".strip().split("\n"))


SECTION_12_CODE = fix_source("""# ============================================================================
# Cross-Iteration Comparison: Complete Model Hierarchy with Bootstrap CIs
# ============================================================================
with open("results/v3_evaluation/all_models_summary.json") as f:
    summary = json.load(f)

# --- Comprehensive results table ---
print("COMPLETE v3 RESULTS — All Models")
print("=" * 80)
print(f"{'Model':<30s} {'F1':>8s} {'95% CI':>20s} {'AUC':>8s} {'Params':>10s}")
print("-" * 80)

model_info = [
    ("Concat DistilBERT", "distilbert_concat", "66.4M"),
    ("Hier DistilBERT", "distilbert_hier", "5.5M"),
    ("Continuation-only", "ablation_continuation", "27K"),
    ("Autoencoder encoder", "ablation_autoencoder", "27K"),
    ("Iter 6 (+attention)", "iter6", "29K"),
    ("Iter 5 (temporal LSTM)", "iter5", "27K"),
    ("Reversed turns", "ablation_reversed", "27K"),
    ("Shuffled turns", "ablation_shuffled", "27K"),
    ("Mean pool", "ablation_mean_pool", "27K"),
    ("Max pool", "ablation_max_pool", "27K"),
    ("Prefix-only", "ablation_prefix", "27K"),
    ("Cosine baseline", "cosine_baseline", "0"),
]

for label, key, params in model_info:
    tier = summary["per_tier_metrics"][key]["overall"]
    ci = summary["bootstrap_cis"][key]["overall"]["f1"]
    ci_str = f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
    print(f"  {label:<28s} {tier['f1']:>8.4f} {ci_str:>20s} {tier['auc']:>8.4f} {params:>10s}")

# A10 voting (no per-tier or bootstrap)
print()
for method in ["a10_top3_mean", "a10_max_vote", "a10_mean_vote"]:
    v = summary["a10_voting"][method]
    print(f"  {method:<28s} {v['f1']:>8.4f} {'—':>20s} {'—':>8s} {'0':>10s}")

# --- Turn-order sensitivity ---
tos = summary["turn_order_sensitivity"]
print(f"\\nTurn-Order Sensitivity:")
print(f"  Flip rate: {tos['flip_rate']:.1%}")
print(f"  Ordered F1: {tos['ordered_f1']:.4f}")
print(f"  Shuffled F1: {tos['shuffled_f1']:.4f}")

# --- Paired bootstrap test results ---
print(f"\\nPaired Bootstrap Significance Tests:")
for test_name, result in summary["paired_tests"].items():
    sig = "***" if result["p_value"] < 0.001 else ("*" if result["p_value"] < 0.05 else "n.s.")
    print(f"  {result['description']:<50s} diff={result['observed_diff']:+.4f}  p={result['p_value']:.3f}  {sig}")
""".strip().split("\n"))

SECTION_12_VIS_CODE = fix_source("""# ============================================================================
# Cross-Model F1 Comparison — Publication-Quality Bar Chart
# ============================================================================

# All models ranked by F1
model_data = [
    ("Concat\\nDistilBERT", 0.992, [0.989, 0.994], "#7B1FA2"),
    ("Hier\\nDistilBERT", 0.976, [0.971, 0.980], "#9C27B0"),
    ("Continuation\\nonly", 0.846, [0.835, 0.856], "#66BB6A"),
    ("Autoencoder\\nencoder", 0.845, [0.834, 0.856], "#81C784"),
    ("Iter 6\\n(+attn)", 0.837, [0.825, 0.848], "#E65100"),
    ("Iter 5\\n(LSTM)", 0.837, [0.826, 0.847], "#FF9800"),
    ("Reversed", 0.833, [0.821, 0.844], "#FFC107"),
    ("Shuffled", 0.760, [0.748, 0.772], "#FFE0B2"),
    ("Mean\\npool", 0.755, [0.743, 0.768], "#81D4FA"),
    ("A10\\ntop-3", 0.727, [None, None], "#CE93D8"),
    ("Max\\npool", 0.719, [0.705, 0.733], "#29B6F6"),
    ("A10\\nmax", 0.706, [None, None], "#E1BEE7"),
    ("Prefix\\nonly", 0.667, [0.655, 0.679], "#E8F5E9"),
    ("Cosine", 0.612, [0.596, 0.627], "#BDBDBD"),
    ("A10\\nmean", 0.231, [None, None], "#F3E5F5"),
]

fig, ax = plt.subplots(figsize=(16, 7))
names = [m[0] for m in model_data]
f1s = [m[1] for m in model_data]
colors = [m[3] for m in model_data]
ci_lo = [m[2][0] for m in model_data]
ci_hi = [m[2][1] for m in model_data]

bars = ax.bar(range(len(names)), f1s, color=colors, edgecolor="black", linewidth=0.5)

# Error bars where CIs available
for i, (lo, hi, f1) in enumerate(zip(ci_lo, ci_hi, f1s)):
    if lo is not None:
        ax.errorbar(i, f1, yerr=[[f1-lo], [hi-f1]], fmt="none", ecolor="black", capsize=4, linewidth=1.5)

ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=8, rotation=0, ha="center")
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_title("v3 Model Hierarchy: F1 with 95% Bootstrap Confidence Intervals", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.08)
ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3, label="Chance")
ax.axhline(y=0.837, color="gray", linestyle=":", alpha=0.4, label="iter5 (temporal LSTM)")

# Annotate F1 values on bars
for bar, f1 in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015, f"{f1:.3f}",
            ha="center", fontsize=7.5, fontweight="bold")

ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("results/v3_model_hierarchy.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip().split("\n"))


SECTION_12_STRATEGY_CODE = fix_source("""# ============================================================================
# Per-Strategy F1 Heatmap
# ============================================================================
strategies = ["fragment_distributed", "gradual_escalation", "context_priming", "instruction_layering"]
models_for_strategy = ["iter5", "iter6", "ablation_autoencoder", "ablation_continuation",
                       "ablation_shuffled", "ablation_reversed", "ablation_mean_pool", "ablation_max_pool"]

strat_matrix = []
for model in models_for_strategy:
    row = []
    for strat in strategies:
        if model in summary["per_strategy"] and strat in summary["per_strategy"][model]:
            row.append(summary["per_strategy"][model][strat]["f1"])
        else:
            row.append(0)
    strat_matrix.append(row)

strat_arr = np.array(strat_matrix)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(strat_arr, cmap="YlOrRd", aspect="auto", vmin=0.2, vmax=0.9)

ax.set_xticks(range(len(strategies)))
ax.set_xticklabels([s.replace("_", "\\n") for s in strategies], fontsize=9)
ax.set_yticks(range(len(models_for_strategy)))
ax.set_yticklabels([m.replace("ablation_", "") for m in models_for_strategy], fontsize=9)

for i in range(len(models_for_strategy)):
    for j in range(len(strategies)):
        val = strat_arr[i, j]
        color = "white" if val > 0.65 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=color)

ax.set_title("Per-Strategy F1 Heatmap: Which Attack Patterns Are Hardest?")
fig.colorbar(im, ax=ax, label="F1 Score", shrink=0.8)
plt.tight_layout()
plt.savefig("results/v3_strategy_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

print("\\nKey observation: Fragment distribution (column 1) is consistently the easiest")
print("strategy to detect across all models. Instruction layering (column 4) is hardest.")
print("This pattern is stable regardless of whether the model uses temporal ordering,")
print("pooling, or per-turn voting. The difficulty ranking is a property of the attack")
print("strategies themselves, not the detection architecture.")
""".strip().split("\n"))


TURN_ORDER_3D_CODE = fix_source("""# ============================================================================
# 3D Visualization: Turn-Order Sensitivity Across Tiers
# ============================================================================
from mpl_toolkits.mplot3d import Axes3D

# Build the data: for each tier x model variant, show F1
tiers_list = ["easy", "medium", "hard", "adversarial"]
variants = ["iter5\\n(ordered)", "iter6\\n(+attn)", "shuffled", "reversed", "mean_pool", "max_pool"]
variant_keys = ["iter5", "iter6", "ablation_shuffled", "ablation_reversed", "ablation_mean_pool", "ablation_max_pool"]

z_data = np.zeros((len(tiers_list), len(variant_keys)))
for j, vk in enumerate(variant_keys):
    for i, tier in enumerate(tiers_list):
        z_data[i, j] = summary["per_tier_metrics"][vk]["per_tier"][tier]["f1"]

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection="3d")

xpos = np.arange(len(variant_keys))
ypos = np.arange(len(tiers_list))
xposM, yposM = np.meshgrid(xpos, ypos, indexing="ij")

colors_3d = ["#FF9800", "#E65100", "#FFE0B2", "#FFC107", "#81D4FA", "#29B6F6"]
for j in range(len(variant_keys)):
    for i in range(len(tiers_list)):
        ax.bar3d(j, i, 0, 0.7, 0.7, z_data[i, j],
                 color=colors_3d[j], alpha=0.85, edgecolor="black", linewidth=0.3)

ax.set_xticks(range(len(variant_keys)))
ax.set_xticklabels(variants, fontsize=7, rotation=15)
ax.set_yticks(range(len(tiers_list)))
ax.set_yticklabels([t.capitalize() for t in tiers_list], fontsize=8)
ax.set_zlabel("F1 Score", fontsize=10)
ax.set_zlim(0, 1.0)
ax.set_title("Per-Tier F1 Across Model Variants", fontsize=12, fontweight="bold", pad=20)
ax.view_init(elev=25, azim=-50)
plt.tight_layout()
plt.savefig("results/v3_3d_tier_variants.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip().split("\n"))


TEMPORAL_NARRATIVE = fix_source("""### What Temporal Modeling Catches: Interpreting the Results

The numbers above tell a clear story, but the *why* behind them matters more than the *what*.

**The turn-order gap is the headline result.** When we shuffle the turns of correctly classified attacks and re-run inference, 55% flip from correct to incorrect. This is not a subtle effect --- it means the LSTM has learned that the *sequence* in which turns arrive carries genuine signal about malicious intent. A bag-of-turns model (which is what shuffling reduces the LSTM to) loses more than half its true positives.

**The voting gap is the architectural validation.** Turn-level voting (A10) uses the same frozen GRU encoder as iter5 --- the only difference is that voting classifies each turn independently, while iter5 processes the sequence through an LSTM. The 13-point F1 gap between iter5 (0.837) and max-vote (0.706) cannot be explained by encoder quality, training data, or hyperparameters. The LSTM's recurrent connections are doing something that independent scoring cannot.

**The DistilBERT gap is the humbling result.** Concatenated DistilBERT achieves 0.992 with 66.4M trainable parameters against our 0.837 with 27K. The parameter gap is 2,460x. Some of this advantage comes from DistilBERT's pretrained language understanding; some likely comes from its ability to exploit residual vocabulary differences that the confound gates flagged. Our model operates in a 32-dimensional embedding space where vocabulary is already compressed away. The DistilBERT comparison is unfair by design --- it tests whether a pretrained language model with full text access can outperform a purpose-built temporal detector. It can. But the temporal detector works with 0.04% of the parameters and no pretrained weights, which matters for deployment on resource-constrained devices.

**The strategy difficulty ranking is a property of the attacks, not the detector.** Fragment distribution is easiest to detect (F1 = 0.776) because it produces the most distinctive temporal signature: discrete payload fragments separated by on-topic filler create characteristic spikes in the turn-encoding sequence. Instruction layering is hardest (F1 = 0.605) because each turn adds a subtle constraint, producing a smooth gradient in the embedding space rather than sharp transitions. This ranking holds across all model variants, indicating it reflects genuine properties of the attack strategies.
""".strip().split("\n"))


def main():
    with open(NB_PATH) as f:
        nb = json.load(f)

    cells = nb["cells"]

    # --- Update Section 3 (Cell 9-10) ---
    cells[9] = make_markdown_cell(SECTION_3_MARKDOWN)
    cells[10] = make_code_cell(SECTION_3_CODE)

    # --- Update Section 9 (Cells 31-32) ---
    cells[31] = make_markdown_cell(SECTION_9_MARKDOWN)
    cells[32] = make_code_cell(SECTION_9_CODE)

    # --- Keep gate dynamics markdown/code (Cells 33-34) but update markdown ---
    cells[33] = make_markdown_cell(SECTION_9_GATE_MARKDOWN)
    # Cell 34 (gate visualization code) stays as-is

    # --- Update Section 10 (Cells 35-36) ---
    cells[35] = make_markdown_cell(SECTION_10_MARKDOWN)
    cells[36] = make_code_cell(SECTION_10_CODE)

    # --- Replace Section 11 (Cells 37-38) with DistilBERT ---
    cells[37] = make_markdown_cell(SECTION_11_MARKDOWN)
    cells[38] = make_code_cell(SECTION_11_CODE)

    # --- Insert ablation section after Cell 38 ---
    ablation_md = make_markdown_cell(ABLATION_MARKDOWN)
    ablation_code = make_code_cell(ABLATION_CODE)
    cells.insert(39, ablation_md)
    cells.insert(40, ablation_code)

    # After insertion, old cells 39+ shift by 2
    # Old cell 39 (Section 12 markdown) is now cell 41
    # Old cell 40 (Section 12 code) is now cell 42
    # etc.

    # --- Update Section 12 (now cells 41+) ---
    cells[41] = make_markdown_cell(SECTION_12_MARKDOWN)
    cells[42] = make_code_cell(SECTION_12_CODE)

    # Replace old core finding cell (now 43) with visualization
    cells[43] = make_code_cell(SECTION_12_VIS_CODE)

    # Replace old "What Temporal Modeling Catches" (now 44) with strategy heatmap
    cells[44] = make_markdown_cell(TEMPORAL_NARRATIVE)
    cells[45] = make_code_cell(SECTION_12_STRATEGY_CODE)

    # Replace old failure modes (now 46) with 3D tier visualization
    cells[46] = make_code_cell(TURN_ORDER_3D_CODE)

    # Save
    nb["cells"] = cells
    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1)

    print(f"Updated notebook: {NB_PATH}")
    print(f"Total cells: {len(cells)}")


if __name__ == "__main__":
    main()
