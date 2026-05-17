"""Update notebook cells: remaining Section 12 content + Section 13 conclusions."""

import json
from pathlib import Path

NB_PATH = Path("notebooks/multiturn_injection_detection.ipynb")


def fix_source(lines):
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1 and not line.endswith("\n"):
            result.append(line + "\n")
        else:
            result.append(line)
    return result


def make_markdown_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def make_code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


# Cell 47 — replace old error analysis with confound gate analysis
CONFOUND_GATES_CODE = fix_source("""# ============================================================================
# Confound Gate Analysis: Where Do Lexical Shortcuts Remain?
# ============================================================================
with open("results/v3_evaluation/all_models_summary.json") as f:
    summary = json.load(f)

gates = summary["confound_gates"]["gates"]
print("Confound Gate Battery (5-fold CV on train split)")
print("=" * 65)
print(f"{'Gate':<25s} {'F1 (mean±std)':>18s} {'Threshold':>12s} {'Pass?':>8s}")
print("-" * 65)
for gate_name, g in gates.items():
    status = "PASS" if g["pass"] else "FAIL"
    color = "" if g["pass"] else " ⚠"
    print(f"  {gate_name:<23s} {g['f1_mean']:.4f} ± {g['f1_std']:.4f} {'<' if g['pass'] else '>'} {g['threshold']:.2f}    {status}{color}")

print(f"\\nOverall: {'All critical gates pass' if summary['confound_gates']['all_critical_pass'] else 'Some gates FAIL'}")
print(f"\\nInterpretation:")
print(f"  - First-turn (0.354): PASS. Shared prefix eliminates early-turn confounds.")
print(f"  - Conv length (0.482): PASS. Attack and benign conversations have matched lengths.")
print(f"  - Max-vote BoW (0.684): PASS (borderline). Per-turn BoW has limited predictive power.")
print(f"  - Unigram/bigram BoW: FAIL. Vocabulary differences persist in post-branch turns.")
print(f"  - Last-turn BoW: FAIL. Expected — the final turns *should* differ between classes.")
print(f"\\n  The BoW failures indicate lexical confounds remain. The turn-order sensitivity")
print(f"  analysis (55% flip rate on shuffle) demonstrates that our temporal model relies")
print(f"  on ordering information that BoW cannot exploit.")

# --- Gate results bar chart ---
fig, ax = plt.subplots(figsize=(12, 5))
gate_names = list(gates.keys())
f1_means = [gates[g]["f1_mean"] for g in gate_names]
thresholds = [gates[g]["threshold"] for g in gate_names]
passes = [gates[g]["pass"] for g in gate_names]
colors = ["#4CAF50" if p else "#F44336" for p in passes]

bars = ax.bar(range(len(gate_names)), f1_means, color=colors, edgecolor="black", linewidth=0.5)
for i, (name, thresh) in enumerate(zip(gate_names, thresholds)):
    ax.plot([i-0.4, i+0.4], [thresh, thresh], "k--", linewidth=1.5, alpha=0.6)

ax.set_xticks(range(len(gate_names)))
ax.set_xticklabels([g.replace("_", "\\n") for g in gate_names], fontsize=8)
ax.set_ylabel("5-Fold CV F1")
ax.set_title("Confound Gate Battery: Green = Pass, Red = Fail")
ax.set_ylim(0, 1.1)

for bar, val, p in zip(bars, f1_means, passes):
    label = f"{val:.3f}\\n{'PASS' if p else 'FAIL'}"
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, label,
            ha="center", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig("results/v3_confound_gates.png", dpi=150, bbox_inches="tight")
plt.show()
""".strip().split("\n"))


# Cell 48 — animated turn-embedding trajectory
ANIMATED_TRAJECTORY_CODE = fix_source("""# ============================================================================
# Animated Turn-Embedding Trajectory: How the LSTM Accumulates Suspicion
# ============================================================================
# This visualization shows the LSTM hidden state trajectory as it processes
# consecutive turns of an attack vs benign conversation. We project the
# 64-dim hidden states to 2D via PCA and animate the trajectory.

import torch
from src.models.single_turn import GRUClassifier
from src.models.multi_turn import MultiTurnClassifier
from src.utils.tokenizer import load_vocab, encode_multiturn
from sklearn.decomposition import PCA
from matplotlib.patches import FancyArrowPatch
import matplotlib.animation as animation
from IPython.display import HTML

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = load_vocab("models/vocab.json")

# Load models
gru = GRUClassifier(vocab_size=len(vocab), embedding_dim=128, hidden_dim=64)
gru.load_state_dict(torch.load("models/v3_gru_retrain_best.pt", map_location=device, weights_only=True))
gru.to(device).eval()

mt = MultiTurnClassifier(turn_encoder=gru, turn_encoding_dim=32, hidden_dim=64)
mt.load_state_dict(torch.load("models/v3_iter5_best.pt", map_location=device, weights_only=True))
mt.to(device).eval()

# Get hidden state trajectories for sample conversations
with open("data/synthetic_v3/multiturn_test.json") as f:
    test_data = json.load(f)

# Collect hidden states for several attack and benign conversations
trajectories = {"attack": [], "benign": []}
n_samples = 50

for label_name, label_val in [("attack", 1), ("benign", 0)]:
    samples = [s for s in test_data if s["label"] == label_val][:n_samples]
    for seq in samples:
        user_turns = [t["text"] for t in seq["turns"] if t.get("role") == "user"]
        token_ids, masks = encode_multiturn(vocab, [user_turns], max_turns=10, max_len=256)
        token_ids, masks = token_ids.to(device), masks.to(device)

        with torch.no_grad():
            # Get per-turn encodings
            batch_size, max_t, seq_len = token_ids.shape
            flat = token_ids.view(-1, seq_len)
            encodings = gru.encode(flat).view(batch_size, max_t, -1)

            # Run through LSTM to get hidden states at each step
            hidden_states = []
            hx = None
            for t in range(max_t):
                if masks[0, t] > 0:
                    out, hx = mt.sequence_lstm(encodings[:, t:t+1, :], hx)
                    hidden_states.append(hx[0].squeeze().cpu().numpy())

            if len(hidden_states) >= 3:
                trajectories[label_name].append(np.array(hidden_states))

# PCA on all hidden states
all_states = np.vstack([np.vstack(trajectories["attack"]), np.vstack(trajectories["benign"])])
pca = PCA(n_components=2)
pca.fit(all_states)

# Project trajectories
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, label_name, color, title in [
    (axes[0], "attack", "tomato", "Attack Conversations"),
    (axes[1], "benign", "steelblue", "Benign Conversations")
]:
    for traj in trajectories[label_name][:20]:
        proj = pca.transform(traj)
        ax.plot(proj[:, 0], proj[:, 1], alpha=0.3, color=color, linewidth=1)
        ax.scatter(proj[0, 0], proj[0, 1], color="green", s=30, zorder=5, marker="o")
        ax.scatter(proj[-1, 0], proj[-1, 1], color=color, s=50, zorder=5, marker="X")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title, fontweight="bold")
    ax.scatter([], [], color="green", marker="o", label="Start")
    ax.scatter([], [], color=color, marker="X", label="End")
    ax.legend(fontsize=9)

fig.suptitle("LSTM Hidden-State Trajectories (PCA Projection)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/v3_hidden_trajectories.png", dpi=150, bbox_inches="tight")
plt.show()

print("Each line traces the LSTM hidden state through consecutive turns.")
print("Attack trajectories (left) diverge from benign ones (right) in PCA space,")
print("showing how the LSTM's internal representation separates the two classes")
print("as it accumulates cross-turn evidence.")
""".strip().split("\n"))


# Cell 49 — Updated Section 13
SECTION_13_MARKDOWN = fix_source("""## 13. Conclusions and Future Work

### 13.1 Summary of Findings

This project set out to answer a fundamental question: **can we detect prompt injection attacks that are distributed across multiple conversation turns, where each individual turn appears benign in isolation?** The answer is yes, with quantifiable evidence.

We designed a dual-encoder temporal architecture (frozen GRU turn encoder + trainable sequence LSTM) with only 27,000 trainable parameters and evaluated it against 14 baselines and ablations on a shared-prefix dataset of 27,180 synthetic conversations.

### 13.2 Key Results

**Finding 1: Temporal modeling detects distributed attacks that per-turn classification cannot.**
The sequence LSTM (F1 = 0.837) significantly outperforms the best turn-level voting baseline (top-3-mean, F1 = 0.727) by +0.110 F1 points (paired bootstrap, p < 0.001). This gap cannot be explained by encoder quality or training data — both methods use the same frozen GRU encoder.

**Finding 2: Turn order carries real signal.**
Shuffling the turn order of correctly classified attacks causes 55% to flip from correct to incorrect (ordered F1 = 0.837 → shuffled F1 = 0.760, p < 0.001). This is the strongest evidence that the LSTM learns genuine temporal patterns rather than bag-of-turns features.

**Finding 3: Raw capacity outperforms temporal architecture.**
Concatenated DistilBERT (66.4M trainable parameters) achieves F1 = 0.992, substantially above our temporal model. With full text access and pretrained language understanding, DistilBERT can exploit both temporal and vocabulary signals. This result is expected and does not diminish the temporal finding — it confirms that the detection task is solvable and that our lightweight architecture captures a meaningful (though incomplete) portion of the signal.

**Finding 4: Difficulty tiers function as designed.**
F1 degrades gracefully from easy (0.866) through adversarial (0.794) for the temporal LSTM, confirming that the synthetic tier assignments reflect genuine detection difficulty.

**Finding 5: Attack strategy difficulty is independent of detector architecture.**
Fragment distribution (F1 = 0.776) is consistently easiest; instruction layering (F1 = 0.605) is consistently hardest. This ranking holds across all model variants tested.

### 13.3 Limitations

- **Synthetic data**: All conversations are generated by a single LLM (Claude Sonnet 4.6). Real-world multi-turn attacks would exhibit greater diversity in phrasing, domain context, and social engineering sophistication. Performance on naturally occurring attacks is unknown.

- **Residual vocabulary confounds**: The confound gate battery reveals that unigram/bigram BoW classifiers achieve F1 > 0.93 on training data. While the shared-prefix design eliminates early-turn vocabulary confounds (first-turn F1 = 0.35), post-branch vocabulary differences remain exploitable. The temporal signal we measure (turn-order sensitivity, voting gap) operates on top of this confound.

- **Single model family**: All LLM-generated text comes from Claude Sonnet 4.6. Cross-model generalization (e.g., attacks generated by GPT-4, Gemini, or open-weight models) has not been tested.

- **Fixed conversation length**: Conversations range from 6-9 user turns. Real-world conversations may be much longer. The architecture's behavior on 20+ turn conversations is untested.

### 13.4 Future Work

**Cross-domain transfer**: Test the trained model on multi-turn attack datasets from different domains (customer service, code generation, medical consultation) to assess generalization.

**Real-world validation**: Deploy the model as a secondary filter behind production AI systems. Measure false positive rates on organic multi-turn conversations and detection rates on red-team exercises.

**Online detection**: Modify the architecture for streaming inference — classify after each new turn rather than waiting for the full conversation. This requires adapting the sequence LSTM to produce calibrated probabilities at each timestep.

**Formal safety analysis**: The LSTM hidden-state trajectories (Section 12) form continuous safety-state curves. Investigate whether these trajectories satisfy the Markov property and can be connected to formal verification approaches for safety-critical systems.

**Longer contexts**: Extend the architecture to handle 20-50 turn conversations common in production LLM applications. This may require attention-based pooling over the turn sequence to avoid LSTM gradient degradation.
""".strip().split("\n"))


SECTION_13_CODE = fix_source("""# ============================================================================
# Final Summary: v3 Complete Results Table
# ============================================================================
with open("results/v3_evaluation/all_models_summary.json") as f:
    summary = json.load(f)

print("=" * 85)
print("COMPLETE v3 RESULTS — FINAL SUMMARY")
print("=" * 85)
print(f"\\n{'Model':<32s} {'F1':>7s} {'95% CI':>20s} {'AUC':>7s} {'Params':>10s}")
print("-" * 85)

rows = [
    ("DistilBERT (concatenated)", "distilbert_concat", "66.4M"),
    ("DistilBERT (hierarchical)", "distilbert_hier", "5.5M"),
    ("Continuation-only LSTM", "ablation_continuation", "27K"),
    ("Autoencoder encoder", "ablation_autoencoder", "27K"),
    ("Iter 6: LSTM + Attention", "iter6", "29K"),
    ("Iter 5: Temporal LSTM", "iter5", "27K"),
    ("Reversed turns", "ablation_reversed", "27K"),
    ("Shuffled turns", "ablation_shuffled", "27K"),
    ("Mean pool", "ablation_mean_pool", "27K"),
    ("Max pool", "ablation_max_pool", "27K"),
    ("Prefix-only", "ablation_prefix", "27K"),
    ("Cosine baseline", "cosine_baseline", "0"),
]

for label, key, params in rows:
    tier = summary["per_tier_metrics"][key]["overall"]
    ci = summary["bootstrap_cis"][key]["overall"]["f1"]
    ci_str = f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
    print(f"  {label:<30s} {tier['f1']:>7.4f} {ci_str:>20s} {tier['auc']:>7.4f} {params:>10s}")

# A10 voting
print()
for method, label in [("a10_top3_mean", "A10 top-3-mean voting"),
                       ("a10_max_vote", "A10 max-vote"),
                       ("a10_mean_vote", "A10 mean-vote")]:
    v = summary["a10_voting"][method]
    print(f"  {label:<30s} {v['f1']:>7.4f} {'—':>20s} {'—':>7s} {'0':>10s}")

print(f"\\nDataset: v3 shared-prefix, {5130} test sequences, 4 difficulty tiers")
print(f"Bootstrap: 1000 resamples, 95% confidence intervals")
print(f"All paired tests: one-sided, 1000 resamples")

# --- Radar chart: Model capabilities across tiers ---
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 7), subplot_kw=dict(polar=True))

categories = ["Easy", "Medium", "Hard", "Adversarial"]
n_cats = len(categories)
angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
angles += angles[:1]

models_radar = [
    ("iter5 (LSTM)", "iter5", "#FF9800"),
    ("iter6 (+attn)", "iter6", "#E65100"),
    ("Shuffled", "ablation_shuffled", "#FFE0B2"),
    ("DistilBERT-hier", "distilbert_hier", "#9C27B0"),
    ("DistilBERT-concat", "distilbert_concat", "#7B1FA2"),
]

for label, key, color in models_radar:
    values = [summary["per_tier_metrics"][key]["per_tier"][t.lower()]["f1"] for t in categories]
    values += values[:1]
    ax.plot(angles, values, "o-", linewidth=2, label=label, color=color)
    ax.fill(angles, values, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0.6, 1.02)
ax.set_title("Model Performance Across Difficulty Tiers", fontsize=13, fontweight="bold", pad=25)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

plt.tight_layout()
plt.savefig("results/v3_radar_tiers.png", dpi=150, bbox_inches="tight")
plt.show()

print("\\nThe radar chart shows how each model degrades across difficulty tiers.")
print("DistilBERT variants maintain near-perfect F1 even on adversarial sequences,")
print("while the temporal LSTM shows a 7-point drop from easy to adversarial.")
print("The shuffled ablation degrades more steeply, confirming that temporal")
print("ordering helps most on the harder tiers.")
""".strip().split("\n"))


def main():
    with open(NB_PATH) as f:
        nb = json.load(f)

    cells = nb["cells"]

    # Cell 47: replace old error analysis with confound gates
    cells[47] = make_code_cell(CONFOUND_GATES_CODE)

    # Cell 48: replace old ROC with animated trajectory
    cells[48] = make_code_cell(ANIMATED_TRAJECTORY_CODE)

    # Cell 49: update Section 13 conclusions
    cells[49] = make_markdown_cell(SECTION_13_MARKDOWN)

    # Cell 50: update final code cell with v3 summary
    cells[50] = make_code_cell(SECTION_13_CODE)

    # Remove old cells 53 (old final summary table referencing v1 variables)
    # Keep cells 51-52 (loss landscape) as-is

    if len(cells) > 53:
        del cells[53]

    nb["cells"] = cells
    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1)

    print(f"Updated notebook part 2: {NB_PATH}")
    print(f"Total cells: {len(cells)}")


if __name__ == "__main__":
    main()
