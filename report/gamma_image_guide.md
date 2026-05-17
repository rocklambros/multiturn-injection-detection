# Image Upload Guide for Gamma.app Presentation

All images are located in `results/` relative to the project root.

## Slide-by-Slide Image Mapping

### Slide 1: Title Slide
- No project image needed. Use Gamma's built-in theme or a generic cybersecurity background.

### Slide 2: Attackers Exploit Conversations, Not Just Messages
- No project image strictly required. The attack example table carries the slide.
- **Optional:** If Gamma supports diagrams, create a simple 4-step flow with a green-to-red gradient showing turn-by-turn escalation.

### Slide 3: Chollet's Heuristic Tells You When Deep Learning Will (and Won't) Help
- **Upload:** `results/v3_data_overview.png`
- **What it shows:** Three-panel figure with dataset composition by difficulty tier, attack strategy distribution, and conversation length histogram.
- **Placement:** Right side or bottom half. Provides visual context for the dataset numbers in the text.
- **Teaching value:** Gives classmates a concrete picture of the data they would need to evaluate the Chollet ratio themselves.

### Slide 4: Freezing a Trained Encoder Creates a Reusable Feature Extractor
- **Upload:** `results/embedding_space_manifold.png`
- **What it shows:** t-SNE projections at three stages through the GRU encoder (raw embeddings, post-GRU, post-projection), showing progressive separation of benign and injection samples.
- **Placement:** Below or beside the architecture diagram. This is the visual payoff for the "freeze and reuse" concept.
- **Teaching value:** Shows classmates what "learning a representation" looks like concretely, connecting the abstract transfer-learning idea to a visible transformation.
- **Alternative:** `results/rnn_convergence_comparison.png` (training curves showing GRU convergence) if you want to emphasize training dynamics instead.

### Slide 5: Results -- A 27K-Parameter LSTM Outperforms All Voting Baselines
- **Upload:** `results/v3_model_hierarchy.png`
- **What it shows:** Horizontal bar chart of all model F1 scores with 95% bootstrap confidence intervals. The temporal LSTM sits clearly above voting baselines and below DistilBERT.
- **Placement:** Full width or right half. This chart tells the entire story of the project at a glance. The table in the slide text provides exact numbers; the chart provides the visual comparison.
- **Why this image matters:** This is the single most important image in the presentation. It is the core result.

### Slide 6: Why Recurrence Matters -- The Controlled Comparison
- **Upload:** `results/v3_iter5_vs_iter6.png`
- **What it shows:** Direct comparison of iter5 (temporal LSTM) vs iter6 (LSTM + attention), along with voting baselines, showing that attention adds negligible value while recurrence adds 13 F1 points over voting.
- **Placement:** Right side, complementing the textual explanation of the controlled comparison.
- **Teaching value:** Reinforces the idea that a single variable change (adding recurrence) produces the performance gap.
- **Alternative:** `results/v3_hidden_trajectories.png` shows LSTM hidden state trajectories diverging for attack vs. benign, providing a mechanistic view of what recurrence does.

### Slide 7: How to Prove Your Model Learned Temporal Patterns (Not Just Vocabulary)
- **Upload:** `results/v3_ablation_summary.png`
- **What it shows:** Grouped bar chart comparing F1 scores for ordered, shuffled, reversed, mean-pool, max-pool, and prefix-only conditions.
- **Placement:** Right side, paired with the ablation table on the left.
- **Teaching value:** Demonstrates ablation methodology as a general technique classmates can use in their own projects: destroy one property of the input, measure the degradation, and attribute the gap to that property.
- **Secondary option:** `results/v3_hidden_trajectories.png` shows LSTM hidden state trajectories diverging for attack vs. benign, providing a mechanistic complement.

### Slide 8: Gate Dynamics Reveal What the LSTM Attends To Across Turns
- **Upload:** `results/gate_activations_heatmap.png`
- **What it shows:** Mean forget/input/output gate activations across conversation turns for a sample attack and benign sequence. The forget gate drops sharply at the divergence point in attack sequences.
- **Placement:** Top or right side. This is the visual centerpiece of the "teaching LSTM gates" narrative.
- **Teaching value:** Connects the abstract gate equations from class to observable behavior on a real task. Classmates can see the forget gate resetting when the attack begins.
- **Secondary option:** `results/v3_strategy_heatmap.png` (per-strategy F1 heatmap) can go below or as a second image if Gamma supports two images per slide. It shows why some attack strategies produce cleaner temporal signatures than others.

### Slide 9: Bigger Models Win on Accuracy, Smaller Models Win on Deployment
- **Upload:** `results/v3_param_efficiency.png`
- **What it shows:** Log-scale scatter plot with parameter count on x-axis and F1 on y-axis. The Pareto frontier runs from the temporal LSTM (lower-left) to DistilBERT concat (upper-right).
- **Placement:** Center of the slide. One image captures the entire efficiency argument.
- **Teaching value:** Illustrates that "best model" depends on deployment constraints, not just accuracy. Forces classmates to think about where their models will run.

### Slide 10: Three Lessons That Generalize Beyond This Project
- **Upload:** `results/v3_strategy_heatmap.png`
- **What it shows:** Heatmap of per-strategy F1 across all model variants. The consistent left-to-right difficulty gradient (fragment distribution easiest, instruction layering hardest) holds regardless of model architecture.
- **Placement:** Right side or bottom half.
- **Teaching value:** Demonstrates that difficulty is a property of the data, not the model. Classmates learn to separate data properties from model properties in their analyses.
- **Alternative:** `results/v3_confound_gates.png` (confound gate battery pass/fail) if you want to emphasize the limitations discussion instead.

### Slide 11: Thank You / Q&A
- **Upload:** `results/v3_radar_tiers.png`
- **What it shows:** Radar chart comparing model performance across the 4 difficulty tiers. DistilBERT forms a near-perfect circle; the temporal LSTM shows visible degradation on the adversarial tier.
- **Placement:** Background or side decoration. Visually appealing closing image.

## Image Priority Ranking

If Gamma limits the number of images or some slides work better text-only, prioritize in this order:

1. **v3_model_hierarchy.png** (Slide 5) - The core result with all models and CIs. Non-negotiable.
2. **v3_ablation_summary.png** (Slide 7) - Teaches ablation methodology with a concrete example.
3. **gate_activations_heatmap.png** (Slide 8) - Connects LSTM gate equations to observable behavior.
4. **v3_iter5_vs_iter6.png** (Slide 6) - The controlled comparison isolating recurrence.
5. **embedding_space_manifold.png** (Slide 4) - Makes "learned representations" tangible.
6. **v3_param_efficiency.png** (Slide 9) - The deployment tradeoff visual.
7. **v3_data_overview.png** (Slide 3) - Dataset context.
8. **v3_strategy_heatmap.png** (Slide 10) - Data difficulty vs. model difficulty distinction.
9. **v3_radar_tiers.png** (Slide 11) - Visual closing.

## Additional High-Impact Images Not Assigned to Slides

These images are available for swapping in or adding to speaker notes:

| Image | Description | Best used for |
|-------|-------------|---------------|
| `cross_iteration_comparison.png` | F1 progression across all 7 iterations | Showing the development arc from baseline to final model |
| `diagnostic_curves_grid.png` | ROC and PR curves for all iterations | Detailed classifier diagnostics during Q&A |
| `error_analysis_grid.png` | Confidence histograms for FP/FN across models | Understanding failure modes during Q&A |
| `loss_landscape_3d.png` | 3D loss surface of the attention model | Visual appeal, optimization surface discussion |
| `v3_confound_gates.png` | Confound gate battery pass/fail results | Foregrounding limitations and honesty about BoW confounds |
| `v3_hidden_trajectories.png` | LSTM hidden state trajectories over turns | Mechanistic evidence of temporal processing |
| `v3_iter5_per_tier.png` | Per-tier F1 breakdown for iter5 | Detailed tier analysis during Q&A |
| `embedding_space_animation.gif` | Animated t-SNE through GRU layers | Live demo only (not compatible with static slides) |
