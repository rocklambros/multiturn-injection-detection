"""Insert 3 advanced visualization cells into the notebook.

Inserts from bottom to top to avoid index shifting issues:
  1. Loss landscape (3D surface) — before final analysis
  2. Gate activations (heatmap) — after multi-turn LSTM, before attention
  3. Embedding space (t-SNE manifold) — after single-turn models, before multi-turn
"""
import json

with open('notebooks/multiturn_injection_detection.ipynb') as f:
    nb = json.load(f)

cells = nb['cells']

# ────────────────────────────────────────────────────────────────
# Cell definitions
# ────────────────────────────────────────────────────────────────

embed_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Representation Space: How the GRU Transforms Input\n",
        "\n",
        "Each layer of the GRU encoder progressively separates benign and injection samples in representation space. The visualization below shows t-SNE projections of 2,000 test samples at three stages through the network, analogous to the manifold transformations described in Olah (2014). In the raw embedding space, the two classes are intermingled. By the time representations pass through the bidirectional GRU and dense projection, injection samples cluster away from benign ones. This \"manifold untangling\" is what makes the encoder effective as a frozen feature extractor for the multi-turn classifier.\n"
    ]
}

embed_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# Embedding Space Manifold Visualization (colah-style)\n",
        "# ============================================================================\n",
        "import numpy as np\n",
        "from pathlib import Path\n",
        "from matplotlib.animation import FuncAnimation, PillowWriter\n",
        "from IPython.display import Image as IPImage, display\n",
        "\n",
        "tsne_path = Path('results/embedding_tsne.npz')\n",
        "if not tsne_path.exists():\n",
        "    print('Cache not found. Run: python scripts/generate_embedding_tsne.py')\n",
        "else:\n",
        "    data = np.load(tsne_path)\n",
        "    tsne_stages = [data['tsne_embedding'], data['tsne_gru'], data['tsne_dense']]\n",
        "    labels = data['labels']\n",
        "    stage_titles = [\n",
        "        'After Embedding Layer\\n(128-dim \\u2192 t-SNE)',\n",
        "        'After Bidirectional GRU\\n(128-dim \\u2192 t-SNE)',\n",
        "        'After Dense Projection\\n(32-dim \\u2192 t-SNE)'\n",
        "    ]\n",
        "\n",
        "    # --- Static 3-panel figure ---\n",
        "    fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
        "    benign_mask = labels == 0\n",
        "    for ax, tsne, title in zip(axes, tsne_stages, stage_titles):\n",
        "        ax.scatter(tsne[benign_mask, 0], tsne[benign_mask, 1],\n",
        "                   c='#2196F3', s=8, alpha=0.35, label='Benign', rasterized=True)\n",
        "        ax.scatter(tsne[~benign_mask, 0], tsne[~benign_mask, 1],\n",
        "                   c='#f44336', s=8, alpha=0.35, label='Injection', rasterized=True)\n",
        "        ax.set_title(title, fontsize=13, fontweight='bold')\n",
        "        ax.set_xticks([]); ax.set_yticks([])\n",
        "        for spine in ax.spines.values(): spine.set_visible(False)\n",
        "        ax.legend(fontsize=9, markerscale=2, framealpha=0.8)\n",
        "    fig.suptitle('GRU Encoder: Progressive Manifold Separation (2,000 test samples)',\n",
        "                 fontsize=15, fontweight='bold', y=1.02)\n",
        "    plt.tight_layout()\n",
        "    plt.savefig('results/embedding_space_manifold.png', dpi=150, bbox_inches='tight')\n",
        "    plt.show()\n",
        "\n",
        "    # --- Animated GIF: smooth morphing between stages ---\n",
        "    stages_norm = []\n",
        "    for s in tsne_stages:\n",
        "        s_c = s - s.mean(axis=0)\n",
        "        stages_norm.append(s_c / (s_c.std() + 1e-8))\n",
        "\n",
        "    fig_a, ax_a = plt.subplots(figsize=(7, 7))\n",
        "    scat_b = ax_a.scatter([], [], c='#2196F3', s=10, alpha=0.45, label='Benign')\n",
        "    scat_i = ax_a.scatter([], [], c='#f44336', s=10, alpha=0.45, label='Injection')\n",
        "    ax_a.set_xlim(-4, 4); ax_a.set_ylim(-4, 4)\n",
        "    ax_a.set_xticks([]); ax_a.set_yticks([])\n",
        "    for spine in ax_a.spines.values(): spine.set_visible(False)\n",
        "    ax_a.legend(fontsize=11, markerscale=2, loc='upper right')\n",
        "    ttl = ax_a.set_title('', fontsize=14, fontweight='bold')\n",
        "\n",
        "    stage_names = ['Embedding Layer', 'GRU Hidden State', 'Dense Encoding']\n",
        "    pause, trans = 15, 30\n",
        "    frame_list = []\n",
        "    for s in range(3):\n",
        "        frame_list += [('hold', s, 0)] * pause\n",
        "        if s < 2:\n",
        "            frame_list += [('move', s, f / trans) for f in range(trans)]\n",
        "\n",
        "    def update(info):\n",
        "        kind, stg, t = info\n",
        "        if kind == 'hold':\n",
        "            coords = stages_norm[stg]\n",
        "            ttl.set_text(f'Stage {stg+1}: {stage_names[stg]}')\n",
        "        else:\n",
        "            coords = (1 - t) * stages_norm[stg] + t * stages_norm[stg + 1]\n",
        "            ttl.set_text(f'{stage_names[stg]} \\u2192 {stage_names[stg+1]}')\n",
        "        scat_b.set_offsets(coords[benign_mask])\n",
        "        scat_i.set_offsets(coords[~benign_mask])\n",
        "        return scat_b, scat_i, ttl\n",
        "\n",
        "    anim = FuncAnimation(fig_a, update, frames=frame_list, blit=True)\n",
        "    anim.save('results/embedding_space_animation.gif', writer=PillowWriter(fps=20))\n",
        "    plt.close(fig_a)\n",
        "    print('Saved results/embedding_space_animation.gif')\n",
        "    display(IPImage(filename='results/embedding_space_animation.gif'))\n"
    ]
}

gate_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### LSTM Gate Dynamics Across Conversation Turns\n",
        "\n",
        "The sequence LSTM processes one turn-encoding per timestep. At each step, three gates control information flow:\n",
        "- **Forget gate**: How much prior context to retain (high = remember everything, low = reset).\n",
        "- **Input gate**: How much new turn information to incorporate.\n",
        "- **Output gate**: What fraction of the cell state to expose as the hidden state.\n",
        "\n",
        "The heatmaps below show mean gate activations (averaged over the 64 hidden units) for a sample attack conversation and a benign conversation. In the attack, notice how the input gate spikes on turns that escalate toward the exploit, while the forget gate stays high to maintain the accumulated attack context.\n"
    ]
}

gate_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# LSTM Gate Activation Visualization\n",
        "# ============================================================================\n",
        "gate_path = Path('results/gate_activations.npz')\n",
        "if not gate_path.exists():\n",
        "    print('Cache not found. Run: python scripts/generate_gate_activations.py')\n",
        "else:\n",
        "    gdata = np.load(gate_path, allow_pickle=True)\n",
        "    with open('results/gate_activation_texts.json') as f:\n",
        "        turn_texts = json.load(f)\n",
        "\n",
        "    forget_mean = gdata['forget_gates'].mean(axis=2)\n",
        "    input_mean = gdata['input_gates'].mean(axis=2)\n",
        "    output_mean = gdata['output_gates'].mean(axis=2)\n",
        "    g_labels = gdata['labels']\n",
        "    g_nturns = gdata['num_turns']\n",
        "\n",
        "    # Pick first attack and first benign\n",
        "    atk_idx = int(np.where(g_labels == 1)[0][0])\n",
        "    ben_idx = int(np.where(g_labels == 0)[0][0])\n",
        "\n",
        "    fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
        "    for ax, idx, label in zip(axes, [atk_idx, ben_idx], ['Attack', 'Benign']):\n",
        "        nt = g_nturns[idx]\n",
        "        gate_matrix = np.stack([\n",
        "            forget_mean[idx, :nt],\n",
        "            input_mean[idx, :nt],\n",
        "            output_mean[idx, :nt]\n",
        "        ])  # (3, num_turns)\n",
        "\n",
        "        im = ax.imshow(gate_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)\n",
        "        ax.set_yticks([0, 1, 2])\n",
        "        ax.set_yticklabels(['Forget', 'Input', 'Output'], fontsize=11)\n",
        "        ax.set_xticks(range(nt))\n",
        "        ax.set_xticklabels([f'T{i+1}' for i in range(nt)], fontsize=10)\n",
        "        ax.set_xlabel('Conversation Turn', fontsize=11)\n",
        "        ax.set_title(f'{label} Conversation ({nt} turns)', fontsize=13, fontweight='bold')\n",
        "\n",
        "        for i in range(3):\n",
        "            for j in range(nt):\n",
        "                val = gate_matrix[i, j]\n",
        "                color = 'white' if val > 0.6 else 'black'\n",
        "                ax.text(j, i, f'{val:.2f}', ha='center', va='center',\n",
        "                        fontsize=9, color=color, fontweight='bold')\n",
        "\n",
        "    fig.colorbar(im, ax=axes, shrink=0.6, label='Mean Gate Activation')\n",
        "    fig.suptitle('Sequence LSTM Gate Dynamics: Attack vs. Benign',\n",
        "                 fontsize=15, fontweight='bold', y=1.02)\n",
        "    plt.tight_layout()\n",
        "    plt.savefig('results/gate_activations_heatmap.png', dpi=150, bbox_inches='tight')\n",
        "    plt.show()\n",
        "\n",
        "    # Print turn texts for context\n",
        "    print(f'\\nAttack conversation (sample {atk_idx}):')\n",
        "    for i, t in enumerate(turn_texts[atk_idx]):\n",
        "        print(f'  Turn {i+1}: {t[:90]}...' if len(t) > 90 else f'  Turn {i+1}: {t}')\n"
    ]
}

loss_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Loss Landscape: Optimization Surface Geometry\n",
        "\n",
        "The 3D surface below visualizes the loss landscape of the multi-turn attention model around its trained parameters. Following the technique from Li et al. (2018), we sample two random directions in the ~29K-dimensional trainable parameter space, apply filter-wise normalization, and evaluate the BCE loss at each point on a 25x25 grid. The frozen turn encoder outputs are pre-computed, so only the sequence LSTM, attention, and classification head are perturbed.\n",
        "\n",
        "A smooth, bowl-shaped landscape indicates the optimizer found a well-conditioned minimum. Sharp or irregular surfaces would suggest sensitivity to parameter perturbations, which could indicate overfitting or optimization instability.\n"
    ]
}

loss_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# 3D Loss Landscape Visualization (Li et al. 2018)\n",
        "# ============================================================================\n",
        "from mpl_toolkits.mplot3d import Axes3D  # noqa: F401\n",
        "\n",
        "land_path = Path('results/loss_landscape.npz')\n",
        "if not land_path.exists():\n",
        "    print('Cache not found. Run: python scripts/generate_loss_landscape.py')\n",
        "else:\n",
        "    ldata = np.load(land_path)\n",
        "    alphas, betas = ldata['alphas'], ldata['betas']\n",
        "    loss_grid = ldata['loss_grid']\n",
        "    A, B = np.meshgrid(alphas, betas, indexing='ij')\n",
        "\n",
        "    # 3D surface plot\n",
        "    fig = plt.figure(figsize=(12, 8))\n",
        "    ax = fig.add_subplot(111, projection='3d')\n",
        "    surf = ax.plot_surface(A, B, loss_grid, cmap='viridis', alpha=0.85,\n",
        "                           linewidth=0.1, edgecolor='none', antialiased=True)\n",
        "    ax.set_xlabel('Direction 1 (\\u03b1)', fontsize=11, labelpad=8)\n",
        "    ax.set_ylabel('Direction 2 (\\u03b2)', fontsize=11, labelpad=8)\n",
        "    ax.set_zlabel('BCE Loss', fontsize=11, labelpad=8)\n",
        "    ax.set_title('Loss Landscape: Multi-Turn Attention Model (~29K trainable params)',\n",
        "                 fontsize=13, fontweight='bold', pad=15)\n",
        "\n",
        "    min_idx = np.unravel_index(loss_grid.argmin(), loss_grid.shape)\n",
        "    ax.scatter([alphas[min_idx[0]]], [betas[min_idx[1]]], [loss_grid[min_idx]],\n",
        "              c='red', s=150, zorder=5, marker='*', label='Trained minimum')\n",
        "    ax.legend(fontsize=11, loc='upper right')\n",
        "    ax.view_init(elev=30, azim=225)\n",
        "\n",
        "    fig.colorbar(surf, shrink=0.45, aspect=12, label='BCE Loss', pad=0.1)\n",
        "    plt.savefig('results/loss_landscape_3d.png', dpi=150, bbox_inches='tight')\n",
        "    plt.show()\n",
        "\n",
        "    # Contour plot (top-down view)\n",
        "    fig2, ax2 = plt.subplots(figsize=(8, 6))\n",
        "    cs = ax2.contourf(A, B, loss_grid, levels=30, cmap='viridis')\n",
        "    ax2.contour(A, B, loss_grid, levels=15, colors='white', linewidths=0.5, alpha=0.4)\n",
        "    ax2.scatter([alphas[min_idx[0]]], [betas[min_idx[1]]], c='red', s=200,\n",
        "               marker='*', zorder=5, edgecolors='white', linewidths=1,\n",
        "               label='Trained minimum')\n",
        "    ax2.set_xlabel('Direction 1 (\\u03b1)', fontsize=12)\n",
        "    ax2.set_ylabel('Direction 2 (\\u03b2)', fontsize=12)\n",
        "    ax2.set_title('Loss Landscape Contours', fontsize=14, fontweight='bold')\n",
        "    ax2.legend(fontsize=11)\n",
        "    fig2.colorbar(cs, label='BCE Loss')\n",
        "    plt.tight_layout()\n",
        "    plt.savefig('results/loss_landscape_contour.png', dpi=150, bbox_inches='tight')\n",
        "    plt.show()\n",
        "\n",
        "    print(f'Loss at trained minimum: {loss_grid[min_idx]:.6f}')\n",
        "    print(f'Loss range: [{loss_grid.min():.4f}, {loss_grid.max():.4f}]')\n"
    ]
}

# ────────────────────────────────────────────────────────────────
# Insert cells (bottom to top to avoid index shifting)
# ────────────────────────────────────────────────────────────────

# 1. Loss landscape: before final analysis (current cell 45)
cells.insert(45, loss_code)
cells.insert(45, loss_md)

# 2. Gate activations: after multi-turn LSTM code (current cell 30), before attention (cell 31)
cells.insert(31, gate_code)
cells.insert(31, gate_md)

# 3. Embedding space: after overfitting analysis (cell 28), before multi-turn intro (cell 29)
cells.insert(29, embed_code)
cells.insert(29, embed_md)

nb['cells'] = cells

with open('notebooks/multiturn_injection_detection.ipynb', 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print(f"Notebook updated: {len(cells)} cells (was 46, added 6)")
