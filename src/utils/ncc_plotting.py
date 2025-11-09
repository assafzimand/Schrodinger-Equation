import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_ncc_results(layer_names, mismatch_rates, save_path: str):
    plt.figure(figsize=(8, 5))
    plt.bar(layer_names, mismatch_rates, color="skyblue")
    plt.xlabel("Layer")
    plt.ylabel("Mismatch Rate")
    plt.title("NCC mismatch rate per layer (u,v classification)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    
def plot_layer_geometry(debug_stats, layers, save_path):
    """
    Plot how layer geometry evolves: center norms, center distances,
    intra-class and inter-class distances across layers.

    Args:
        debug_stats: list of dicts, one per layer, containing the means/stdevs.
        layers: list of layer names (same order as debug_stats).
        save_path: output image file path.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Extract metrics from debug_stats
    center_norm_means = [d["center_norm_mean"] for d in debug_stats]
    center_norm_stds  = [d["center_norm_std"] for d in debug_stats]
    center_dist_means = [d["center_dist_mean"] for d in debug_stats]
    center_dist_stds  = [d["center_dist_std"] for d in debug_stats]
    intra_means       = [d["intra_mean"] for d in debug_stats]
    intra_stds        = [d["intra_std"] for d in debug_stats]
    inter_means       = [d["inter_mean"] for d in debug_stats]
    inter_stds        = [d["inter_std"] for d in debug_stats]

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.title("Center and Class Distance Evolution")
    plt.plot(layers, center_norm_means, 'b-o', label='Center norm mean')
    plt.fill_between(layers,
                     np.array(center_norm_means) - np.array(center_norm_stds),
                     np.array(center_norm_means) + np.array(center_norm_stds),
                     color='b', alpha=0.2)
    plt.plot(layers, center_dist_means, 'r-o', label='Center-to-center distance mean')
    plt.fill_between(layers,
                     np.array(center_dist_means) - np.array(center_dist_stds),
                     np.array(center_dist_means) + np.array(center_dist_stds),
                     color='r', alpha=0.2)
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.title("Intra vs. Inter-Class Distances")
    plt.plot(layers, intra_means, 'g-o', label='Intra-class mean')
    plt.fill_between(layers,
                     np.array(intra_means) - np.array(intra_stds),
                     np.array(intra_means) + np.array(intra_stds),
                     color='g', alpha=0.2)
    plt.plot(layers, inter_means, 'm-o', label='Inter-class mean')
    plt.fill_between(layers,
                     np.array(inter_means) - np.array(inter_stds),
                     np.array(inter_means) + np.array(inter_stds),
                     color='m', alpha=0.2)
    plt.xlabel("Layer")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  ✓ Geometry evolution plot saved to {save_path}")



def plot_all_layer_confusions(confusions: list, layer_names: list, save_path: str):
    n = len(layer_names)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    vmax = max(cm.max() for cm in confusions)
    for ax, cm, name in zip(axes, confusions, layer_names):
        cm = cm.astype(np.float32)
        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)
        sns.heatmap(cm_norm, ax=ax, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{name} (normalized)")
        ax.set_xlabel("Predicted (u,v) class")
        ax.set_ylabel("True (u,v) class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_all_layer_dists(dists: list, layer_names: list, save_path: str):
    plt.figure(figsize=(8, 6))
    for d, name in zip(dists, layer_names):
        plt.hist(d, bins=50, alpha=0.5, density=True, label=name)
    plt.xlabel("Distance to nearest (u,v) class center")
    plt.ylabel("Density")
    plt.title("Cluster compactness per layer (u,v classification)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# --- NEW ---
def plot_layer_structure_evolution(debug_stats: list, layer_names: list, mismatch_rates: list, save_path: str):
    """Plot evolution of intra/inter distances, ratios, and center norms across layers."""
    if not debug_stats:
        print("⚠️ No debug stats available for layer structure evolution plot.")
        return

    layers = layer_names
    intra_means = np.array([d["intra_mean"] for d in debug_stats])
    inter_means = np.array([d["inter_mean"] for d in debug_stats])
    intra_stds = np.array([d["intra_std"] for d in debug_stats])
    inter_stds = np.array([d["inter_std"] for d in debug_stats])
    center_dists = np.array([d["center_dist_mean"] for d in debug_stats])
    center_dists_std = np.array([d["center_dist_std"] for d in debug_stats])
    center_norms = np.array([d["center_norm_mean"] for d in debug_stats])
    center_norms_std = np.array([d["center_norm_std"] for d in debug_stats])

    ratio = intra_means / inter_means

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 1. Intra/inter ratio
    axes[0, 0].plot(layers, ratio, marker='o', color='purple')
    axes[0, 0].set_title("Intra / Inter Class Distance Ratio (lower = better)")
    axes[0, 0].set_ylabel("Ratio")
    axes[0, 0].set_xlabel("Layer")

    # 2. Center norms (mean ± std)
    axes[0, 1].errorbar(layers, center_norms, yerr=center_norms_std, fmt='-o', color='teal')
    axes[0, 1].set_title("Center Norms Evolution (mean ± std)")
    axes[0, 1].set_xlabel("Layer")

    # 3. Center–center distances (mean ± std)
    axes[1, 0].errorbar(layers, center_dists, yerr=center_dists_std, fmt='-o', color='darkorange')
    axes[1, 0].set_title("Center–Center Distances (mean ± std)")
    axes[1, 0].set_xlabel("Layer")

    # 4. Intra vs inter distances
    axes[1, 1].plot(layers, intra_means, marker='o', label="Intra", color='blue')
    axes[1, 1].plot(layers, inter_means, marker='o', label="Inter", color='red')
    axes[1, 1].fill_between(layers, intra_means - intra_stds, intra_means + intra_stds, alpha=0.2, color='blue')
    axes[1, 1].fill_between(layers, inter_means - inter_stds, inter_means + inter_stds, alpha=0.2, color='red')
    axes[1, 1].legend()
    axes[1, 1].set_title("Intra vs Inter Distances Across Layers")
    axes[1, 1].set_xlabel("Layer")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved layer structure evolution plot to {save_path}")


def plot_layer_structure_stats(debug_stats: dict[str, dict[str, float]], save_path: str):
    """
    Plot layer-wise geometric statistics (center norms, true/false class distances).
    """
    layers = list(debug_stats.keys())

    # Extract metrics
    true_means = [debug_stats[l]['true_class_mean'] for l in layers]
    false_means = [debug_stats[l]['false_class_mean'] for l in layers]
    center_mean = [debug_stats[l]['center_norm_mean'] for l in layers]
    center_dist = [debug_stats[l]['center_dist_mean'] for l in layers]

    plt.figure(figsize=(8,5))
    plt.plot(layers, true_means, 'b-o', label="True-class mean dist")
    plt.plot(layers, false_means, 'r-o', label="False-class mean dist")
    plt.xlabel("Layer")
    plt.ylabel("Mean distance")
    plt.title("Class compactness vs. separation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "ncc_true_false_distance_evolution.png"))
    plt.close()


def plot_linear_separability_summary(
    layer_names: list[str],
    probe_acc: list[float],
    fisher: list[float],
    pos_margin_frac: list[float],
    own_mean: list[float],
    other_mean: list[float],
    save_path: str
):
    """
    3-panel summary:
      (A) Linear probe accuracy (↑ better)
      (B) Fisher ratio (↑ better)
      (C) Margin quality: fraction(margin>0) and the two mean distances
    """
    import matplotlib.pyplot as plt

    x = np.arange(len(layer_names))

    fig = plt.figure(figsize=(14, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(layer_names, probe_acc, marker='o')
    ax1.set_title("Linear probe accuracy per layer")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("Accuracy")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(layer_names, fisher, marker='o', color='tab:orange')
    ax2.set_title("Fisher ratio per layer (between / within)")
    ax2.set_ylabel("Fisher ratio")

    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot(layer_names, pos_margin_frac, marker='o', label="P(margin>0)")
    ax3.plot(layer_names, own_mean, marker='o', linestyle='--', label="mean own-center dist")
    ax3.plot(layer_names, other_mean, marker='o', linestyle='--', label="mean nearest-other dist")
    ax3.set_title("Center margin statistics")
    ax3.set_ylabel("Value")
    ax3.legend()
    ax3.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
