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


def plot_all_layer_confusions(confusions: list, layer_names: list, save_path: str):
    n = len(layer_names)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    vmax = max(cm.max() for cm in confusions)
    for ax, cm, name in zip(axes, confusions, layer_names):
        sns.heatmap(cm, ax=ax, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(name)
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
