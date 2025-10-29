import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_ncc_results(layer_names, mismatch_rates, save_path: str):
    plt.figure(figsize=(8, 5))
    plt.bar(layer_names, mismatch_rates, color="skyblue")
    plt.xlabel("Layer")
    plt.ylabel("Mismatch Rate")
    plt.title("NCC mismatch rate per layer")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_all_layer_confusions(confusions: list, layer_names: list, save_path: str):
    """Show all confusion matrices (1x5 grid)."""
    n = len(layer_names)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    vmax = max(cm.max() for cm in confusions)
    for ax, cm, name in zip(axes, confusions, layer_names):
        sns.heatmap(cm, ax=ax, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_all_layer_dists(dists: list, layer_names: list, save_path: str):
    """Overlay distance histograms for all layers on one figure."""
    plt.figure(figsize=(8, 6))
    for d, name in zip(dists, layer_names):
        plt.hist(d, bins=50, alpha=0.5, density=True, label=name)
    plt.xlabel("Distance to nearest center")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Cluster compactness per layer")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
