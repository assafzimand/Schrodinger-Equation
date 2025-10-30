"""Neural Collapse Center (NCC) metrics with GPU acceleration using PyTorch."""

import numpy as np
import torch
from typing import Dict, Tuple
import matplotlib.pyplot as plt


def compute_class_centers_torch(embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute per-class mean embedding vectors using PyTorch (GPU-accelerated)."""
    device = embeddings.device
    dim = embeddings.shape[1]
    centers = torch.zeros((num_classes, dim), dtype=torch.float32, device=device)
    for k in range(num_classes):
        mask = labels == k
        if mask.any():
            centers[k] = embeddings[mask].mean(dim=0)
        else:
            centers[k] = float('nan')
    return centers


def assign_to_nearest_center_torch(embeddings: torch.Tensor, centers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign each embedding to its nearest class center using PyTorch (GPU-accelerated)."""
    valid = ~torch.isnan(centers).any(dim=1)
    centers_valid = centers[valid]
    dists = torch.norm(embeddings.unsqueeze(1) - centers_valid.unsqueeze(0), dim=2)
    idx = dists.argmin(dim=1)
    valid_indices = torch.arange(centers.shape[0], device=centers.device)[valid]
    assigned = valid_indices[idx]
    dist_vals = dists[torch.arange(len(embeddings), device=embeddings.device), idx]
    return assigned, dist_vals


def ncc_mismatch_rate(
    embeddings: np.ndarray,
    labels_true: np.ndarray,
    num_classes: int,
    use_gpu: bool = True,
    layer_name: str = "",
    save_debug_dir: str = None,
) -> Dict[str, np.ndarray]:
    """Compute NCC mismatch rate, assigned classes, and distances in (u,v) space,
    with added diagnostics on embedding geometry.
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    emb_torch = torch.from_numpy(embeddings).float().to(device)
    labels_torch = torch.from_numpy(labels_true).long().to(device)

    # --- Compute centers
    centers = compute_class_centers_torch(emb_torch, labels_torch, num_classes)

    # --- Debug statistics ----------------------------------------------------
    # 1. Norms of class centers
    center_norms = torch.norm(centers, dim=1)
    center_norm_mean = torch.nanmean(center_norms).item()
    center_norm_std = torch.nanstd(center_norms).item()

    # 2. Pairwise distances between centers
    valid = ~torch.isnan(centers).any(dim=1)
    centers_valid = centers[valid]
    if centers_valid.shape[0] > 1:
        pdist = torch.cdist(centers_valid, centers_valid, p=2)
        upper = pdist[torch.triu_indices(pdist.shape[0], pdist.shape[1], offset=1)]
        center_dist_mean = upper.mean().item()
        center_dist_std = upper.std().item()
    else:
        center_dist_mean = float("nan")
        center_dist_std = float("nan")

    # 3. Intra/inter-class spreads
    intra_dists, inter_dists = [], []
    for k in range(num_classes):
        mask = labels_torch == k
        if mask.sum() < 2 or torch.isnan(centers[k]).any():
            continue
        emb_k = emb_torch[mask]
        d_intra = torch.norm(emb_k - centers[k], dim=1)
        intra_dists.append(d_intra.mean().item())
        others = emb_torch[~mask]
        if others.shape[0] > 0:
            d_inter = torch.norm(others - centers[k], dim=1)
            inter_dists.append(d_inter.mean().item())
    intra_mean = np.nanmean(intra_dists) if len(intra_dists) else np.nan
    inter_mean = np.nanmean(inter_dists) if len(inter_dists) else np.nan
    intra_std = np.nanstd(intra_dists) if len(intra_dists) else np.nan
    inter_std = np.nanstd(inter_dists) if len(inter_dists) else np.nan

    print(f"\n[DEBUG] Layer {layer_name}:")
    print(f"  Center norms:     mean={center_norm_mean:.4f}, std={center_norm_std:.4f}")
    print(f"  Center distances: mean={center_dist_mean:.4f}, std={center_dist_std:.4f}")
    print(f"  Intra-class dist: mean={intra_mean:.4f}, std={intra_std:.4f}")
    print(f"  Inter-class dist: mean={inter_mean:.4f}, std={inter_std:.4f}")

    # Optionally save debug plots
    if save_debug_dir:
        try:
            import os
            os.makedirs(save_debug_dir, exist_ok=True)
            fig, ax = plt.subplots(2, 2, figsize=(10, 8))
            ax = ax.ravel()
            ax[0].hist(center_norms.cpu().numpy(), bins=40, color="steelblue", alpha=0.7)
            ax[0].set_title(f"{layer_name} - Center Norms")
            if centers_valid.shape[0] > 1:
                ax[1].hist(upper.cpu().numpy(), bins=40, color="darkorange", alpha=0.7)
                ax[1].set_title(f"{layer_name} - Center-to-Center Distances")
            ax[2].hist(intra_dists, bins=40, color="seagreen", alpha=0.7)
            ax[2].set_title(f"{layer_name} - Intra-class Distances")
            ax[3].hist(inter_dists, bins=40, color="crimson", alpha=0.7)
            ax[3].set_title(f"{layer_name} - Inter-class Distances")
            plt.tight_layout()
            fig.savefig(f"{save_debug_dir}/debug_stats_{layer_name}.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"  [WARN] Could not save debug plot for {layer_name}: {e}")

    # -------------------------------------------------------------------------
    # Continue standard NCC computation
    assigned, dist_vals = assign_to_nearest_center_torch(emb_torch, centers)
    mismatch = (assigned != labels_torch).float().mean().item()

    return {
        "mismatch_rate": mismatch,
        "assigned": assigned.cpu().numpy().copy(),
        "distances": dist_vals.cpu().numpy().copy(),
        "centers": centers.cpu().numpy().copy(),
        "debug": {
            "center_norm_mean": center_norm_mean,
            "center_norm_std": center_norm_std,
            "center_dist_mean": center_dist_mean,
            "center_dist_std": center_dist_std,
            "intra_mean": intra_mean,
            "intra_std": intra_std,
            "inter_mean": inter_mean,
            "inter_std": inter_std,
        },
    }
