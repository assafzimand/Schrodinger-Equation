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
    valid_norms = center_norms[~torch.isnan(center_norms)]
    center_norm_mean = valid_norms.mean().item() if valid_norms.numel() > 0 else float("nan")
    center_norm_std = valid_norms.std().item() if valid_norms.numel() > 1 else float("nan")

    # 2. Pairwise distances between centers
    valid = ~torch.isnan(centers).any(dim=1)
    centers_valid = centers[valid]
    if centers_valid.shape[0] > 1:
        pdist = torch.cdist(centers_valid, centers_valid, p=2)
        upper = pdist[torch.triu_indices(pdist.shape[0], pdist.shape[1], offset=1)]
        center_dist_mean = upper.mean().item()
        center_dist_std = upper.std().item()
    else:
        upper = torch.tensor([], device=device)
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

    intra_dists = np.array(intra_dists)
    inter_dists = np.array(inter_dists)
    intra_mean = np.nanmean(intra_dists) if intra_dists.size else np.nan
    inter_mean = np.nanmean(inter_dists) if inter_dists.size else np.nan
    intra_std = np.nanstd(intra_dists) if intra_dists.size else np.nan
    inter_std = np.nanstd(inter_dists) if inter_dists.size else np.nan

    print(f"\n[DEBUG] Layer {layer_name}:")
    print(f"  Center norms:     mean={center_norm_mean:.4f}, std={center_norm_std:.4f}")
    print(f"  Center distances: mean={center_dist_mean:.4f}, std={center_dist_std:.4f}")
    print(f"  Intra-class dist: mean={intra_mean:.4f}, std={intra_std:.4f}")
    print(f"  Inter-class dist: mean={inter_mean:.4f}, std={inter_std:.4f}")

    # --- Debug plot generation ----------------------------------------------
    if save_debug_dir:
        import os
        os.makedirs(save_debug_dir, exist_ok=True)
        try:
            fig, ax = plt.subplots(2, 2, figsize=(10, 8))
            ax = ax.ravel()
            ax[0].hist(valid_norms.cpu().numpy(), bins=40, color="steelblue", alpha=0.7)
            ax[0].set_title(f"{layer_name} - Center Norms")

            if upper.numel() > 0:
                ax[1].hist(upper.cpu().numpy(), bins=40, color="darkorange", alpha=0.7)
                ax[1].set_title(f"{layer_name} - Center-to-Center Distances")
            else:
                ax[1].text(0.5, 0.5, "No valid pairs", ha="center")

            if intra_dists.size > 0:
                ax[2].hist(intra_dists, bins=40, color="seagreen", alpha=0.7)
            else:
                ax[2].text(0.5, 0.5, "Empty", ha="center")
            ax[2].set_title(f"{layer_name} - Intra-class Distances")

            if inter_dists.size > 0:
                ax[3].hist(inter_dists, bins=40, color="crimson", alpha=0.7)
            else:
                ax[3].text(0.5, 0.5, "Empty", ha="center")
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

# --- Linear separability & compactness helpers --------------------------------
import math
from typing import Dict, Tuple

def _stratified_split_idx(y: np.ndarray, val_frac: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Return train_idx, val_idx with simple stratified split."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = max(1, int(len(idx) * val_frac))
        val_idx.append(idx[:k])
        train_idx.append(idx[k:])
    return np.concatenate(train_idx), np.concatenate(val_idx)

def linear_probe_accuracy_torch(
    embeddings: np.ndarray,
    labels_true: np.ndarray,
    num_classes: int,
    device: str = "cuda",
    epochs: int = 20,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
) -> float:
    """
    Train a tiny linear classifier on frozen embeddings and return val accuracy.
    (Multinomial logistic regression; no external deps.)
    """
    x = torch.from_numpy(embeddings).float()
    y = torch.from_numpy(labels_true).long()
    train_idx, val_idx = _stratified_split_idx(labels_true, val_frac=0.2, seed=42)
    x_tr, y_tr = x[train_idx], y[train_idx]
    x_va, y_va = x[val_idx], y[val_idx]

    model = torch.nn.Linear(x.shape[1], num_classes)
    model.to(device)
    x_tr = x_tr.to(device); y_tr = y_tr.to(device)
    x_va = x_va.to(device); y_va = y_va.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(x_tr)
        loss = ce(logits, y_tr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(x_va).argmax(dim=1) == y_va).float().mean().item()
    return float(acc)

def fisher_ratio(
    embeddings: np.ndarray,
    labels_true: np.ndarray,
    num_classes: int,
) -> float:
    """
    Fisher ratio = trace(S_between) / trace(S_within)
    Higher is better (large between-class, small within-class).
    """
    x = torch.from_numpy(embeddings).float()
    y = torch.from_numpy(labels_true).long()
    C = num_classes

    # class means and counts
    means, Ns = [], []
    for c in range(C):
        mask = (y == c)
        if mask.any():
            xc = x[mask]
            means.append(xc.mean(dim=0, keepdim=True))
            Ns.append(xc.shape[0])
        else:
            means.append(torch.zeros((1, x.shape[1]), dtype=x.dtype))
            Ns.append(0)
    M = torch.cat(means, dim=0)  # (C, D)
    N = torch.tensor(Ns, dtype=torch.float32).unsqueeze(1)  # (C, 1)

    # Weighted global mean
    total = N.sum().clamp_min(1.0)
    mu = (M * N).sum(dim=0, keepdim=True) / total  # (1, D)

    # Within-class scatter (average squared distance to own mean)
    Sw = torch.tensor(0.0)
    for c in range(C):
        mask = (y == c)
        if mask.any():
            xc = x[mask] - M[c]
            Sw += (xc.pow(2).sum() / max(1, xc.shape[0]))

    # Between-class scatter (weighted distance of means to global mean)
    Sb = ((M - mu).pow(2).sum(dim=1) * N.squeeze(1)).sum() / total

    return float((Sb + 1e-12) / (Sw + 1e-12))


def center_margin_stats(
    embeddings: np.ndarray,
    labels_true: np.ndarray,
    centers: np.ndarray
) -> Dict[str, float]:
    """
    For each sample: margin = (nearest-other-center distance) - (own-center distance).
    Return mean margin and the fraction with positive margin.
    """
    x = torch.from_numpy(embeddings).float()
    y = torch.from_numpy(labels_true).long()
    C = centers.shape[0]
    cen = torch.from_numpy(centers).float()
    # distances to all centers
    # x: (N, D), cen: (C, D) -> dists: (N, C)
    dists = torch.cdist(x, cen, p=2)
    own = dists[torch.arange(x.shape[0]), y]
    # mask out own center and take min of others
    inf = torch.full_like(dists, float('inf'))
    d_others = torch.where(
        torch.nn.functional.one_hot(y, num_classes=C).bool(),
        inf, dists
    ).min(dim=1).values
    margin = d_others - own
    return {
        "mean_margin": float(margin.mean().item()),
        "pos_margin_frac": float((margin > 0).float().mean().item()),
        "own_dist_mean": float(own.mean().item()),
        "other_min_mean": float(d_others.mean().item()),
    }
