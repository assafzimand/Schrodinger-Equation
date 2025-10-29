"""Neural Collapse Center (NCC) metrics with GPU acceleration using PyTorch."""

import numpy as np
import torch
from typing import Dict, Tuple


def compute_class_centers_torch(embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute per-class mean embedding vectors using PyTorch (GPU-accelerated).
    
    Args:
        embeddings: (N, D) tensor of embeddings
        labels: (N,) tensor of class labels
        num_classes: Total number of classes
        
    Returns:
        (num_classes, D) tensor of class centers
    """
    device = embeddings.device
    dim = embeddings.shape[1]
    centers = torch.zeros((num_classes, dim), dtype=torch.float32, device=device)
    
    for k in range(num_classes):
        mask = labels == k
        if mask.any():
            centers[k] = embeddings[mask].mean(dim=0)
        else:
            centers[k] = float('nan')  # mark empty class
    
    return centers


def assign_to_nearest_center_torch(embeddings: torch.Tensor, centers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign each embedding to its nearest class center using PyTorch (GPU-accelerated).
    
    Args:
        embeddings: (N, D) tensor of embeddings
        centers: (num_classes, D) tensor of class centers
        
    Returns:
        assigned: (N,) tensor of assigned class indices
        dist_vals: (N,) tensor of distances to nearest center
    """
    # Filter out NaN centers (empty classes)
    valid = ~torch.isnan(centers).any(dim=1)
    centers_valid = centers[valid]
    
    # Compute pairwise distances: (N, num_valid_classes)
    # Broadcasting: embeddings (N, 1, D) - centers (1, num_valid_classes, D) -> (N, num_valid_classes, D)
    dists = torch.norm(embeddings.unsqueeze(1) - centers_valid.unsqueeze(0), dim=2)
    
    # Find nearest center for each embedding
    idx = dists.argmin(dim=1)
    
    # Map back to original class indices
    valid_indices = torch.arange(centers.shape[0], device=centers.device)[valid]
    assigned = valid_indices[idx]
    
    # Get distances to assigned centers
    dist_vals = dists[torch.arange(len(embeddings), device=embeddings.device), idx]
    
    return assigned, dist_vals


def ncc_mismatch_rate(embeddings: np.ndarray, labels_true: np.ndarray, num_classes: int, use_gpu: bool = True) -> Dict[str, np.ndarray]:
    """Compute NCC mismatch rate, assigned classes, and distances.
    
    Uses PyTorch for GPU acceleration when available.
    
    Args:
        embeddings: (N, D) numpy array of embeddings
        labels_true: (N,) numpy array of true class labels
        num_classes: Total number of classes
        use_gpu: Whether to use GPU if available
        
    Returns:
        Dictionary with:
            - mismatch_rate: float, fraction of mismatched assignments
            - assigned: (N,) numpy array of assigned class indices
            - distances: (N,) numpy array of distances to nearest center
            - centers: (num_classes, D) numpy array of class centers
    """
    # Move to GPU if available
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    
    # Convert to torch tensors
    emb_torch = torch.from_numpy(embeddings).float().to(device)
    labels_torch = torch.from_numpy(labels_true).long().to(device)
    
    # Compute centers
    centers = compute_class_centers_torch(emb_torch, labels_torch, num_classes)
    
    # Assign to nearest center
    assigned, dist_vals = assign_to_nearest_center_torch(emb_torch, centers)
    
    # Compute mismatch rate
    mismatch = (assigned != labels_torch).float().mean().item()
    
    # Convert back to numpy
    assigned_np = assigned.cpu().numpy()
    dist_vals_np = dist_vals.cpu().numpy()
    centers_np = centers.cpu().numpy()
    
    return {
        "mismatch_rate": mismatch,
        "assigned": assigned_np,
        "distances": dist_vals_np,
        "centers": centers_np,
    }
