# src/utils/rfwfc_utils.py
"""
Utilities to interface our Schrödinger model with the RFWFC repo (DL_Layer_Analysis).

This module:
- Loads a trained model by MLflow run_id (reuses your ncc_analysis pattern)
- Wraps our (x,t)->h dataset as a PyTorch Dataset compatible with RFWFC's code
- Provides a simple arg Namespace that mirrors the notebook's 'loaded_example_args'
- Collects layer names and forward hooks if needed by RFWFC
"""

import os
import json
import types
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# We rely on your project structure:
from src.model.schrodinger_model import SchrodingerNet  # adjust if class name differs
# Reuse your MLflow loader pattern from ncc_analysis.py:
# (We inline a minimal version to avoid import cycles.)
import mlflow
from mlflow.tracking import MlflowClient


# ----------------------------
# Dataset wrapper
# ----------------------------

class XTtoH_Dataset(Dataset):
    """
    Wraps (x, t) inputs and complex target h=u+iv as (u,v) float tensor.

    __getitem__ returns:
       inputs: tensor of shape [2] => (x, t)
       target: tensor of shape [2] => (u, v)

    This satisfies many generic loaders. If RFWFC expects different,
    tweak __getitem__ accordingly (TODO note below).
    """
    def __init__(self, x: np.ndarray, t: np.ndarray, u: np.ndarray, v: np.ndarray):
        assert x.shape == t.shape == u.shape == v.shape
        self.x = torch.from_numpy(x.astype(np.float32)).view(-1, 1)
        self.t = torch.from_numpy(t.astype(np.float32)).view(-1, 1)
        self.inputs = torch.cat([self.x, self.t], dim=1)  # [N,2]
        self.targets = torch.from_numpy(
            np.stack([u, v], axis=1).astype(np.float32)
        )  # [N,2]

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ----------------------------
# Simple args “namespace” mirroring the notebook
# ----------------------------

@dataclass
class RFWFCRunArgs:
    # core knobs mirrored from the example notebook
    trees: int = 100
    depth: int = 10  # max depth for RFWFC Random Forests
    feature_dimension: int = 100  # embedding dimension (hidden layer width)
    low_range_epsilon: float = 0.4
    high_range_epsilon: float = 0.1

    # IO / bookkeeping
    checkpoints_folder: str = ""   # where RFWFC expects to write JSONs
    output_folder: str = ""        # usually same as checkpoints_folder
    create_umap: bool = False
    use_clustering: bool = True
    
    batch_size: int = 4096
    seed: int = 42
    calc_test: bool = True
    
    # Optional device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    env_name: str = "schrodinger_1d"
    checkpoint_file_name: str = "best_model.pt"

    # Layer list (we also pass separately)
    layers: Optional[List[str]] = None


def build_args(checkpoints_folder: str, trees: int = 100,
               low_eps: float = 0.4, high_eps: float = 0.1,
               use_clustering: bool = True, create_umap: bool = False) -> RFWFCRunArgs:
    os.makedirs(checkpoints_folder, exist_ok=True)
    return RFWFCRunArgs(
        trees=trees,
        low_range_epsilon=low_eps,
        high_range_epsilon=high_eps,
        checkpoints_folder=checkpoints_folder,
        output_folder=checkpoints_folder,
        create_umap=create_umap,
        use_clustering=use_clustering,
    )


# ----------------------------
# MLflow loading (same style as your NCC)
# ----------------------------

class WrappedSchrodingerNet(nn.Module):
    """
    Wraps the SchrödingerNet to make it compatible with RFWFC,
    which expects a single-tensor input of shape (N, 2).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        # Expose inner layers for RFWFC introspection
        for name in ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5", "output"]:
            if hasattr(model, name):
                setattr(self, name, getattr(model, name))

    def forward(self, X):
        # Split the 2D input into (x, t) tensors
        if X.ndim == 2 and X.shape[1] == 2:
            x = X[:, 0:1]
            t = X[:, 1:2]
            return self.model(x, t)
        else:
            raise ValueError(f"Expected input shape (N, 2), got {X.shape}")


def load_model_from_mlflow(run_id: str, device: Optional[torch.device] = None) -> nn.Module:
    """Load trained model using the same logic as ncc_analysis.py."""
    import re, tempfile, os
    import mlflow
    from mlflow.tracking import MlflowClient

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params

    hidden_layers = int(params.get('hidden_layers', 5))
    hidden_neurons = int(params.get('hidden_neurons', 100))
    activation = params.get('activation', 'tanh')

    print(f"Loading model from run {run_id}: layers={hidden_layers}, neurons={hidden_neurons}, activation={activation}")
    model = SchrodingerNet(
        hidden_layers=hidden_layers,
        hidden_neurons=hidden_neurons,
        activation=activation
    )

    # Reuse the checkpoint loading logic from ncc_analysis
    checkpoint_loaded = False
    selected_artifact_path = None
    try:
        candidate_paths = []
        try:
            for art in client.list_artifacts(run_id, path=""):
                candidate_paths.append(art.path)
        except Exception:
            pass
        try:
            for art in client.list_artifacts(run_id, path="checkpoints"):
                candidate_paths.append(art.path)
        except Exception:
            pass
        norm_paths = [p.replace("\\", "/") for p in candidate_paths]
        for name in ["best_model.pt", "final_model.pt"]:
            for p in norm_paths:
                if p.endswith(name):
                    selected_artifact_path = p
                    break
            if selected_artifact_path is not None:
                break
        if selected_artifact_path is None:
            import re
            pattern = re.compile(r"checkpoint_epoch(\d+)\.pt$")
            best_ckpt, max_epoch = None, -1
            for p in norm_paths:
                m = pattern.search(p)
                if m:
                    e = int(m.group(1))
                    if e > max_epoch:
                        best_ckpt, max_epoch = p, e
            selected_artifact_path = best_ckpt

        if selected_artifact_path:
            tmpdir = tempfile.mkdtemp()
            local_path = client.download_artifacts(run_id, selected_artifact_path, tmpdir)
            checkpoint = torch.load(local_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            checkpoint_loaded = True
            print(f"  ✓ Model loaded from MLflow artifact: {selected_artifact_path}")
    except Exception as e:
        print("  Could not load from MLflow:", e)

    if not checkpoint_loaded:
        raise RuntimeError(f"Failed to load model weights for run {run_id}")

    model.to(device)
    model.eval()
    return model


# ----------------------------
# Dataset helpers
# ----------------------------

def load_npz_dataset(npz_path: str, num_samples: Optional[int] = None) -> Tuple[np.ndarray, ...]:
    """
    Loads your processed dataset.npz.

    Expected keys (adjust if different):
       - 'x_all', 't_all', 'u_all', 'v_all'
    or:
       - collocation: x_f, t_f, u_f, v_f  (if you saved GT everywhere)
    """
    data = np.load(npz_path)
    x = data["x_f"]
    t = data["t_f"]
    u = data["u_f"]
    v = data["v_f"]
    N = x.shape[0]
    if num_samples is not None and num_samples < N:
        idx = np.random.RandomState(123).choice(N, size=num_samples, replace=False)
        x, t, u, v = x[idx], t[idx], u[idx], v[idx]
    return x, t, u, v


def make_dataloaders(x, t, u, v, batch_size=4096) -> Tuple[DataLoader, Dataset, Dataset]:
    ds = XTtoH_Dataset(x, t, u, v)
    # simple 80/20 split for analysis
    N = len(ds)
    n_train = int(0.8 * N)
    idx = np.random.RandomState(42).permutation(N)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    train_ds = torch.utils.data.Subset(ds, train_idx.tolist())
    test_ds = torch.utils.data.Subset(ds, test_idx.tolist())
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return loader, train_ds, test_ds


# ----------------------------
# Layer list & hooks (if needed by RFWFC)
# ----------------------------

def get_named_hidden_layers(model: nn.Module) -> List[str]:
    """
    Returns the names of hidden layers as strings.
    Works for both plain SchrodingerNet and WrappedSchrodingerNet(model).
    """
    print("🔍 Inspecting model layers in get_named_hidden_layers...")
    found_layers = []

    for n, _ in model.named_modules():
        if any(f"layer_{i}" in n for i in range(1, 6)):
            print("  found:", n)
            # Normalize nested names (e.g., "model.layer_1" → "layer_1")
            clean = n.split(".")[-1]
            if clean not in found_layers:
                found_layers.append(clean)

    # enforce canonical order
    found_layers = sorted(found_layers, key=lambda s: int(s.split("_")[1]))
    print("✅ Normalized hidden layer names:", found_layers)
    return found_layers



def infer_feature_dimension_from_model(model: nn.Module, fallback: int = 100) -> int:
    # Try to read the first hidden layer width from your named layers
    for name, module in model.named_modules():
        if name == "layer_1" and isinstance(module, nn.Linear):
            return module.out_features
    # Generic fallback
    for module in model.modules():
        if isinstance(module, nn.Linear):
            return module.out_features
    return fallback
