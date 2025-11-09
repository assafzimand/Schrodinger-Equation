"""
NCC analysis for the Schrödinger PINN model.

Evaluates hidden-layer smoothness using the Nearest Class Center (NCC) metric.
"""
import sys, os, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import json
import mlflow
from matplotlib import pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path
from sklearn.metrics import confusion_matrix
from src.utils.ncc import ncc_mismatch_rate, linear_probe_accuracy_torch, fisher_ratio, center_margin_stats
from src.utils.ncc_plotting import (
    plot_ncc_results,
    plot_all_layer_confusions,
    plot_all_layer_dists,
    plot_layer_structure_evolution,
    plot_layer_geometry,
    plot_linear_separability_summary
)
from src.model.schrodinger_model import SchrodingerNet


def load_model_from_mlflow(run_id: str, device: torch.device, mlruns_path: str = "./mlruns"):
    """Load a trained SchrodingerNet model from MLflow using run_id.
    
    Args:
        run_id: MLflow run identifier
        device: Device to load model on
        mlruns_path: Path to MLflow tracking directory
        
    Returns:
        Loaded SchrodingerNet model
    """
    mlflow.set_tracking_uri(f"file:{mlruns_path}")
    
    # Get run information from MLflow
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params
    
    # Extract model architecture parameters
    hidden_layers = int(params.get('hidden_layers', 5))
    hidden_neurons = int(params.get('hidden_neurons', 100))
    activation = params.get('activation', 'tanh')
    
    print(f"Loading model from run {run_id}:")
    print(f"  hidden_layers={hidden_layers}, hidden_neurons={hidden_neurons}, activation={activation}")
    
    # Initialize model with correct architecture
    model = SchrodingerNet(
        hidden_layers=hidden_layers,
        hidden_neurons=hidden_neurons,
        activation=activation
    )
    
    # Try to load checkpoint from MLflow artifacts first (root and 'checkpoints/')
    checkpoint_loaded = False
    selected_artifact_path = None

    try:
        import re
        import tempfile
        import os

        candidate_paths = []

        # 1) Root artifacts
        try:
            root_artifacts = client.list_artifacts(run_id, path="")
            for art in root_artifacts:
                candidate_paths.append(art.path)
        except Exception:
            pass

        # 2) 'checkpoints/' subdir
        try:
            ckpt_artifacts = client.list_artifacts(run_id, path="checkpoints")
            for art in ckpt_artifacts:
                candidate_paths.append(art.path)
        except Exception:
            pass

        # Prefer best_model.pt, then final_model.pt
        norm_paths = [p.replace("\\", "/") for p in candidate_paths]
        for name in ["best_model.pt", "final_model.pt"]:
            for p in norm_paths:
                if p.endswith(name):
                    selected_artifact_path = p
                    break
            if selected_artifact_path is not None:
                break

        # If still none, pick the checkpoint_epoch with the largest epoch number
        if selected_artifact_path is None:
            max_epoch = -1
            best_ckpt = None
            pattern = re.compile(r"checkpoint_epoch(\d+)\.pt$")
            for p in norm_paths:
                m = pattern.search(p)
                if m:
                    epoch_num = int(m.group(1))
                    if epoch_num > max_epoch:
                        max_epoch = epoch_num
                        best_ckpt = p
            if best_ckpt is not None:
                selected_artifact_path = best_ckpt

        if selected_artifact_path is not None:
            temp_dir = tempfile.mkdtemp()
            local_path = client.download_artifacts(run_id, selected_artifact_path, temp_dir)

            # Load checkpoint
            checkpoint = torch.load(local_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Cleanup temp file(s)
            try:
                os.remove(local_path)
            except Exception:
                pass
            try:
                os.rmdir(temp_dir)
            except Exception:
                pass

            checkpoint_loaded = True
            exp_id = run.info.experiment_id
            full_artifact_path = f"mlruns/{exp_id}/{run_id}/artifacts/{selected_artifact_path}"
            print(f"  ✓ Model loaded from MLflow artifact: {full_artifact_path}")
        else:
            print("  No suitable checkpoint artifact found in MLflow; trying local directory...")
    except Exception as e:
        print(f"  Could not load from MLflow artifacts: {e}")
        print("  Trying local checkpoint directory...")
    
    # Fallback to local checkpoint directory
    if not checkpoint_loaded:
        from pathlib import Path
        checkpoint_dir = Path("outputs/checkpoints")
        
        # Try best_model.pt first
        best_model_path = checkpoint_dir / "best_model.pt"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            checkpoint_loaded = True
            print(f"  ✓ Model loaded from local checkpoint: {best_model_path}")
        else:
            # Try final_model.pt
            final_model_path = checkpoint_dir / "final_model.pt"
            if final_model_path.exists():
                checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                checkpoint_loaded = True
                print(f"  ✓ Model loaded from local checkpoint: {final_model_path}")
            else:
                # Try the latest checkpoint_epoch*.pt if present locally
                import re, glob
                pattern = str(checkpoint_dir / "checkpoint_epoch*.pt")
                ckpts = glob.glob(pattern)
                if ckpts:
                    # pick the one with the largest epoch
                    def epoch_num(p):
                        m = re.search(r"checkpoint_epoch(\d+)\.pt$", p.replace("\\", "/"))
                        return int(m.group(1)) if m else -1
                    latest = max(ckpts, key=epoch_num)
                    checkpoint = torch.load(latest, map_location=device, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    checkpoint_loaded = True
                    print(f"  ✓ Model loaded from local checkpoint: {latest}")
                else:
                    raise ValueError(
                        f"No checkpoint found for run {run_id} in MLflow or local directory"
                    )
    
    model.to(device)
    model.eval()
    
    return model


def load_dataset_npz(path: str) -> Dict[str, np.ndarray]:
    """Load dataset from .npz file.
    
    Expected keys: x_f, t_f, u_f, v_f (collocation points and values)
    """
    data = np.load(path)
    
    # Verify required keys exist
    required_keys = ['x_f', 't_f', 'u_f', 'v_f']
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Dataset missing required key: {key}")
    
    print(f"Loaded dataset from {path}:")
    print(f"  x_f shape: {data['x_f'].shape}")
    print(f"  t_f shape: {data['t_f'].shape}")
    print(f"  u_f shape: {data['u_f'].shape}")
    print(f"  v_f shape: {data['v_f'].shape}")
    
    return data


def make_class_labels_from_solver(data: Dict[str, np.ndarray], bins: int,
                                  u_range: List[float] = None,
                                  v_range: List[float] = None) -> np.ndarray:
    """Compute ground-truth 2D (u,v)-based class labels.

    The (u,v) domain is divided into a bins×bins grid.
    Each (u,v) sample is assigned to one of bins**2 classes.

    Args:
        data: Dict with ground-truth arrays containing keys 'u_f', 'v_f'
        bins: Number of bins per axis (total classes = bins**2)
        u_range: [min, max] range for u
        v_range: [min, max] range for v

    Returns:
        labels: (N,) numpy array of class indices in [0, bins**2 - 1]
    """
    if u_range is None:
        u_range = [np.percentile(data["u_f"], 1), np.percentile(data["u_f"], 99)]
    if v_range is None:
        v_range = [np.percentile(data["v_f"], 1), np.percentile(data["v_f"], 99)]

    u = np.clip(data["u_f"], u_range[0], u_range[1])
    v = np.clip(data["v_f"], v_range[0], v_range[1])

    # Uniform bin edges
    u_edges = np.linspace(u_range[0], u_range[1], bins + 1)
    v_edges = np.linspace(v_range[0], v_range[1], bins + 1)

    # Compute 2D bin indices
    u_bin = np.clip(np.digitize(u, u_edges) - 1, 0, bins - 1)
    v_bin = np.clip(np.digitize(v, v_edges) - 1, 0, bins - 1)

    # Flatten to single class index: class_id = u_bin * bins + v_bin
    labels = (u_bin * bins + v_bin).astype(np.int32)
    return labels



def collect_layer_activations(model, x, t, layers: List[str], batch_size: int, device: torch.device):
    """Forward pass with hook collection.
    
    Args:
        model: SchrodingerNet model
        x: Spatial coordinates (N,) or (N,1)
        t: Temporal coordinates (N,) or (N,1)
        layers: List of layer names to collect activations from
        batch_size: Batch size for forward passes
        device: Device to run on
        
    Returns:
        Dictionary mapping layer names to activation arrays (N, hidden_dim)
    """
    activations = {ln: [] for ln in layers}

    def get_hook(name):
        def hook(_, __, output):
            # Hook captures output after activation
            activations[name].append(output.detach().cpu())
        return hook

    # register hooks on specified layers
    handles = []
    for ln in layers:
        layer = getattr(model, ln)
        handles.append(layer.register_forward_hook(get_hook(ln)))

    model.eval()
    N = x.shape[0]
    
    print(f"Collecting activations from {len(layers)} layers...")
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = torch.from_numpy(x[i:i+batch_size]).float().to(device)
            tb = torch.from_numpy(t[i:i+batch_size]).float().to(device)
            
            # Ensure shape is (batch, 1)
            if xb.ndim == 1:
                xb = xb.unsqueeze(1)
            if tb.ndim == 1:
                tb = tb.unsqueeze(1)
            
            # SchrodingerNet.forward expects (x, t) separately
            _ = model(xb, tb)

    # Remove hooks
    for h in handles:
        h.remove()

    # Concatenate batches
    for ln in layers:
        activations[ln] = torch.cat(activations[ln], dim=0).numpy()
        print(f"  {ln}: {activations[ln].shape}")
    
    return activations


def main():
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="NCC analysis for Schrödinger PINN")
    parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID to analyze")
    parser.add_argument("--dataset", type=str, default="data/processed/dataset.npz",
                        help="Path to dataset .npz file")
    parser.add_argument("--bins", type=int, default=2, help="Number of bins for amplitude classes")
    parser.add_argument("--u_range", type=float, nargs=2, default=None,
                        help="u range for binning")
    parser.add_argument("--v_range", type=float, nargs=2, default=None,
                        help="v range for binning")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for forward passes")
    
    args = parser.parse_args()
    
    # ---- CONFIG ----
    bins = args.bins
    u_range = args.u_range
    v_range = args.v_range
    num_classes = bins ** 2
    layers_to_eval = ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5"]
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = args.run_id
    dataset_path = args.dataset
    output_dir = Path("outputs") / "plots" / f"ncc_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("NCC Analysis for Schrödinger PINN")
    print("=" * 80)
    print(f"Run ID: {run_id}")
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}")
    print(f"Bins: {bins}, U range: {u_range}, V range: {v_range}")
    print(f"Output directory: {output_dir}")
    print("=" * 80 + "\n")

    # ---- LOAD MODEL + DATA ----
    print("Step 1: Loading model from MLflow...")
    model = load_model_from_mlflow(run_id, device)
    model.to(device)
    
    print("\nStep 2: Loading dataset...")
    data = load_dataset_npz(dataset_path)
    x, t = data["x_f"], data["t_f"]
    
    print("\nStep 3: Creating class labels...")
    labels_true = make_class_labels_from_solver(data, bins, u_range, v_range)
    print(f"  Bins per axis: {bins} → total classes = {num_classes}")
    print(f"  Labels shape: {labels_true.shape}")
    print(f"  Unique classes: {len(np.unique(labels_true))}")

    # ---- FORWARD + HOOKS ----
    print("\nStep 4: Collecting layer activations...")
    activations = collect_layer_activations(model, x, t, layers_to_eval, batch_size, device)

    # ---- NCC METRICS ----
    print("\nStep 5: Computing NCC metrics...")
    mismatch_rates = []
    confusions, dists = [], []
    debug_stats = []  # collect layer diagnostics
    linear_probe_accs, fisher_ratios = [], []
    pos_margin_fracs, own_center_means, other_center_means = [], [], []
    for ln in layers_to_eval:
        emb = activations[ln]
        result = ncc_mismatch_rate(
            emb,
            labels_true,
            num_classes,
            layer_name=ln,
            save_debug_dir=output_dir / "debug_stats"
        )
        mismatch_rates.append(result["mismatch_rate"])
        confusions.append(confusion_matrix(labels_true, result["assigned"], labels=range(num_classes)))
        dists.append(result["distances"])
        debug_stats.append(result["debug"]) 
        # --- NEW: linear separability & margin metrics per layer ---
        probe_acc = linear_probe_accuracy_torch(
            emb, labels_true, num_classes, device=("cuda" if torch.cuda.is_available() else "cpu"),
            epochs=20, lr=1e-2, weight_decay=0.0
        )
        fisher = fisher_ratio(emb, labels_true, num_classes)
        mstats = center_margin_stats(emb, labels_true, result["centers"])

        linear_probe_accs.append(probe_acc)
        fisher_ratios.append(fisher)
        pos_margin_fracs.append(mstats["pos_margin_frac"])
        own_center_means.append(mstats["own_dist_mean"])
        other_center_means.append(mstats["other_min_mean"])

        print(f"  {ln}: mismatch_rate={result['mismatch_rate']:.4f}")

    # ---- SAVE METRICS + PLOTS ----
    print("\nStep 6: Saving results...")
    metrics_path = output_dir / "ncc_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(dict(zip(layers_to_eval, mismatch_rates)), f, indent=2)
    print(f"  ✓ Metrics saved to {metrics_path}")

    plot_ncc_results(layers_to_eval, mismatch_rates, output_dir / "ncc_layer_smoothness.png")
    print(f"  ✓ Smoothness plot saved to {output_dir / 'ncc_layer_smoothness.png'}")
    
    plot_all_layer_confusions(confusions, layers_to_eval, output_dir / "ncc_confusion_all_layers.png")
    print(f"  ✓ Confusion matrices saved to {output_dir / 'ncc_confusion_all_layers.png'}")
    
    plot_all_layer_dists(dists, layers_to_eval, output_dir / "ncc_distance_hist_all_layers.png")
    print(f"  ✓ Distance histograms saved to {output_dir / 'ncc_distance_hist_all_layers.png'}")

    print("\n" + "=" * 80)
    print(f"✓ NCC analysis complete! Results saved to {output_dir}")
    print("=" * 80)
    
    # Collect debug metrics across layers
    plot_layer_structure_evolution(debug_stats, layers_to_eval, mismatch_rates,
                               output_dir / "ncc_layer_structure_evolution.png")
    plot_layer_geometry(debug_stats, layers_to_eval, output_dir / "ncc_layer_geometry_evolution.png")
    plot_linear_separability_summary(
        layer_names=layers_to_eval,
        probe_acc=linear_probe_accs,
        fisher=fisher_ratios,
        pos_margin_frac=pos_margin_fracs,
        own_mean=own_center_means,
        other_mean=other_center_means,
        save_path=os.path.join(output_dir, "ncc_linear_separability.png"),
    )
    print(f"  ✓ Saved linear separability summary → {os.path.join(output_dir, 'ncc_linear_separability.png')}")




if __name__ == "__main__":
    main()

