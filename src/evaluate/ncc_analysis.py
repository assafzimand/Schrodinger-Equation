"""
NCC analysis for the Schrödinger PINN model.

Evaluates hidden-layer smoothness using the Nearest Class Center (NCC) metric.
"""

import numpy as np
import torch
import json
import mlflow
from typing import Dict, List, Optional
from pathlib import Path
from sklearn.metrics import confusion_matrix
from src.utils.ncc import ncc_mismatch_rate
from src.utils.ncc_plotting import (
    plot_ncc_results,
    plot_all_layer_confusions,
    plot_all_layer_dists,
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
    
    # Try to load checkpoint from MLflow artifacts first
    checkpoint_loaded = False
    artifact_path = None
    
    try:
        artifacts = client.list_artifacts(run_id, path="checkpoints")
        
        for artifact in artifacts:
            if "best_model" in artifact.path:
                artifact_path = artifact.path
                break
        
        if artifact_path is None and artifacts:
            # Fallback to any checkpoint in MLflow
            artifact_path = artifacts[0].path
        
        if artifact_path:
            # Download artifact to temp location
            import tempfile
            import os
            temp_dir = tempfile.mkdtemp()
            local_path = client.download_artifacts(run_id, artifact_path, temp_dir)
            
            # Load checkpoint
            checkpoint = torch.load(local_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Cleanup temp file
            os.remove(local_path)
            os.rmdir(temp_dir)
            
            checkpoint_loaded = True
            print(f"  ✓ Model loaded from MLflow artifact: {artifact_path}")
    
    except Exception as e:
        print(f"  Could not load from MLflow artifacts: {e}")
        print(f"  Trying local checkpoint directory...")
    
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


def make_class_labels_from_solver(data: Dict[str, np.ndarray], bins: int, amp_range: List[float]) -> np.ndarray:
    """Compute ground-truth amplitude and bin-based class labels."""
    u, v = data["u_f"], data["v_f"]
    amp = np.sqrt(u**2 + v**2)
    edges = np.linspace(amp_range[0], amp_range[1], bins + 1)
    labels = np.clip(np.digitize(amp, edges) - 1, 0, bins - 1)
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
    parser.add_argument("--bins", type=int, default=200, help="Number of bins for amplitude classes")
    parser.add_argument("--amp_range", type=float, nargs=2, default=[0, 4],
                        help="Amplitude range for binning")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for forward passes")
    
    args = parser.parse_args()
    
    # ---- CONFIG ----
    bins = args.bins
    amp_range = args.amp_range
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
    print(f"Bins: {bins}, Amplitude range: {amp_range}")
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
    labels_true = make_class_labels_from_solver(data, bins, amp_range)
    print(f"  Labels shape: {labels_true.shape}")
    print(f"  Unique classes: {len(np.unique(labels_true))}")

    # ---- FORWARD + HOOKS ----
    print("\nStep 4: Collecting layer activations...")
    activations = collect_layer_activations(model, x, t, layers_to_eval, batch_size, device)

    # ---- NCC METRICS ----
    print("\nStep 5: Computing NCC metrics...")
    mismatch_rates = []
    confusions, dists = [], []
    for ln in layers_to_eval:
        emb = activations[ln]
        result = ncc_mismatch_rate(emb, labels_true, bins)
        mismatch_rates.append(result["mismatch_rate"])
        confusions.append(confusion_matrix(labels_true, result["assigned"], labels=range(bins)))
        dists.append(result["distances"])
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


if __name__ == "__main__":
    main()
