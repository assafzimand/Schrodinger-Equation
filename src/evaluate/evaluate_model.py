"""Model evaluation script for comparing PINN predictions with solver.

Loads a trained model and compares its predictions against the ground truth
from the split-step Fourier solver across the full domain.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config_loader import Config
from src.model.schrodinger_model import SchrodingerNet
from src.solver.nlse_solver import solve_nlse_full_grid
from src.utils.plotting import (
    plot_comparison,
    plot_loss_curves,
    plot_phase_comparison,
)


def load_checkpoint(checkpoint_path: Path, device: str = "cuda") -> Tuple[SchrodingerNet, Config]:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint["config"]

    model = SchrodingerNet(
        hidden_layers=config.train.hidden_layers,
        hidden_neurons=config.train.hidden_neurons,
        activation=config.train.activation,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config


def predict_on_grid(
    model: SchrodingerNet,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    device: str = "cuda",
    batch_size: int = 1000,
) -> np.ndarray:
    """Predict model output on a full grid.

    Args:
        model: Trained model
        x_grid: Spatial grid (nx,)
        t_grid: Temporal grid (nt,)
        device: Device to use
        batch_size: Batch size for prediction

    Returns:
        Complex array of predictions (nt, nx)
    """
    model.eval()

    nx = len(x_grid)
    nt = len(t_grid)

    h_pred = np.zeros((nt, nx), dtype=complex)

    with torch.no_grad():
        for t_idx, t_val in enumerate(t_grid):
            # Create batch for this time point
            x_batch = torch.from_numpy(x_grid).float().unsqueeze(1).to(device)
            t_batch = torch.full((nx, 1), t_val, dtype=torch.float32).to(device)

            # Predict in batches
            for i in range(0, nx, batch_size):
                end_i = min(i + batch_size, nx)
                x_chunk = x_batch[i:end_i]
                t_chunk = t_batch[i:end_i]

                h_chunk = model.predict_h(x_chunk, t_chunk)
                h_pred[t_idx, i:end_i] = h_chunk.cpu().numpy()

    return h_pred


def compute_metrics(
    h_pred: np.ndarray,
    h_true: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
) -> Dict[str, float]:
    """Compute various error metrics.

    Args:
        h_pred: Model predictions (nt, nx)
        h_true: Ground truth (nt, nx)
        x_grid: Spatial grid
        t_grid: Temporal grid

    Returns:
        Dictionary of metrics
    """
    # Compute errors
    error = h_pred - h_true
    abs_error = np.abs(error)

    # L2 norms
    pred_norm = np.sqrt(np.sum(np.abs(h_pred) ** 2))
    true_norm = np.sqrt(np.sum(np.abs(h_true) ** 2))
    error_norm = np.sqrt(np.sum(abs_error ** 2))

    # Relative L2 error
    relative_l2 = error_norm / true_norm

    # Pointwise metrics
    max_abs_error = abs_error.max()
    mean_abs_error = abs_error.mean()
    std_abs_error = abs_error.std()

    # Relative pointwise error
    relative_pointwise = abs_error / (np.abs(h_true) + 1e-10)
    max_relative_error = relative_pointwise.max()
    mean_relative_error = relative_pointwise.mean()

    # Error vs time
    error_vs_time = abs_error.mean(axis=1)
    max_error_time_idx = error_vs_time.argmax()
    max_error_time = t_grid[max_error_time_idx]

    return {
        "relative_l2_error": float(relative_l2),
        "max_abs_error": float(max_abs_error),
        "mean_abs_error": float(mean_abs_error),
        "std_abs_error": float(std_abs_error),
        "max_relative_error": float(max_relative_error),
        "mean_relative_error": float(mean_relative_error),
        "max_error_time": float(max_error_time),
        "pred_norm": float(pred_norm),
        "true_norm": float(true_norm),
    }


def evaluate_model(
    checkpoint_path: Path,
    output_dir: Optional[Path] = None,
    device: str = "auto",
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate trained model against ground truth solver.

    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save plots (default: outputs/evaluation)
        device: Device to use
        verbose: Print progress

    Returns:
        Dictionary of evaluation metrics
    """
    if output_dir is None:
        output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print("=" * 70)
        print("Model Evaluation")
        print("=" * 70)
        print(f"\nCheckpoint: {checkpoint_path}")
        print(f"Device: {device}")

    # Load model
    if verbose:
        print("\n1. Loading model...")
    model, config = load_checkpoint(checkpoint_path, device)
    if verbose:
        print(f"   ✓ Model loaded ({model.count_parameters():,} parameters)")

    # Generate ground truth
    if verbose:
        print("\n2. Computing ground truth solution...")

    x_grid, t_grid, h_true = solve_nlse_full_grid(
        x_min=config.solver.x_min,
        x_max=config.solver.x_max,
        t_min=config.solver.t_min,
        t_max=config.solver.t_max,
        nx=config.solver.nx,
        nt=config.solver.nt,
        alpha=config.solver.alpha,
    )

    if verbose:
        print(f"   ✓ Grid: {len(x_grid)} × {len(t_grid)} points")

    # Predict on grid
    if verbose:
        print("\n3. Generating model predictions...")

    h_pred = predict_on_grid(model, x_grid, t_grid, device=device)

    if verbose:
        print(f"   ✓ Predictions generated")

    # Compute metrics
    if verbose:
        print("\n4. Computing metrics...")

    metrics = compute_metrics(h_pred, h_true, x_grid, t_grid)

    if verbose:
        print("\n   Metrics:")
        print(f"   Relative L² error:     {metrics['relative_l2_error']:.6f}")
        print(f"   Max absolute error:    {metrics['max_abs_error']:.6f}")
        print(f"   Mean absolute error:   {metrics['mean_abs_error']:.6f}")
        print(f"   Max relative error:    {metrics['max_relative_error']:.6f}")
        print(f"   Mean relative error:   {metrics['mean_relative_error']:.6f}")
        print(f"   Worst time point:      t={metrics['max_error_time']:.3f}")

    # Generate plots
    if verbose:
        print("\n5. Generating plots...")

    # Main comparison plot
    fig1 = plot_comparison(
        x_grid,
        t_grid,
        h_pred,
        h_true,
        save_path=output_dir / "model_vs_solver_comparison.png",
    )

    # Phase comparison at key times
    time_indices = [0, len(t_grid) // 2, -1]
    for idx in time_indices:
        fig2 = plot_phase_comparison(
            x_grid,
            t_grid,
            h_pred,
            h_true,
            time_idx=idx,
            save_path=output_dir / f"phase_comparison_t{idx:03d}.png",
        )
        plt.close(fig2)

    if verbose:
        print(f"   ✓ Plots saved to: {output_dir}")

    # Log to MLflow if active run
    try:
        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "eval/relative_l2_error": metrics["relative_l2_error"],
                    "eval/max_abs_error": metrics["max_abs_error"],
                    "eval/mean_abs_error": metrics["mean_abs_error"],
                }
            )
            mlflow.log_artifacts(str(output_dir))
    except Exception:
        pass  # No active run, skip MLflow logging

    if verbose:
        print("\n" + "=" * 70)
        print("✓ Evaluation completed!")
        print("=" * 70)

    plt.close("all")

    return metrics


def load_mlflow_losses(run_id: str) -> Dict[str, list]:
    """Load loss history from MLflow run.

    Args:
        run_id: MLflow run ID

    Returns:
        Dictionary with loss histories
    """
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    metrics = {}
    for key in [
        "train/total_loss",
        "train/mse_0",
        "train/mse_b",
        "train/mse_f",
        "eval/relative_l2_error",
    ]:
        history = client.get_metric_history(run_id, key)
        values = [m.value for m in history]
        short_key = key.split("/")[-1]
        metrics[short_key] = values

    return metrics


if __name__ == "__main__":
    """Evaluate the trained model."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained PINN model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/final_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (cuda/cpu/auto)",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Run training first: python src/train/engine.py")
        sys.exit(1)

    # Evaluate
    metrics = evaluate_model(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        device=args.device,
        verbose=True,
    )

    # Save metrics to file
    import json

    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {metrics_file}")

