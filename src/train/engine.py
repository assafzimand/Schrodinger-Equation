"""Training engine for physics-informed neural network.

Implements the training loop with:
- Adam optimizer
- MLflow experiment tracking
- Relative L² error metric
- Checkpoint saving
- Progress monitoring
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path (before project imports)
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

from src.config_loader import Config
from src.loss.physics_loss import SchrodingerLoss
from src.model.schrodinger_model import SchrodingerNet
from src.utils.metrics import phase_aligned_rel_l2_torch


def compute_relative_l2_error(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    u_true: torch.Tensor,
    v_true: torch.Tensor,
) -> float:
    """Compute phase-invariant relative L² error between prediction and ground truth.

    From PRD §5:
    Relative L² error = ||h_pred - h_true||_L2 / ||h_true||_L2
    
    Phase-invariant: optimal global phase is computed to best align h_pred with h_true,
    accounting for the gauge freedom in the Schrödinger equation.

    Args:
        model: Neural network model
        x: Spatial coordinates
        t: Temporal coordinates
        u_true: True real part
        v_true: True imaginary part

    Returns:
        Phase-invariant relative L² error as a scalar
    """
    with torch.no_grad():
        # Predict
        uv_pred = model(x, t)
        u_pred = uv_pred[:, 0]
        v_pred = uv_pred[:, 1]

        # Compute L² norms with phase alignment
        h_pred = torch.complex(u_pred, v_pred)
        h_true = torch.complex(u_true, v_true)

        relative_error = phase_aligned_rel_l2_torch(h_pred, h_true).item()

    return relative_error


def setup_mlflow(config: Config) -> str:
    """Set up MLflow experiment tracking.

    Args:
        config: Configuration object

    Returns:
        Run ID string
    """
    # Set tracking URI to local directory
    mlflow.set_tracking_uri("file:./mlruns")

    # Set or create experiment
    experiment_name = config.train.mlflow_experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        _experiment_id = mlflow.create_experiment(experiment_name)
    else:
        _experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    # Start run
    run = mlflow.start_run()

    # Log configuration
    mlflow.log_params(
        {
            "epochs": config.train.epochs,
            "learning_rate": config.train.learning_rate,
            "batch_size": config.train.batch_size,
            "hidden_layers": config.train.hidden_layers,
            "hidden_neurons": config.train.hidden_neurons,
            "activation": config.train.activation,
            "optimizer": config.train.optimizer,
            "device": config.train.device,
            "seed": config.train.seed,
            "n_collocation": config.dataset.n_collocation,
            "n_initial": config.dataset.n_initial,
            "n_boundary": config.dataset.n_boundary,
        }
    )

    return run.info.run_id


def split_dataset(
    dataset: Dict[str, np.ndarray],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Split dataset into train and validation sets.

    Args:
        dataset: Full dataset dictionary
        train_ratio: Ratio of data for training (default: 0.9)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    np.random.seed(seed)
    
    # Split collocation data
    n_collocation = len(dataset["x_f"])
    n_train = int(n_collocation * train_ratio)
    indices = np.random.permutation(n_collocation)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = {
        "x_f": dataset["x_f"][train_indices],
        "t_f": dataset["t_f"][train_indices],
        "u_f": dataset["u_f"][train_indices],
        "v_f": dataset["v_f"][train_indices],
        # Keep all IC and BC for both (they're small)
        "x_0": dataset["x_0"],
        "t_0": dataset["t_0"],
        "u_0": dataset["u_0"],
        "v_0": dataset["v_0"],
        "x_b": dataset["x_b"],
        "t_b": dataset["t_b"],
    }

    val_dataset = {
        "x_f": dataset["x_f"][val_indices],
        "t_f": dataset["t_f"][val_indices],
        "u_f": dataset["u_f"][val_indices],
        "v_f": dataset["v_f"][val_indices],
        "x_0": dataset["x_0"],
        "t_0": dataset["t_0"],
        "u_0": dataset["u_0"],
        "v_0": dataset["v_0"],
        "x_b": dataset["x_b"],
        "t_b": dataset["t_b"],
    }

    return train_dataset, val_dataset


def create_dataloaders(
    dataset: Dict[str, np.ndarray],
    batch_size: int,
    device: str,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for collocation, initial, and boundary points.

    Args:
        dataset: Dictionary with dataset arrays
        batch_size: Batch size for training
        device: Device to place tensors on

    Returns:
        Tuple of (collocation_loader, initial_loader, boundary_loader)
    """
    # Collocation data (ensure float32 for GPU efficiency)
    x_f = torch.from_numpy(dataset["x_f"]).float().unsqueeze(1).to(device)
    t_f = torch.from_numpy(dataset["t_f"]).float().unsqueeze(1).to(device)
    u_f = torch.from_numpy(dataset["u_f"]).float().to(device)
    v_f = torch.from_numpy(dataset["v_f"]).float().to(device)

    collocation_dataset = TensorDataset(x_f, t_f, u_f, v_f)
    collocation_loader = DataLoader(
        collocation_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Initial condition data (ensure float32 for GPU efficiency)
    x_0 = torch.from_numpy(dataset["x_0"]).float().unsqueeze(1).to(device)
    t_0 = torch.from_numpy(dataset["t_0"]).float().unsqueeze(1).to(device)
    u_0 = torch.from_numpy(dataset["u_0"]).float().to(device)
    v_0 = torch.from_numpy(dataset["v_0"]).float().to(device)

    initial_dataset = TensorDataset(x_0, t_0, u_0, v_0)
    initial_loader = DataLoader(
        initial_dataset,
        batch_size=len(initial_dataset),  # Use all initial points
        shuffle=False,
    )

    # Boundary condition data (ensure float32 for GPU efficiency)
    x_b = torch.from_numpy(dataset["x_b"]).float().unsqueeze(1).to(device)
    t_b = torch.from_numpy(dataset["t_b"]).float().unsqueeze(1).to(device)

    boundary_dataset = TensorDataset(x_b, t_b)
    boundary_loader = DataLoader(
        boundary_dataset,
        batch_size=len(boundary_dataset),  # Use all boundary points
        shuffle=False,
    )

    return collocation_loader, initial_loader, boundary_loader


def train_epoch(
    model: nn.Module,
    loss_fn: SchrodingerLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    collocation_loader: DataLoader,
    initial_loader: DataLoader,
    boundary_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch with AMP support.

    Args:
        model: Neural network model
        loss_fn: Physics-informed loss function
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        collocation_loader: DataLoader for collocation points
        initial_loader: DataLoader for initial condition
        boundary_loader: DataLoader for boundary conditions
        device: Device to use

    Returns:
        Dictionary with average losses for the epoch
    """
    model.train()

    total_loss_sum = 0.0
    mse_0_sum = 0.0
    mse_b_sum = 0.0
    mse_f_sum = 0.0
    n_batches = 0

    # Timing accumulators (per-batch averages)
    time_forward_sum = 0.0
    time_backward_sum = 0.0
    time_step_sum = 0.0
    # Loss component timings
    time_ic_sum = 0.0
    time_bc_sum = 0.0
    time_pde_sum = 0.0
    time_pde_predict_sum = 0.0
    time_pde_derivatives_sum = 0.0
    time_pde_residual_sum = 0.0

    # Get initial and boundary data (used for all batches)
    x_0, t_0, u_0, v_0 = next(iter(initial_loader))
    x_b, t_b = next(iter(boundary_loader))
    
    # Enable AMP on CUDA only
    use_amp = (device.type == "cuda")

    # Iterate over collocation batches
    for x_f, t_f, u_f, v_f in collocation_loader:
        optimizer.zero_grad(set_to_none=True)

        # Forward pass with autocast
        t_fwd0 = time.time()
        with torch.cuda.amp.autocast(enabled=use_amp):
            # Compute loss with all components
            total_loss, mse_0, mse_b, mse_f = loss_fn(
                model,
                x_0,
                t_0,
                u_0,
                v_0,
                x_b,
                t_b,
                x_f,
                t_f,
                return_components=True,
            )
        t_fwd1 = time.time()

        # Backpropagation with gradient scaling
        t_bwd0 = t_fwd1
        scaler.scale(total_loss).backward()
        t_bwd1 = time.time()
        scaler.step(optimizer)
        t_step1 = time.time()
        scaler.update()

        # Accumulate losses
        total_loss_sum += total_loss.item()
        mse_0_sum += mse_0.item()
        mse_b_sum += mse_b.item()
        mse_f_sum += mse_f.item()
        n_batches += 1

        # Accumulate timings
        time_forward_sum += (t_fwd1 - t_fwd0)
        time_backward_sum += (t_bwd1 - t_bwd0)
        time_step_sum += (t_step1 - t_bwd1)

        # Loss internals (may not exist if exception)
        lt = getattr(loss_fn, "last_timings", {}) or {}
        time_ic_sum += float(lt.get("time/ic", 0.0))
        time_bc_sum += float(lt.get("time/bc", 0.0))
        time_pde_sum += float(lt.get("time/pde", 0.0))
        time_pde_predict_sum += float(lt.get("time/pde/predict", 0.0))
        time_pde_derivatives_sum += float(lt.get("time/pde/derivatives", 0.0))
        time_pde_residual_sum += float(lt.get("time/pde/residual", 0.0))

    # Average losses
    return {
        "total_loss": total_loss_sum / n_batches,
        "mse_0": mse_0_sum / n_batches,
        "mse_b": mse_b_sum / n_batches,
        "mse_f": mse_f_sum / n_batches,
        # Timings (per-batch averages)
        "time/forward": time_forward_sum / n_batches,
        "time/backward": time_backward_sum / n_batches,
        "time/step": time_step_sum / n_batches,
        "time/ic": time_ic_sum / n_batches,
        "time/bc": time_bc_sum / n_batches,
        "time/pde": time_pde_sum / n_batches,
        "time/pde/predict": time_pde_predict_sum / n_batches,
        "time/pde/derivatives": time_pde_derivatives_sum / n_batches,
        "time/pde/residual": time_pde_residual_sum / n_batches,
    }


def evaluate(
    model: nn.Module,
    dataset: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on full dataset (no gradient computation).

    Args:
        model: Neural network model
        dataset: Full dataset
        device: Device to use

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    t0 = time.time()
    with torch.no_grad():
        # Evaluate on collocation points (ensure float32)
        x_f = torch.from_numpy(dataset["x_f"]).float().unsqueeze(1).to(device)
        t_f = torch.from_numpy(dataset["t_f"]).float().unsqueeze(1).to(device)
        u_f = torch.from_numpy(dataset["u_f"]).float().to(device)
        v_f = torch.from_numpy(dataset["v_f"]).float().to(device)
        t1 = time.time()

        rel_l2_error = compute_relative_l2_error(model, x_f, t_f, u_f, v_f)
        t2 = time.time()

    return {
        "relative_l2_error": rel_l2_error,
        "eval_time_prepare": t1 - t0,
        "eval_time_predict_metric": t2 - t1,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    config: Config,
    filename: Optional[str] = None,
) -> Path:
    """Save model checkpoint.

    Args:
        model: Neural network model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        config: Configuration
        filename: Optional filename (default: checkpoint_epochN.pt)

    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(config.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_epoch{epoch:04d}.pt"

    checkpoint_path = checkpoint_dir / filename

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "config": config,
        },
        checkpoint_path,
    )

    return checkpoint_path


def train(
    config: Config,
    dataset: Dict[str, np.ndarray],
    eval_dataset: Optional[Dict[str, np.ndarray]] = None,
    verbose: bool = True,
) -> nn.Module:
    """Main training function.

    Args:
        config: Configuration object
        dataset: Training dataset
        eval_dataset: Optional evaluation dataset (separate from training)
        verbose: Whether to print progress

    Returns:
        Trained model
    """
    if verbose:
        print("=" * 70)
        print("Training Schrödinger PINN")
        print("=" * 70)

    # Set random seed
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)

    # Determine device
    if config.train.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.train.device)
    
    # Enable cudnn benchmark for better performance
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if verbose:
        print(f"\n[Engine] Using device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create model
    model = SchrodingerNet(
        hidden_layers=config.train.hidden_layers,
        hidden_neurons=config.train.hidden_neurons,
        activation=config.train.activation,
    ).to(device)

    if verbose:
        print(f"\nModel: {model.count_parameters():,} parameters")

    # Create loss function
    loss_fn = SchrodingerLoss()

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
    )

    # Create dataloaders
    collocation_loader, initial_loader, boundary_loader = create_dataloaders(
        dataset, config.train.batch_size, str(device)
    )

    if verbose:
        print(
            f"\nDataset: {len(dataset['x_f'])} collocation, "
            f"{len(dataset['x_0'])} initial, "
            f"{len(dataset['x_b'])} boundary"
        )
        print(f"Batch size: {config.train.batch_size}")
        print(f"Batches per epoch: {len(collocation_loader)}")

    # Setup MLflow
    run_id = setup_mlflow(config)
    if verbose:
        print(f"\nMLflow run ID: {run_id}")

    # Initialize GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Training loop
    if verbose:
        print(f"\nTraining for {config.train.epochs} epochs...")
        print("-" * 70)

    start_time = time.time()

    # Evaluation cadence: every 20% of epochs + final
    eval_interval = max(1, int(config.train.epochs * 0.2))
    
    # Use separate eval dataset if provided, otherwise use training dataset
    dataset_for_eval = eval_dataset if eval_dataset is not None else dataset
    
    for epoch in range(1, config.train.epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model,
            loss_fn,
            optimizer,
            scaler,
            collocation_loader,
            initial_loader,
            boundary_loader,
            device,
        )

        # Determine if we should evaluate this epoch
        should_evaluate = (epoch % eval_interval == 0) or (epoch == config.train.epochs)
        
        if should_evaluate:
            # Evaluate on eval dataset
            eval_metrics = evaluate(model, dataset_for_eval, device)

            # Log to MLflow (both train and eval metrics)
            mlflow.log_metrics(
                {
                    "train/total_loss": train_metrics["total_loss"],
                    "train/mse_0": train_metrics["mse_0"],
                    "train/mse_b": train_metrics["mse_b"],
                    "train/mse_f": train_metrics["mse_f"],
                    # Timings (train)
                    "time_forward": train_metrics.get("time/forward", 0.0),
                    "time_backward": train_metrics.get("time/backward", 0.0),
                    "time_step": train_metrics.get("time/step", 0.0),
                    "time_ic": train_metrics.get("time/ic", 0.0),
                    "time_bc": train_metrics.get("time/bc", 0.0),
                    "time_pde": train_metrics.get("time/pde", 0.0),
                    "time_pde_predict": train_metrics.get("time/pde/predict", 0.0),
                    "time_pde_derivatives": train_metrics.get("time/pde/derivatives", 0.0),
                    "time_pde_residual": train_metrics.get("time/pde/residual", 0.0),
                    "eval/relative_l2_error": eval_metrics["relative_l2_error"],
                    "eval_time_prepare": eval_metrics.get("eval_time_prepare", 0.0),
                    "eval_time_predict_metric": eval_metrics.get("eval_time_predict_metric", 0.0),
                },
                step=epoch,
            )
        else:
            # Log only training metrics
            mlflow.log_metrics(
                {
                    "train/total_loss": train_metrics["total_loss"],
                    "train/mse_0": train_metrics["mse_0"],
                    "train/mse_b": train_metrics["mse_b"],
                    "train/mse_f": train_metrics["mse_f"],
                    # Timings (train)
                    "time_forward": train_metrics.get("time/forward", 0.0),
                    "time_backward": train_metrics.get("time/backward", 0.0),
                    "time_step": train_metrics.get("time/step", 0.0),
                    "time_ic": train_metrics.get("time/ic", 0.0),
                    "time_bc": train_metrics.get("time/bc", 0.0),
                    "time_pde": train_metrics.get("time/pde", 0.0),
                    "time_pde_predict": train_metrics.get("time/pde/predict", 0.0),
                    "time_pde_derivatives": train_metrics.get("time/pde/derivatives", 0.0),
                    "time_pde_residual": train_metrics.get("time/pde/residual", 0.0),
                },
                step=epoch,
            )

        epoch_time = time.time() - epoch_start

        # Update epoch time metric (log separately to avoid overwriting the dict above)
        mlflow.log_metric("time_epoch", epoch_time, step=epoch)

        # Print progress (at eval epochs only)
        if verbose and should_evaluate:
            l2_str = f"L²: {eval_metrics['relative_l2_error']:.6f}" if should_evaluate else "L²: (skipped)"
            print(
                f"Epoch {epoch:4d}/{config.train.epochs} | "
                f"Loss: {train_metrics['total_loss']:.6f} | "
                f"MSE₀: {train_metrics['mse_0']:.6f} | "
                f"MSE_b: {train_metrics['mse_b']:.6f} | "
                f"MSE_f: {train_metrics['mse_f']:.6f} | "
                f"{l2_str} | "
                f"t_fwd={train_metrics.get('time/forward',0.0):.2f}s, "
                f"t_bwd={train_metrics.get('time/backward',0.0):.2f}s, "
                f"t_step={train_metrics.get('time/step',0.0):.2f}s | "
                f"epoch={epoch_time:.1f}s"
            )

        # Save checkpoint periodically
        if epoch % max(1, config.train.epochs // 5) == 0:
            checkpoint_path = save_checkpoint(
                model,
                optimizer,
                epoch,
                train_metrics["total_loss"],
                config,
            )
            mlflow.log_artifact(str(checkpoint_path))

    total_time = time.time() - start_time

    # Save final model
    final_checkpoint = save_checkpoint(
        model,
        optimizer,
        config.train.epochs,
        train_metrics["total_loss"],
        config,
        filename="final_model.pt",
    )
    mlflow.log_artifact(str(final_checkpoint))

    if verbose:
        print("-" * 70)
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Final loss: {train_metrics['total_loss']:.6f}")
        print(f"Final L² error: {eval_metrics['relative_l2_error']:.6f}")
        print(f"Model saved: {final_checkpoint}")

    # End MLflow run
    mlflow.end_run()

    return model


if __name__ == "__main__":
    """Sanity test with 2 epochs on data subset."""
    from src.config_loader import load_config
    from src.data.generate_dataset import load_dataset

    print("Running 2-epoch sanity test...")

    # Load config
    config = load_config("config/train.yaml")

    # Override for quick test
    config.train.epochs = 2
    config.train.batch_size = 512

    # Load dataset
    dataset = load_dataset(config.dataset.save_path)

    # Use subset for sanity test
    subset_size = 1000
    dataset_subset = {
        "x_f": dataset["x_f"][:subset_size],
        "t_f": dataset["t_f"][:subset_size],
        "u_f": dataset["u_f"][:subset_size],
        "v_f": dataset["v_f"][:subset_size],
        "x_0": dataset["x_0"],
        "t_0": dataset["t_0"],
        "u_0": dataset["u_0"],
        "v_0": dataset["v_0"],
        "x_b": dataset["x_b"],
        "t_b": dataset["t_b"],
    }

    print(f"Using subset: {subset_size} collocation points")

    # Train
    model = train(config, dataset_subset, verbose=True)

    print("\n✓ Sanity test completed!")

