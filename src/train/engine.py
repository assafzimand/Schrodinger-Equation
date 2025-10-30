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
    subset_size: int = 0,
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
        subset_size: Number of random points to use (0 = use all)

    Returns:
        Phase-invariant relative L² error as a scalar
    """
    with torch.no_grad():
        # Optionally subsample for faster evaluation
        if subset_size > 0 and len(x) > subset_size:
            idx = torch.randperm(len(x), device=x.device)[:subset_size]
            x = x[idx]
            t = t[idx]
            u_true = u_true[idx]
            v_true = v_true[idx]
        
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
    params_to_log = {
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
    
    # Add scheduler parameters if configured
    if config.train.scheduler.type is not None:
        params_to_log["scheduler_type"] = config.train.scheduler.type
        params_to_log["scheduler_factor"] = config.train.scheduler.factor
        params_to_log["scheduler_patience"] = config.train.scheduler.patience
        params_to_log["scheduler_cooldown"] = config.train.scheduler.cooldown
        params_to_log["scheduler_min_lr"] = config.train.scheduler.min_lr
    
    mlflow.log_params(params_to_log)

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
        "x_b_left": dataset["x_b_left"],
        "t_b_left": dataset["t_b_left"],
        "x_b_right": dataset["x_b_right"],
        "t_b_right": dataset["t_b_right"],
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
        "x_b_left": dataset["x_b_left"],
        "t_b_left": dataset["t_b_left"],
        "x_b_right": dataset["x_b_right"],
        "t_b_right": dataset["t_b_right"],
    }

    return train_dataset, val_dataset


def create_dataloaders(
    dataset: Dict[str, np.ndarray],
    batch_size: int,
    device: str,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for collocation, initial, and boundary points.

    Args:
        dataset: Dictionary with dataset arrays
        batch_size: Batch size for training
        device: Device to place tensors on

    Returns:
        Tuple of (collocation, initial, boundary_left, boundary_right)
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

    # Boundary condition data (now split)
    x_b_left = torch.from_numpy(dataset["x_b_left"]).float().unsqueeze(1).to(device)
    t_b_left = torch.from_numpy(dataset["t_b_left"]).float().unsqueeze(1).to(device)
    boundary_left_dataset = TensorDataset(x_b_left, t_b_left)
    boundary_left_loader = DataLoader(
        boundary_left_dataset,
        batch_size=len(boundary_left_dataset),
        shuffle=False,
    )
    
    x_b_right = torch.from_numpy(dataset["x_b_right"]).float().unsqueeze(1).to(device)
    t_b_right = torch.from_numpy(dataset["t_b_right"]).float().unsqueeze(1).to(device)
    boundary_right_dataset = TensorDataset(x_b_right, t_b_right)
    boundary_right_loader = DataLoader(
        boundary_right_dataset,
        batch_size=len(boundary_right_dataset),
        shuffle=False,
    )

    return collocation_loader, initial_loader, boundary_left_loader, boundary_right_loader


def train_epoch(
    model: nn.Module,
    loss_fn: SchrodingerLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    collocation_loader: DataLoader,
    initial_loader: DataLoader,
    boundary_left_loader: DataLoader,
    boundary_right_loader: DataLoader,
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
        boundary_left_loader: DataLoader for left boundary
        boundary_right_loader: DataLoader for right boundary
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
    x_b_left, t_b_left = next(iter(boundary_left_loader))
    x_b_right, t_b_right = next(iter(boundary_right_loader))
    
    # Enable AMP on CUDA only
    use_amp = (device.type == "cuda")

    # Iterate over collocation batches
    for x_f, t_f, u_f, v_f in collocation_loader:
        optimizer.zero_grad(set_to_none=True)

        # Forward pass with autocast (updated to torch.amp API)
        t_fwd0 = time.time()
        with torch.amp.autocast("cuda", enabled=use_amp):
            # Compute loss with all components
            total_loss, mse_0, mse_b, mse_f = loss_fn(
                model,
                x_0,
                t_0,
                u_0,
                v_0,
                x_b_left,
                t_b_left,
                x_b_right,
                t_b_right,
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

    # Create learning rate scheduler (if configured)
    scheduler = None
    if config.train.scheduler.type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.train.scheduler.factor,
            patience=config.train.scheduler.patience,
            cooldown=config.train.scheduler.cooldown,
            min_lr=config.train.scheduler.min_lr,
            threshold=config.train.scheduler.threshold,
            threshold_mode=config.train.scheduler.threshold_mode,
            verbose=verbose,
        )
        if verbose:
            print(f"\nScheduler: ReduceLROnPlateau (monitor=train_loss, patience={config.train.scheduler.patience})")
    elif config.train.scheduler.type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.train.scheduler.patience,
            gamma=config.train.scheduler.factor,
            verbose=verbose,
        )
        if verbose:
            print(f"\nScheduler: StepLR (step_size={config.train.scheduler.patience}, gamma={config.train.scheduler.factor})")
    elif config.train.scheduler.type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.train.epochs,
            eta_min=config.train.scheduler.min_lr,
            verbose=verbose,
        )
        if verbose:
            print(f"\nScheduler: CosineAnnealingLR (T_max={config.train.epochs})")

    # Create dataloaders
    (
        collocation_loader, 
        initial_loader, 
        boundary_left_loader, 
        boundary_right_loader
    ) = create_dataloaders(
        dataset, config.train.batch_size, str(device)
    )

    if verbose:
        print(
            f"\nDataset: {len(dataset['x_f'])} collocation, "
            f"{len(dataset['x_0'])} initial, "
            f"{len(dataset['x_b_left']) + len(dataset['x_b_right'])} boundary"
        )
        print(f"Batch size: {config.train.batch_size}")
        print(f"Batches per epoch: {len(collocation_loader)}")

    # Setup MLflow
    run_id = setup_mlflow(config)
    if verbose:
        print(f"\nMLflow run ID: {run_id}")

    # Initialize GradScaler for AMP (updated to torch.amp API)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Training loop
    if verbose:
        print(f"\nTraining for {config.train.epochs} epochs...")
        print("-" * 70)

    start_time = time.time()

    # Evaluation cadence: every 20% of epochs + final
    eval_interval = max(1, int(config.train.epochs * 0.2))
    
    # Use separate eval dataset if provided, otherwise use training dataset
    dataset_for_eval = eval_dataset if eval_dataset is not None else dataset

    # Track best eval L2 and save best_model.pt
    best_eval_l2 = float("inf")
    best_checkpoint_path: Optional[Path] = None
    
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
            boundary_left_loader,
            boundary_right_loader,
            device,
        )

        # Always evaluate relative L2 each epoch on eval dataset (lightweight)
        eval_metrics = evaluate(model, dataset_for_eval, device)

        # Log metrics each epoch (train + eval L2)
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
                # Eval L2
                "eval/relative_l2_error": eval_metrics["relative_l2_error"],
                "eval_time_prepare": eval_metrics.get("eval_time_prepare", 0.0),
                "eval_time_predict_metric": eval_metrics.get("eval_time_predict_metric", 0.0),
            },
            step=epoch,
        )

        epoch_time = time.time() - epoch_start

        # Update epoch time metric (log separately to avoid overwriting the dict above)
        mlflow.log_metric("time_epoch", epoch_time, step=epoch)

        # Step the learning rate scheduler (if configured)
        if scheduler is not None:
            if config.train.scheduler.type == "reduce_on_plateau":
                # ReduceLROnPlateau monitors training loss
                scheduler.step(train_metrics["total_loss"])
            else:
                # StepLR and CosineAnnealingLR step automatically
                scheduler.step()
            
            # Log current learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

        # Print progress each epoch (include L2)
        if verbose:
            l2_str = f"L²: {eval_metrics['relative_l2_error']:.6f}"
            lr_str = f"LR: {optimizer.param_groups[0]['lr']:.2e}" if scheduler is not None else ""
            print(
                f"Epoch {epoch:4d}/{config.train.epochs} | "
                f"Loss: {train_metrics['total_loss']:.6f} | "
                f"MSE₀: {train_metrics['mse_0']:.6f} | "
                f"MSE_b: {train_metrics['mse_b']:.6f} | "
                f"MSE_f: {train_metrics['mse_f']:.6f} | "
                f"{l2_str} | "
                f"{lr_str + ' | ' if lr_str else ''}"
                f"t_fwd={train_metrics.get('time/forward',0.0):.2f}s, "
                f"t_bwd={train_metrics.get('time/backward',0.0):.2f}s, "
                f"t_step={train_metrics.get('time/step',0.0):.2f}s | "
                f"epoch={epoch_time:.1f}s"
            )

        # Update best checkpoint every epoch based on eval L2
        if eval_metrics["relative_l2_error"] < best_eval_l2:
            best_eval_l2 = eval_metrics["relative_l2_error"]
            best_checkpoint_path = save_checkpoint(
                model,
                optimizer,
                epoch,
                train_metrics["total_loss"],
                config,
                filename="best_model.pt",
            )
            mlflow.log_artifact(str(best_checkpoint_path))

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
        "x_b_left": dataset["x_b_left"],
        "t_b_left": dataset["t_b_left"],
        "x_b_right": dataset["x_b_right"],
        "t_b_right": dataset["t_b_right"],
    }

    print(f"Using subset: {subset_size} collocation points")

    # Train
    model = train(config, dataset_subset, verbose=True)

    print("\n✓ Sanity test completed!")

