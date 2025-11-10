"""Full training script with comprehensive tracking.

This script trains the model on the ENTIRE dataset (no split) with:
- All collocation points used for training
- MLflow tracking
- Loss curve plots saved after training
- All artifacts organized per experiment
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.data.generate_dataset import load_dataset
from src.train.engine import split_dataset
from src.utils.plotting import plot_loss_curves
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

from src.loss.physics_loss import SchrodingerLoss
from src.model.schrodinger_model import SchrodingerNet
from src.solver.nlse_solver import solve_nlse_full_grid
from src.utils.plotting import plot_solution_heatmap


def resolve_checkpoint_path(
    run_id: str = None,
    checkpoint_path: str = None,
    mlruns_path: str = "./mlruns"
) -> Path:
    """Resolve checkpoint path from MLflow run ID or direct path.

    This function follows the same logic as load_model_from_mlflow in
    ncc_analysis.py to ensure consistent checkpoint resolution.

    Args:
        run_id: MLflow run ID to find checkpoint from
        checkpoint_path: Direct path to checkpoint file
        mlruns_path: Path to MLflow tracking directory

    Returns:
        Path to checkpoint file

    Raises:
        FileNotFoundError: If checkpoint cannot be found
    """
    if checkpoint_path is not None:
        # Direct path provided
        path = Path(checkpoint_path)
        if not path.exists():
            msg = f"Checkpoint not found: {checkpoint_path}"
            raise FileNotFoundError(msg)
        return path

    if run_id is not None:
        import re
        import tempfile
        import os

        # Try to load from MLflow artifacts first
        mlflow.set_tracking_uri(f"file:{mlruns_path}")
        client = mlflow.tracking.MlflowClient()

        try:
            run = client.get_run(run_id)
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
                ckpt_artifacts = client.list_artifacts(
                    run_id, path="checkpoints"
                )
                for art in ckpt_artifacts:
                    candidate_paths.append(art.path)
            except Exception:
                pass

            # Prefer best_model.pt, then final_model.pt
            norm_paths = [p.replace("\\", "/") for p in candidate_paths]
            selected_artifact_path = None
            for name in ["best_model.pt", "final_model.pt"]:
                for p in norm_paths:
                    if p.endswith(name):
                        selected_artifact_path = p
                        break
                if selected_artifact_path is not None:
                    break

            # If still none, pick checkpoint with largest epoch number
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
                local_path = client.download_artifacts(
                    run_id, selected_artifact_path, temp_dir
                )
                # Return the downloaded path
                return Path(local_path)

        except Exception as e:
            print(f"  Could not load from MLflow artifacts: {e}")
            print("  Trying local checkpoint directory...")

        # Fallback to local checkpoint directory
        checkpoint_dir = Path("outputs/checkpoints")

        # Try best_model.pt first
        best_model_path = checkpoint_dir / "best_model.pt"
        if best_model_path.exists():
            return best_model_path

        # Try final_model.pt
        final_model_path = checkpoint_dir / "final_model.pt"
        if final_model_path.exists():
            return final_model_path

        # Try latest checkpoint_epoch*.pt if present locally
        import glob
        pattern = str(checkpoint_dir / "checkpoint_epoch*.pt")
        ckpts = glob.glob(pattern)
        if ckpts:
            # pick the one with the largest epoch
            def epoch_num(p):
                m = re.search(
                    r"checkpoint_epoch(\d+)\.pt$",
                    p.replace("\\", "/")
                )
                return int(m.group(1)) if m else -1
            latest = max(ckpts, key=epoch_num)
            return Path(latest)

        msg = (f"No checkpoint found for run_id: {run_id} "
               f"in MLflow or local directory")
        raise FileNotFoundError(msg)

    return None


def train_full_model(
    config_path: str = "config/train.yaml",
    weight_initial: float = 1.0,
    weight_boundary: float = 1.0,
    weight_residual: float = 1.0,
):
    """Train model on full dataset with comprehensive tracking.
    
    Args:
        config_path: Path to config file
        weight_initial: Weight for initial condition loss
        weight_boundary: Weight for boundary condition loss
        weight_residual: Weight for PDE residual loss
    """
    
    print("=" * 70)
    print("Full Training Run - Schrödinger PINN")
    print("=" * 70)
    
    # Load config
    config = load_config(config_path)
    
    # Determine device
    if config.train.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.train.device
    
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Set seeds
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)
    
    # Load full training dataset
    print(f"\nLoading training dataset: {config.dataset.save_path}")
    dataset_full = load_dataset(config.dataset.save_path)
    print(f"  Total collocation: {len(dataset_full['x_f'])}")
    print(f"  Initial: {len(dataset_full['x_0'])}")
    print(f"  Boundary: {len(dataset_full['x_b_left']) + len(dataset_full['x_b_right'])}")
    
    # Use entire dataset for training (no split)
    print("\nUsing ENTIRE dataset for training (no split)...")
    train_dataset = dataset_full
    
    # Load separate evaluation dataset
    eval_dataset_path = "data/processed/dataset_eval.npz"
    print(f"\nLoading evaluation dataset: {eval_dataset_path}")
    try:
        eval_dataset = load_dataset(eval_dataset_path)
        print(f"  Eval collocation: {len(eval_dataset['x_f'])}")
        print(f"  Eval initial: {len(eval_dataset['x_0'])}")
        print(f"  Eval boundary: {len(eval_dataset['x_b_left']) + len(eval_dataset['x_b_right'])}")
    except FileNotFoundError:
        print(f"  ⚠ Evaluation dataset not found, using training data for eval")
        eval_dataset = None
    
    # Create model
    print(f"\nCreating model...")
    model = SchrodingerNet(
        hidden_layers=config.train.hidden_layers,
        hidden_neurons=config.train.hidden_neurons,
        activation=config.train.activation,
    ).to(device)
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Create loss and optimizer
    loss_fn = SchrodingerLoss(
        weight_initial=weight_initial,
        weight_boundary=weight_boundary,
        weight_residual=weight_residual,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    
    # Load pretrained checkpoint if specified
    start_epoch = 1
    has_pretrained = (config.train.pretrained_checkpoint is not None or
                      config.train.pretrained_run_id is not None)
    if has_pretrained:
        try:
            checkpoint_path = resolve_checkpoint_path(
                run_id=config.train.pretrained_run_id,
                checkpoint_path=config.train.pretrained_checkpoint
            )

            if checkpoint_path is not None:
                print(f"\nLoading pretrained checkpoint: {checkpoint_path}")
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=device,
                    weights_only=False
                )

                # Load model state
                model.load_state_dict(checkpoint['model_state_dict'])
                print("  ✓ Model state loaded")

                # Load optimizer state if resuming training
                has_optimizer = 'optimizer_state_dict' in checkpoint
                if config.train.resume_training and has_optimizer:
                    optimizer.load_state_dict(
                        checkpoint['optimizer_state_dict']
                    )
                    print("  ✓ Optimizer state loaded")

                    # Resume from checkpoint epoch
                    if 'epoch' in checkpoint:
                        start_epoch = checkpoint['epoch'] + 1
                        print(f"  ✓ Resuming from epoch {start_epoch}")
                else:
                    msg = ("  ℹ Starting fresh training from "
                           "pretrained weights (epoch 1)")
                    print(msg)

                if 'loss' in checkpoint:
                    print(f"  Checkpoint loss: {checkpoint['loss']:.6f}")

        except FileNotFoundError as e:
            print(f"\n⚠ Warning: {e}")
            print("  Continuing with random initialization...")
        except Exception as e:
            print(f"\n⚠ Error loading checkpoint: {e}")
            print("  Continuing with random initialization...")
            import traceback
            traceback.print_exc()
    
    # Create learning rate scheduler (if configured)
    scheduler = None
    if config.train.scheduler.type is None:
        print("  Scheduler: None (disabled)")
    elif config.train.scheduler.type == "reduce_on_plateau":
        print(f"  Setting up ReduceLROnPlateau...")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.train.scheduler.factor,
            patience=config.train.scheduler.patience,
            cooldown=config.train.scheduler.cooldown,
            min_lr=config.train.scheduler.min_lr,
            threshold=config.train.scheduler.threshold,
            threshold_mode=config.train.scheduler.threshold_mode,
            verbose=True,
        )
        print(f"  Scheduler: ReduceLROnPlateau (patience={config.train.scheduler.patience}, factor={config.train.scheduler.factor})")
    elif config.train.scheduler.type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.train.scheduler.patience,
            gamma=config.train.scheduler.factor,
            verbose=True,
        )
        print(f"  Scheduler: StepLR (step_size={config.train.scheduler.patience})")
    elif config.train.scheduler.type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.train.epochs,
            eta_min=config.train.scheduler.min_lr,
            verbose=True,
        )
        print(f"  Scheduler: CosineAnnealingLR (T_max={config.train.epochs})")
    
    print(f"  Loss weights: IC={weight_initial}, BC={weight_boundary}, Residual={weight_residual}")
    
    # Create dataloaders (ensure float32 for GPU efficiency)
    def create_loaders(ds, batch_size):
        x_f = torch.from_numpy(ds["x_f"]).float().unsqueeze(1).to(device)
        t_f = torch.from_numpy(ds["t_f"]).float().unsqueeze(1).to(device)
        u_f = torch.from_numpy(ds["u_f"]).float().to(device)
        v_f = torch.from_numpy(ds["v_f"]).float().to(device)
        collocation_dataset = TensorDataset(x_f, t_f, u_f, v_f)
        collocation_loader = DataLoader(collocation_dataset, batch_size=batch_size, shuffle=True)
        
        x_0 = torch.from_numpy(ds["x_0"]).float().unsqueeze(1).to(device)
        t_0 = torch.from_numpy(ds["t_0"]).float().unsqueeze(1).to(device)
        u_0 = torch.from_numpy(ds["u_0"]).float().to(device)
        v_0 = torch.from_numpy(ds["v_0"]).float().to(device)
        initial_dataset = TensorDataset(x_0, t_0, u_0, v_0)
        initial_loader = DataLoader(initial_dataset, batch_size=len(initial_dataset), shuffle=False)
        
        x_b_left = torch.from_numpy(ds["x_b_left"]).float().unsqueeze(1).to(device)
        t_b_left = torch.from_numpy(ds["t_b_left"]).float().unsqueeze(1).to(device)
        boundary_left_dataset = TensorDataset(x_b_left, t_b_left)
        boundary_left_loader = DataLoader(boundary_left_dataset, batch_size=len(boundary_left_dataset), shuffle=False)
        
        x_b_right = torch.from_numpy(ds["x_b_right"]).float().unsqueeze(1).to(device)
        t_b_right = torch.from_numpy(ds["t_b_right"]).float().unsqueeze(1).to(device)
        boundary_right_dataset = TensorDataset(x_b_right, t_b_right)
        boundary_right_loader = DataLoader(boundary_right_dataset, batch_size=len(boundary_right_dataset), shuffle=False)
        
        return collocation_loader, initial_loader, boundary_left_loader, boundary_right_loader
    
    train_coll_loader, train_init_loader, train_bound_left_loader, train_bound_right_loader = create_loaders(train_dataset, config.train.batch_size)
    
    print(f"  Batch size: {config.train.batch_size}")
    print(f"  Train batches/epoch: {len(train_coll_loader)}")
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(config.train.mlflow_experiment)
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\nMLflow run ID: {run_id}")
        
        # Log parameters
        log_params = {
            "epochs": config.train.epochs,
            "learning_rate": config.train.learning_rate,
            "batch_size": config.train.batch_size,
            "hidden_layers": config.train.hidden_layers,
            "hidden_neurons": config.train.hidden_neurons,
            "activation": config.train.activation,
            "device": device,
            "seed": config.train.seed,
            "n_train": len(train_dataset["x_f"]),
            "weight_initial": weight_initial,
            "weight_boundary": weight_boundary,
            "weight_residual": weight_residual,
        }
        
        # Add scheduler parameters if configured
        if config.train.scheduler.type is not None:
            log_params["scheduler_type"] = config.train.scheduler.type
            log_params["scheduler_factor"] = config.train.scheduler.factor
            log_params["scheduler_patience"] = config.train.scheduler.patience
            log_params["scheduler_cooldown"] = config.train.scheduler.cooldown
            log_params["scheduler_min_lr"] = config.train.scheduler.min_lr
        
        mlflow.log_params(log_params)
        
        # Training history
        history = {
            "train/total_loss": [],
            "train/mse_0": [],
            "train/mse_b": [],
            "train/mse_f": [],
            "train/relative_l2_error": [],
            "eval/relative_l2_error": [],
        }
        
        # Create eval loader if eval dataset available
        if eval_dataset is not None:
            eval_coll_loader, _, _, _ = create_loaders(eval_dataset, config.train.batch_size)
        
        print(f"\nTraining for {config.train.epochs} epochs...")
        print("-" * 70)
        
        start_time = time.time()
        
        # Store checkpoints for evolution plot at the end
        evolution_checkpoints = []
        evolution_epochs = [1, 250, 500, 750, 1000] if config.train.epochs >= 1000 else [1, config.train.epochs // 4, config.train.epochs // 2, 3 * config.train.epochs // 4, config.train.epochs]
        
        # Initialize AMP scaler (updated to torch.amp API)
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
        use_amp = (device == "cuda")
        
        # Evaluation cadence: every 20% of epochs + final
        eval_interval = max(1, int(config.train.epochs * 0.2))

        # Best-checkpoint tracking (prefer eval L2 if eval set available)
        best_metric = float("inf")
        best_metric_name = "eval/relative_l2_error" if eval_dataset is not None else "train/relative_l2_error"
        best_checkpoint_path = Path(config.train.checkpoint_dir) / "best_model.pt"
        
        for epoch in range(start_epoch, config.train.epochs + 1):
            epoch_start = time.time()
            # Training
            model.train()
            train_loss_sum = 0.0
            train_mse0_sum = 0.0
            train_mseb_sum = 0.0
            train_msef_sum = 0.0
            n_batches = 0
            
            t_fetch0 = time.time()
            x_0, t_0, u_0, v_0 = next(iter(train_init_loader))
            x_b_left, t_b_left = next(iter(train_bound_left_loader))
            x_b_right, t_b_right = next(iter(train_bound_right_loader))
            t_fetch1 = time.time()
            
            # Timing accumulators (per-batch averages)
            time_forward_sum = 0.0
            time_backward_sum = 0.0
            time_step_sum = 0.0
            time_ic_sum = 0.0
            time_bc_sum = 0.0
            time_pde_sum = 0.0
            time_pde_predict_sum = 0.0
            time_pde_derivatives_sum = 0.0
            time_pde_residual_sum = 0.0

            t_batches0 = time.time()
            for x_f, t_f, u_f, v_f in train_coll_loader:
                optimizer.zero_grad(set_to_none=True)
                
                # Forward pass with AMP (updated to torch.amp API)
                t_fwd0 = time.time()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    total_loss, mse_0, mse_b, mse_f = loss_fn(
                        model, x_0, t_0, u_0, v_0, 
                        x_b_left, t_b_left, x_b_right, t_b_right,
                        x_f, t_f,
                        return_components=True
                    )
                t_fwd1 = time.time()
                
                # Backward pass with gradient scaling
                t_bwd0 = t_fwd1
                scaler.scale(total_loss).backward()
                t_bwd1 = time.time()
                scaler.step(optimizer)
                t_step1 = time.time()
                scaler.update()
                
                train_loss_sum += total_loss.item()
                train_mse0_sum += mse_0.item()
                train_mseb_sum += mse_b.item()
                train_msef_sum += mse_f.item()
                n_batches += 1

                # Timing aggregation
                time_forward_sum += (t_fwd1 - t_fwd0)
                time_backward_sum += (t_bwd1 - t_bwd0)
                time_step_sum += (t_step1 - t_bwd1)
                lt = getattr(loss_fn, "last_timings", {}) or {}
                time_ic_sum += float(lt.get("time/ic", 0.0))
                time_bc_sum += float(lt.get("time/bc", 0.0))
                time_pde_sum += float(lt.get("time/pde", 0.0))
                time_pde_predict_sum += float(lt.get("time/pde/predict", 0.0))
                time_pde_derivatives_sum += float(lt.get("time/pde/derivatives", 0.0))
                time_pde_residual_sum += float(lt.get("time/pde/residual", 0.0))
            t_batches1 = time.time()
            
            train_metrics = {
                "total_loss": train_loss_sum / n_batches,
                "mse_0": train_mse0_sum / n_batches,
                "mse_b": train_mseb_sum / n_batches,
                "mse_f": train_msef_sum / n_batches,
                # per-batch average timings
                "time_forward": time_forward_sum / n_batches,
                "time_backward": time_backward_sum / n_batches,
                "time_step": time_step_sum / n_batches,
                "time_ic": time_ic_sum / n_batches,
                "time_bc": time_bc_sum / n_batches,
                "time_pde": time_pde_sum / n_batches,
                "time_pde_predict": time_pde_predict_sum / n_batches,
                "time_pde_derivatives": time_pde_derivatives_sum / n_batches,
                "time_pde_residual": time_pde_residual_sum / n_batches,
            }
            
            # Compute phase-invariant L2 error on training data (no grad)
            def compute_l2(loader, subset_size=0):
                from src.utils.metrics import phase_aligned_rel_l2_torch
                t_prep0 = time.time()
                x, t, u, v = [], [], [], []
                for batch in loader:
                    x.append(batch[0])
                    t.append(batch[1])
                    u.append(batch[2])
                    v.append(batch[3])
                x = torch.cat(x)
                t = torch.cat(t)
                u = torch.cat(u)
                v = torch.cat(v)
                
                # Optionally subsample for faster evaluation
                if subset_size > 0 and len(x) > subset_size:
                    idx = torch.randperm(len(x), device=x.device)[:subset_size]
                    x = x[idx]
                    t = t[idx]
                    u = u[idx]
                    v = v[idx]
                
                t_prep1 = time.time()
                model.eval()
                with torch.no_grad():
                    t_pred0 = time.time()
                    uv_pred = model(x, t)
                    t_pred1 = time.time()
                    h_pred = torch.complex(uv_pred[:, 0], uv_pred[:, 1])
                    h_true = torch.complex(u, v)
                    # Use phase-aligned relative L² error
                    rel = phase_aligned_rel_l2_torch(h_pred, h_true).item()
                return rel, (t_prep1 - t_prep0), (t_pred1 - t_pred0)
            
            eval_subset = config.train.eval_subset_size
            train_l2, time_l2_prep, time_l2_pred = compute_l2(train_coll_loader, eval_subset)
            
            # Update history
            history["train/total_loss"].append(train_metrics["total_loss"])
            history["train/mse_0"].append(train_metrics["mse_0"])
            history["train/mse_b"].append(train_metrics["mse_b"])
            history["train/mse_f"].append(train_metrics["mse_f"])
            history["train/relative_l2_error"].append(train_l2)
            
            # Evaluate relative L2 on eval dataset every epoch (lightweight) if available
            t_ml0 = time.time()
            if eval_dataset is not None:
                eval_l2, eval_time_l2_prep, eval_time_l2_pred = compute_l2(eval_coll_loader)
                history["eval/relative_l2_error"].append(eval_l2)
                # Update best checkpoint every 100 epochs (and at the final epoch) using eval L2
                if (epoch % 100 == 0 or epoch == config.train.epochs) and eval_l2 < best_metric:
                    best_metric = eval_l2
                    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": train_metrics["total_loss"],
                        "best_metric_name": best_metric_name,
                        "best_metric_value": best_metric,
                        "config": config,
                    }, best_checkpoint_path)
                    mlflow.log_artifact(str(best_checkpoint_path))
                    print(f"  ✓ New best {best_metric_name}: {best_metric:.6f} at epoch {epoch} → saved best_model.pt")
            else:
                eval_l2, eval_time_l2_prep, eval_time_l2_pred = (train_l2, 0.0, 0.0)

            # Log metrics each epoch
            mlflow.log_metrics({
                "train/total_loss": train_metrics["total_loss"],
                "train/mse_0": train_metrics["mse_0"],
                "train/mse_b": train_metrics["mse_b"],
                "train/mse_f": train_metrics["mse_f"],
                "train/relative_l2_error": train_l2,
                "eval/relative_l2_error": eval_l2,
                # timings
                "time_forward": train_metrics.get("time_forward", 0.0),
                "time_backward": train_metrics.get("time_backward", 0.0),
                "time_step": train_metrics.get("time_step", 0.0),
                "time_ic": train_metrics.get("time_ic", 0.0),
                "time_bc": train_metrics.get("time_bc", 0.0),
                "time_pde": train_metrics.get("time_pde", 0.0),
                "time_pde_predict": train_metrics.get("time_pde_predict", 0.0),
                "time_pde_derivatives": train_metrics.get("time_pde_derivatives", 0.0),
                "time_pde_residual": train_metrics.get("time_pde_residual", 0.0),
                "time_fetch_icbc": (t_fetch1 - t_fetch0),
                "time_batches": (t_batches1 - t_batches0),
                "time_l2_prepare": time_l2_prep,
                "time_l2_predict": time_l2_pred,
                "eval_time_l2_prepare": eval_time_l2_prep,
                "eval_time_l2_predict": eval_time_l2_pred,
            }, step=epoch)
            t_ml1 = time.time()
            
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
            
            # Compute epoch time
            epoch_time = time.time() - epoch_start
            elapsed_total = time.time() - start_time
            
            # Print progress (every epoch for short runs, periodically for long runs)
            print_freq = 1 if config.train.epochs <= 50 else max(1, config.train.epochs // 20)
            if epoch % print_freq == 0 or epoch == 1:
                lr_str = f" | LR: {optimizer.param_groups[0]['lr']:.2e}" if scheduler is not None else ""
                print(
                    f"Epoch {epoch:4d}/{config.train.epochs} | "
                    f"Loss: {train_metrics['total_loss']:.6f} | "
                    f"L²: {train_l2:.6f}{lr_str} | "
                    f"t_fwd={train_metrics.get('time_forward',0.0):.2f}s, "
                    f"t_bwd={train_metrics.get('time_backward',0.0):.2f}s, "
                    f"t_step={train_metrics.get('time_step',0.0):.2f}s | "
                    f"batches={t_batches1 - t_batches0:.2f}s, "
                    f"l2_prep={time_l2_prep:.2f}s, l2_pred={time_l2_pred:.2f}s, "
                    f"mlflow={t_ml1 - t_ml0:.2f}s | "
                    f"epoch={epoch_time:.1f}s | "
                    f"Total: {elapsed_total:.1f}s"
                )
            
            # Save periodic checkpoints (if enabled)
            if config.train.checkpoint_ratio > 0:
                checkpoint_interval = max(1, int(config.train.epochs * config.train.checkpoint_ratio))
                if epoch % checkpoint_interval == 0:
                    checkpoint_path = Path(config.train.checkpoint_dir) / f"checkpoint_epoch{epoch:04d}.pt"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": train_metrics["total_loss"],
                        "config": config,
                    }, checkpoint_path)
                    mlflow.log_artifact(str(checkpoint_path))
            
            # Save model state for evolution plot at specific epochs
            if epoch in evolution_epochs:
                evolution_checkpoints.append({
                    "epoch": epoch,
                    "model_state": model.state_dict().copy(),
                })
        
        total_time = time.time() - start_time
        
        # Save final model
        final_checkpoint = Path(config.train.checkpoint_dir) / "final_model.pt"
        torch.save({
            "epoch": config.train.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": train_metrics["total_loss"],
            "config": config,
        }, final_checkpoint)
        mlflow.log_artifact(str(final_checkpoint))
        
        # Generate and save loss curves
        print("\nGenerating loss curves...")
        plots_dir = Path("outputs") / "plots" / run_id
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Rename keys for plotting
        plot_history = {
            "total_loss": history["train/total_loss"],
            "mse_0": history["train/mse_0"],
            "mse_b": history["train/mse_b"],
            "mse_f": history["train/mse_f"],
            "relative_l2_error": history["train/relative_l2_error"],
        }
        
        loss_curve_path = plots_dir / "training_curves.png"
        plot_loss_curves(plot_history, title="Training Loss Curves", save_path=loss_curve_path)
        mlflow.log_artifact(str(loss_curve_path))
        
        
        print(f"  ✓ Loss curves saved to: {plots_dir}")
        
        # Generate training evolution plot
        print("\nGenerating training evolution plot...")
        evolution_plot_path = plots_dir / "training_evolution.png"
        
        # Generate ground truth
        x_grid_gt, t_grid_gt, h_solution_gt = solve_nlse_full_grid(
            x_min=config.solver.x_min,
            x_max=config.solver.x_max,
            t_min=config.solver.t_min,
            t_max=config.solver.t_max,
            nx=256,
            nt=100,
            alpha=config.solver.alpha,
        )
        
        # Create evolution plot
        n_checkpoints = len(evolution_checkpoints)
        fig, axes = plt.subplots(1, n_checkpoints, figsize=(4 * n_checkpoints, 4))
        if n_checkpoints == 1:
            axes = [axes]
        
        for idx, checkpoint_data in enumerate(evolution_checkpoints):
            # Load model state
            model.load_state_dict(checkpoint_data["model_state"])
            model.eval()
            
            # Generate predictions
            with torch.no_grad():
                x_mesh, t_mesh = np.meshgrid(x_grid_gt, t_grid_gt)
                x_flat = torch.from_numpy(x_mesh.flatten()).unsqueeze(1).float().to(device)
                t_flat = torch.from_numpy(t_mesh.flatten()).unsqueeze(1).float().to(device)
                
                batch_size_pred = 2048
                h_pred_list = []
                for i in range(0, len(x_flat), batch_size_pred):
                    x_batch = x_flat[i:i+batch_size_pred]
                    t_batch = t_flat[i:i+batch_size_pred]
                    uv_pred = model(x_batch, t_batch)
                    h_pred_batch = torch.complex(uv_pred[:, 0], uv_pred[:, 1])
                    h_pred_list.append(h_pred_batch.cpu())
                
                h_pred_flat = torch.cat(h_pred_list)
                h_pred_grid = h_pred_flat.numpy().reshape(len(t_grid_gt), len(x_grid_gt))
            
            # Plot
            im = axes[idx].imshow(
                np.abs(h_pred_grid),
                aspect='auto',
                origin='lower',
                extent=[x_grid_gt[0], x_grid_gt[-1], t_grid_gt[0], t_grid_gt[-1]],
                cmap='viridis'
            )
            axes[idx].set_xlabel('x')
            if idx == 0:
                axes[idx].set_ylabel('t')
            axes[idx].set_title(f'Epoch {checkpoint_data["epoch"]}')
            plt.colorbar(im, ax=axes[idx], label='|h(x,t)|')
        
        plt.tight_layout()
        plt.savefig(evolution_plot_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(str(evolution_plot_path))
        plt.close()
        
        print(f"  ✓ Evolution plot saved to: {evolution_plot_path}")
        
        print("\n" + "-" * 70)
        print(f"Training completed in {total_time / 60:.1f} minutes")
        print(f"Final train loss: {train_metrics['total_loss']:.6f}")
        print(f"Final train L² error: {train_l2:.6f}")
        if eval_dataset is not None and len(history["eval/relative_l2_error"]) > 0:
            print(f"Final eval L² error: {history['eval/relative_l2_error'][-1]:.6f}")
        print(f"\nAll artifacts saved to MLflow run: {run_id}")
        print(f"Loss curves: {plots_dir}")
        print(f"Checkpoints: {Path(config.train.checkpoint_dir)}")
        
        # Save run metadata to plots directory for easy reference
        import json
        metadata = {
            "run_id": run_id,
            "experiment": config.train.mlflow_experiment,
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "epochs": config.train.epochs,
                "learning_rate": config.train.learning_rate,
                "batch_size": config.train.batch_size,
                "hidden_layers": config.train.hidden_layers,
                "hidden_neurons": config.train.hidden_neurons,
                "activation": config.train.activation,
                "weight_initial": weight_initial,
                "weight_boundary": weight_boundary,
                "weight_residual": weight_residual,
                "seed": config.train.seed,
            },
            "final_metrics": {
                "train_loss": train_metrics['total_loss'],
                "train_l2_error": train_l2,
                "eval_l2_error": history['eval/relative_l2_error'][-1] if eval_dataset is not None and len(history["eval/relative_l2_error"]) > 0 else None,
            }
        }
        metadata_path = plots_dir / "run_info.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Run automatic evaluation on final model
        print("\n" + "=" * 70)
        print("Running Automatic Evaluation on Final Model")
        print("=" * 70)
        
        from src.evaluate.evaluate_model import evaluate_model
        
        final_checkpoint_path = Path(config.train.checkpoint_dir) / "final_model.pt"
        # Prefer best checkpoint if it exists
        checkpoint_to_evaluate = best_checkpoint_path if best_checkpoint_path.exists() else final_checkpoint_path
        if checkpoint_to_evaluate == best_checkpoint_path:
            print(f"Evaluating best checkpoint (based on {best_metric_name}): {checkpoint_to_evaluate}")
        else:
            print(f"Best checkpoint not found; evaluating final model: {checkpoint_to_evaluate}")
        eval_output_dir = Path("outputs") / "evaluation" / run_id
        
        try:
            evaluate_model(
                checkpoint_path=checkpoint_to_evaluate,
                output_dir=eval_output_dir,
                device=device,
            )
            
            # Save run metadata to evaluation directory as well
            eval_metadata_path = eval_output_dir / "run_info.json"
            with open(eval_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Log evaluation artifacts to MLflow
            for file in eval_output_dir.glob("*.png"):
                mlflow.log_artifact(str(file), artifact_path="evaluation")
            
            metrics_file = eval_output_dir / "metrics.json"
            if metrics_file.exists():
                mlflow.log_artifact(str(metrics_file), artifact_path="evaluation")
            
            print(f"\n✓ Evaluation complete! Results saved to: {eval_output_dir}")
            
        except Exception as e:
            print(f"\n✗ Evaluation failed: {e}")
            print("  You can run evaluation manually later.")
        
        print("=" * 70)


if __name__ == "__main__":
    train_full_model()

