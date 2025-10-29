#!/usr/bin/env python3
"""Query MLflow run parameters by run_id.

Usage:
    python get_run_params.py <run_id>
    python get_run_params.py 9f610e032baa490e8c5da94f4d2f8714
    
Or from Python:
    from get_run_params import get_run_params
    params = get_run_params("9f610e032baa490e8c5da94f4d2f8714")
"""

import sys
from pathlib import Path
import mlflow
from typing import Dict, Any, Optional


def get_run_params(run_id: str, mlruns_path: str = "./mlruns") -> Optional[Dict[str, Any]]:
    """Get all parameters and key metrics for a given MLflow run_id.
    
    Args:
        run_id: MLflow run ID (found in outputs/plots/{run_id}/ or outputs/evaluation/{run_id}/)
        mlruns_path: Path to MLflow tracking directory (default: ./mlruns)
    
    Returns:
        Dictionary with parameters, metrics, and metadata, or None if run not found
    """
    mlflow.set_tracking_uri(f"file:{mlruns_path}")
    
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        result = {
            "run_id": run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "parameters": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "tags": dict(run.data.tags),
        }
        
        return result
        
    except Exception as e:
        print(f"Error: Could not find run {run_id}")
        print(f"Details: {e}")
        return None


def print_run_info(run_info: Dict[str, Any]) -> None:
    """Pretty-print run information."""
    if not run_info:
        return
    
    print("=" * 80)
    print(f"MLflow Run: {run_info['run_id']}")
    print("=" * 80)
    
    print("\n📋 TRAINING PARAMETERS:")
    print("-" * 80)
    params = run_info['parameters']
    
    # Group parameters by category
    training_params = ['epochs', 'learning_rate', 'batch_size', 'optimizer', 'scheduler']
    model_params = ['hidden_layers', 'hidden_neurons', 'activation']
    loss_params = ['weight_initial', 'weight_boundary', 'weight_residual']
    data_params = ['n_train', 'seed']
    other_params = ['device', 'dtype', 'deterministic']
    
    print("\n  Training:")
    for p in training_params:
        if p in params:
            print(f"    {p:20s} = {params[p]}")
    
    print("\n  Model Architecture:")
    for p in model_params:
        if p in params:
            print(f"    {p:20s} = {params[p]}")
    
    print("\n  Loss Weights:")
    for p in loss_params:
        if p in params:
            print(f"    {p:20s} = {params[p]}")
    
    print("\n  Data & Other:")
    for p in data_params + other_params:
        if p in params:
            print(f"    {p:20s} = {params[p]}")
    
    # Show any additional params not in the above categories
    shown_params = set(training_params + model_params + loss_params + data_params + other_params)
    remaining = {k: v for k, v in params.items() if k not in shown_params}
    if remaining:
        print("\n  Additional:")
        for k, v in remaining.items():
            print(f"    {k:20s} = {v}")
    
    print("\n" + "-" * 80)
    print("\n📊 FINAL METRICS:")
    print("-" * 80)
    metrics = run_info['metrics']
    
    key_metrics = [
        'train/total_loss',
        'train/relative_l2_error',
        'eval/relative_l2_error',
        'val/total_loss',
        'val/relative_l2_error',
    ]
    
    for m in key_metrics:
        if m in metrics:
            print(f"  {m:30s} = {metrics[m]:.6f}")
    
    # Show timing if available
    timing_metrics = {k: v for k, v in metrics.items() if 'time' in k.lower()}
    if timing_metrics:
        print("\n  Timing:")
        for k, v in timing_metrics.items():
            print(f"    {k:28s} = {v:.4f}s")
    
    print("\n" + "=" * 80)
    print(f"\n💾 Outputs for this run:")
    print(f"  Plots:      outputs/plots/{run_info['run_id']}/")
    print(f"  Evaluation: outputs/evaluation/{run_info['run_id']}/")
    print(f"  MLflow UI:  mlflow ui --backend-store-uri ./mlruns")
    print("=" * 80)


def main():
    """Command-line interface."""
    if len(sys.argv) < 2:
        print("Usage: python get_run_params.py <run_id>")
        print("\nExample:")
        print("  python get_run_params.py 9f610e032baa490e8c5da94f4d2f8714")
        print("\nTo find run_id:")
        print("  - Look at folder names in outputs/plots/")
        print("  - Look at folder names in outputs/evaluation/")
        print("  - Check grid_search_results.csv")
        sys.exit(1)
    
    run_id = sys.argv[1]
    run_info = get_run_params(run_id)
    
    if run_info:
        print_run_info(run_info)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

