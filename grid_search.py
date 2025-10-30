"""Grid search script for hyperparameter optimization.

Runs multiple training configurations and saves all artifacts per run.
"""

import sys
import os
import itertools
from pathlib import Path
from typing import Dict, List
import yaml
import pandas as pd
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import mlflow
from train_full import train_full_model


def load_grid_config(config_path: str = "config/grid_search.yaml") -> Dict:
    """Load grid search configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_grid(parameters: Dict[str, List]) -> List[Dict]:
    """Generate all combinations of parameters."""
    param_names = list(parameters.keys())
    param_values = [parameters[name] for name in param_names]
    combinations = list(itertools.product(*param_values))
    
    grid = []
    for combo in combinations:
        grid.append(dict(zip(param_names, combo)))
    
    return grid


def run_grid_search():
    """Run grid search over hyperparameters."""
    
    print("=" * 80)
    print("GRID SEARCH - Schrödinger PINN Hyperparameter Optimization")
    print("=" * 80)
    
    # Load grid search config
    grid_config = load_grid_config()
    
    # Generate parameter grid
    param_grid = generate_grid(grid_config['grid_search']['parameters'])
    
    epochs = grid_config['grid_search']['epochs']
    base_config = grid_config['grid_search']['base_config']
    
    print(f"\nGrid Configuration:")
    print(f"  Total combinations: {len(param_grid)}")
    print(f"  Epochs per run: {epochs}")
    print(f"  Base config: {base_config}")
    print(f"  Parameters:")
    for param_name, param_values in grid_config['grid_search']['parameters'].items():
        print(f"    {param_name}: {param_values}")
    
    # Estimate time (rough: 4s per epoch)
    time_per_run = epochs * 4 / 60  # minutes
    total_time = len(param_grid) * time_per_run
    print(f"\n  Estimated time per run: {time_per_run:.1f} minutes")
    print(f"  Estimated total time: {total_time / 60:.1f} hours")
    
    # Confirm (support non-interactive mode via --yes or GS_AUTO_YES=1)
    auto_yes = ('--yes' in sys.argv) or (os.environ.get('GS_AUTO_YES') == '1')
    if not auto_yes:
        response = input(f"\nStart grid search with {len(param_grid)} runs? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return
    else:
        print(f"\n[Auto-Confirm] Proceeding with {len(param_grid)} runs.")
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(grid_config['grid_search']['experiment_name'])
    
    results = []
    start_time_total = datetime.now()
    
    # Modify config file temporarily to use grid search epochs
    temp_config_path = Path("config/train_temp.yaml")
    with open(base_config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    original_epochs = config_data['train']['epochs']
    config_data['train']['epochs'] = epochs
    config_data['train']['mlflow_experiment'] = grid_config['grid_search']['experiment_name']
    
    with open(temp_config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    # Run grid search
    for idx, params in enumerate(param_grid, 1):
        print(f"\n{'=' * 80}")
        print(f"Run {idx}/{len(param_grid)}")
        print(f"  weight_initial={params['weight_initial']}")
        print(f"  weight_residual={params['weight_residual']}")
        print(f"  batch_size={params['batch_size']}")
        if 'learning_rate' in params:
            print(f"  learning_rate={params['learning_rate']}")
        if 'scheduler_type' in params and params['scheduler_type'] is not None:
            print(f"  scheduler_type={params['scheduler_type']}")
            print(f"  scheduler_patience={params.get('scheduler_patience', 100)}")
        print(f"{'=' * 80}")
        
        run_start = datetime.now()
        
        try:
            # Update batch size and learning rate in temp config
            config_data['train']['batch_size'] = params['batch_size']
            if 'learning_rate' in params:
                config_data['train']['learning_rate'] = params['learning_rate']
            
            # Update scheduler configuration
            if 'scheduler' not in config_data['train']:
                config_data['train']['scheduler'] = {}
            
            config_data['train']['scheduler']['type'] = params.get('scheduler_type', None)
            if params.get('scheduler_type') is not None:
                config_data['train']['scheduler']['factor'] = params.get('scheduler_factor', 0.5)
                config_data['train']['scheduler']['patience'] = params.get('scheduler_patience', 100)
                config_data['train']['scheduler']['cooldown'] = params.get('scheduler_cooldown', 50)
                config_data['train']['scheduler']['min_lr'] = params.get('scheduler_min_lr', 1e-7)
            
            with open(temp_config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            # Run training with custom loss weights
            train_full_model(
                config_path=str(temp_config_path),
                weight_initial=params['weight_initial'],
                weight_boundary=params['weight_boundary'],
                weight_residual=params['weight_residual'],
            )
            
            # Get run info from MLflow (last run)
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(grid_config['grid_search']['experiment_name'])
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if runs:
                run = runs[0]
                run_id = run.info.run_id
                metrics = run.data.metrics
                mlflow_params = run.data.params
                
                result_entry = {
                    'run': idx,
                    'run_id': run_id,
                    'weight_initial': params['weight_initial'],
                    'weight_boundary': params['weight_boundary'],
                    'weight_residual': params['weight_residual'],
                    'batch_size': params['batch_size'],
                    'final_train_loss': metrics.get('train/total_loss'),
                    'final_val_loss': metrics.get('val/total_loss'),
                    'final_train_l2': metrics.get('train/relative_l2_error'),
                    'final_val_l2': metrics.get('val/relative_l2_error'),
                    'duration_minutes': (datetime.now() - run_start).total_seconds() / 60,
                    'timestamp': run_start.isoformat(),
                }
                
                # Add learning_rate if present in params
                if 'learning_rate' in params:
                    result_entry['learning_rate'] = params['learning_rate']
                
                # Add scheduler info if present in params
                if 'scheduler_type' in params:
                    result_entry['scheduler_type'] = params['scheduler_type']
                    if params['scheduler_type'] is not None:
                        result_entry['scheduler_patience'] = params.get('scheduler_patience')
                        result_entry['scheduler_factor'] = params.get('scheduler_factor')
                
                # Add key MLflow parameters for reference (epochs, hidden layers, etc.)
                result_entry['epochs'] = mlflow_params.get('epochs')
                result_entry['hidden_layers'] = mlflow_params.get('hidden_layers')
                result_entry['hidden_neurons'] = mlflow_params.get('hidden_neurons')
                result_entry['activation'] = mlflow_params.get('activation')
                
                results.append(result_entry)
                
                print(f"\n✓ Run {idx} completed | Run ID: {run_id}")
                if 'val/total_loss' in metrics:
                    print(f"  Final Val Loss: {metrics['val/total_loss']:.6f}")
                if 'val/relative_l2_error' in metrics:
                    print(f"  Final Val L²: {metrics['val/relative_l2_error']:.6f}")
            else:
                print(f"\n⚠ Run {idx} completed but couldn't retrieve metrics")
                results.append({
                    'run': idx,
                    'run_id': None,
                    **params,
                    'timestamp': run_start.isoformat(),
                })
        
        except Exception as e:
            print(f"\n✗ Run {idx} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'run': idx,
                'run_id': None,
                **params,
                'error': str(e),
                'timestamp': run_start.isoformat(),
            })
    
    # Cleanup temp config
    if temp_config_path.exists():
        temp_config_path.unlink()
    
    # Save results
    results_df = pd.DataFrame(results)
    # Coerce metric columns to numeric for sorting/summarization
    for col in [
        'final_val_loss', 'final_val_l2', 'final_train_loss', 'final_train_l2',
        'duration_minutes']:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
    results_file = Path(grid_config['grid_search']['results_file'])
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_file, index=False)
    
    total_duration = (datetime.now() - start_time_total).total_seconds() / 3600
    
    print("\n" + "=" * 80)
    print(f"Grid Search Complete! ({total_duration:.2f} hours)")
    print("=" * 80)
    print(f"\nResults saved to: {results_file}")
    
    # Summary
    if len(results_df) > 0 and 'final_val_loss' in results_df.columns:
        print("\nTop 5 Configurations (by validation loss):")
        print("-" * 80)
        top5 = results_df.dropna(subset=['final_val_loss']).nsmallest(5, 'final_val_loss')
        if len(top5) > 0:
            # Build column list dynamically to include learning_rate and scheduler_type if present
            display_cols = ['run', 'weight_initial', 'weight_residual', 'batch_size']
            if 'learning_rate' in top5.columns:
                display_cols.append('learning_rate')
            if 'scheduler_type' in top5.columns:
                display_cols.append('scheduler_type')
                if 'scheduler_patience' in top5.columns:
                    display_cols.append('scheduler_patience')
            display_cols.extend(['final_val_loss', 'final_val_l2'])
            print(top5[display_cols].to_string(index=False))
        else:
            print("No successful runs with metrics.")
    
    print("\n" + "=" * 80)
    print(f"All artifacts (plots, evaluations) saved per run ID")
    print(f"  Training evolution: outputs/plots/{{run_id}}/training_evolution.png")
    print(f"  Loss curves: outputs/plots/{{run_id}}/")
    print(f"  Evaluation: outputs/evaluation/{{run_id}}/")
    print("=" * 80)


if __name__ == "__main__":
    run_grid_search()
