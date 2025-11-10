# Loading Pretrained Models

This guide explains how to load pretrained model checkpoints and continue training from them.

## Configuration Options

Three new configuration parameters have been added to `train.yaml`:

### 1. `pretrained_checkpoint`
Direct path to a checkpoint file (`.pt` file).

**Example:**
```yaml
train:
  pretrained_checkpoint: "outputs/checkpoints/best_model.pt"
```

### 2. `pretrained_run_id`
MLflow run ID to load the checkpoint from. The system will automatically:
1. Search MLflow artifacts (both root and `checkpoints/` subdirectory)
2. Prefer `best_model.pt`, then `final_model.pt`, then the latest `checkpoint_epoch*.pt`
3. Download the checkpoint from MLflow to a temporary directory
4. Fall back to local `outputs/checkpoints/` if not found in MLflow

**Example:**
```yaml
train:
  pretrained_run_id: "abc123def456789"
```

**Note:** 
- If both `pretrained_checkpoint` and `pretrained_run_id` are set, `pretrained_checkpoint` takes priority.
- The checkpoint resolution follows the same logic as `ncc_analysis.py` for consistency.

### 3. `resume_training`
Controls whether to resume training from the checkpoint epoch or start fresh.

- `true` (default): Resume from the epoch where the checkpoint was saved
  - Loads both model weights and optimizer state
  - Training continues from `checkpoint_epoch + 1`
  
- `false`: Start from epoch 1 with pretrained weights
  - Only loads model weights (not optimizer state)
  - Useful for fine-tuning or transfer learning

**Example:**
```yaml
train:
  resume_training: true  # Resume from checkpoint epoch
  # or
  resume_training: false  # Start from epoch 1 with pretrained weights
```

## Finding MLflow Run IDs

If you want to load from a specific MLflow run, you need to find its run ID. Here are several ways:

### Method 1: From Training Output
When training completes, the run ID is printed:
```
MLflow run ID: abc123def456789
```

### Method 2: From Grid Search Results
Check the `grid_search_results.csv` file:
```csv
run,run_id,weight_initial,weight_boundary,weight_residual,...
1,abc123def456789,1.0,1.0,1.0,...
```

### Method 3: From MLflow UI
```bash
mlflow ui
```
Then open http://localhost:5000 in your browser and copy the run ID from the UI.

### Method 4: Programmatically
```python
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("schrodinger_step1")
runs = client.search_runs(experiment_ids=[experiment.experiment_id])
for run in runs:
    print(f"Run ID: {run.info.run_id}")
```

## Usage Examples

### Example 1: Resume Training from a Specific Checkpoint

Create or modify your `config/train.yaml`:

```yaml
train:
  epochs: 2000
  learning_rate: 0.0001
  batch_size: 512
  
  # Load pretrained checkpoint and resume training
  pretrained_checkpoint: "outputs/checkpoints/best_model.pt"
  resume_training: true
  
  mlflow_experiment: schrodinger_continued
  checkpoint_dir: outputs/checkpoints
```

Then run:
```bash
python train_full.py
```

### Example 2: Fine-tune from Pretrained Weights

Create a new config file `config/finetune.yaml`:

```yaml
train:
  epochs: 1000
  learning_rate: 0.00001  # Lower learning rate for fine-tuning
  batch_size: 512
  
  # Load pretrained weights but start from epoch 1
  pretrained_checkpoint: "outputs/checkpoints/best_model.pt"
  resume_training: false
  
  mlflow_experiment: schrodinger_finetune
  checkpoint_dir: outputs/checkpoints_finetune
```

Then run:
```bash
python train_full.py --config config/finetune.yaml
```

### Example 3: Continue from MLflow Run ID

If you know the MLflow run ID:

```yaml
train:
  epochs: 2000
  
  # Load from MLflow run
  pretrained_run_id: "a1b2c3d4e5f6g7h8"
  resume_training: true
  
  mlflow_experiment: schrodinger_continued
```

### Example 4: Grid Search with Pretrained Weights

You can also use pretrained models in grid search. Modify your base config that grid search uses:

```yaml
# config/grid_search_continued.yaml
train:
  epochs: 1000
  
  # Start all grid search runs from this pretrained model
  pretrained_checkpoint: "outputs/checkpoints/best_model.pt"
  resume_training: false  # Start fresh for fair comparison
  
  mlflow_experiment: schrodinger_grid_search_v2
```

## What Gets Loaded

When loading a checkpoint, the following are available:

- `model_state_dict`: Model weights (always loaded)
- `optimizer_state_dict`: Optimizer state (loaded if `resume_training=true`)
- `epoch`: Checkpoint epoch number (used if `resume_training=true`)
- `loss`: Training loss at checkpoint time (displayed for info)
- `config`: Original training configuration (not used, preserved for reference)

## Error Handling

The training script will gracefully handle missing or invalid checkpoints:

- If checkpoint file is not found, training continues with random initialization
- If checkpoint is corrupted, training continues with random initialization
- Errors are logged but don't stop the training process

## Best Practices

1. **For resuming interrupted training:** Use `resume_training: true`
2. **For fine-tuning:** Use `resume_training: false` with a lower learning rate
3. **For transfer learning:** Use `resume_training: false` and possibly modify the loss weights
4. **Always specify the checkpoint path explicitly** when running important experiments
5. **Keep your best checkpoints** - the system saves both `best_model.pt` and `final_model.pt`

## Verification

When training starts, you'll see output like:

```
Loading pretrained checkpoint: outputs/checkpoints/best_model.pt
  ✓ Model state loaded
  ✓ Optimizer state loaded
  ✓ Resuming from epoch 1001
  Checkpoint loss: 0.002345
```

Or for fresh training from pretrained weights:

```
Loading pretrained checkpoint: outputs/checkpoints/best_model.pt
  ✓ Model state loaded
  ℹ Starting fresh training from pretrained weights (epoch 1)
  Checkpoint loss: 0.002345
```

