# Learning Rate Scheduler Implementation

## Overview

The training pipeline now supports learning rate schedulers to automatically adjust the learning rate during training based on training loss. This is a traditional approach to improve convergence and find better local minima.

## Implementation Details

### Supported Schedulers

1. **ReduceLROnPlateau** (recommended for traditional loss-based scheduling)
   - Monitors: Training loss (`train/total_loss`)
   - Reduces LR by `factor` when loss plateaus for `patience` epochs
   - Waits `cooldown` epochs after each reduction
   - Stops reducing when LR reaches `min_lr`
   
2. **StepLR**
   - Reduces LR by `factor` (gamma) every `step_size` epochs
   - Fixed schedule, doesn't monitor metrics
   
3. **CosineAnnealingLR**
   - Gradually reduces LR following a cosine curve
   - Reaches `min_lr` at the end of training

### Configuration

#### Base Config (`config/train.yaml`)

```yaml
train:
  scheduler:
    type: reduce_on_plateau  # or 'step', 'cosine', or null
    factor: 0.5              # Multiplicative factor for LR reduction
    patience: 100            # Epochs to wait before reducing (for ReduceLROnPlateau)
    cooldown: 50             # Epochs to wait after reduction
    min_lr: 1.0e-7           # Minimum learning rate
    threshold: 1.0e-4        # Threshold for measuring improvement
    threshold_mode: rel      # 'rel' or 'abs'
```

#### Grid Search (`config/grid_search.yaml`)

You can search over scheduler configurations:

```yaml
grid_search:
  parameters:
    scheduler_type: [null, reduce_on_plateau]
    scheduler_patience: [50, 100, 200]
    scheduler_factor: [0.5, 0.3]
    # ... other parameters
```

### Monitoring and Logging

- Current learning rate is logged to MLflow as `learning_rate` at each epoch
- Scheduler parameters are logged to MLflow run parameters
- Terminal output includes current LR when scheduler is active

Example output:
```
Epoch   50/10000 | Loss: 0.123456 | MSE₀: 0.001234 | MSE_b: 0.000123 | MSE_f: 0.012345 | L²: 0.054321 | LR: 1.00e-03 | ...
```

### Why Monitor Training Loss?

The traditional approach is to monitor **validation loss** for schedulers. However, for PINNs:

1. **Training loss is the primary signal**: It directly reflects how well the model satisfies the physics constraints (IC, BC, PDE residual)
2. **Eval L² is computed per-epoch**: While we compute relative L² error every epoch, the scheduler uses training loss to maintain the traditional optimization paradigm
3. **Loss plateau → Physics plateau**: When training loss stops improving, the model has likely converged to a local minimum. Reducing LR helps it escape and refine further.

### Testing

A test configuration is provided:

```bash
python train_full.py --config config/test_scheduler.yaml
```

This runs a 50-epoch test with:
- ReduceLROnPlateau monitoring training loss
- Patience = 10 epochs
- Factor = 0.5 (50% reduction)
- You should see LR reductions in the terminal output

### Integration with Existing Code

The scheduler integrates seamlessly:

1. **`src/config_loader.py`**: Added `SchedulerConfig` dataclass
2. **`src/train/engine.py`**: 
   - Creates scheduler instance based on config
   - Steps scheduler every epoch
   - Logs current LR to MLflow
3. **`grid_search.py`**: Handles scheduler parameters in grid search
4. **`config/train.yaml`**: Default scheduler config (disabled by default)
5. **`config/grid_search.yaml`**: Grid search scheduler parameters

### Example Use Cases

#### 1. Basic Training with Scheduler
```yaml
# config/train.yaml
train:
  learning_rate: 0.001
  scheduler:
    type: reduce_on_plateau
    patience: 100
    factor: 0.5
```

Run: `python train_full.py --config config/train.yaml`

#### 2. Grid Search Over Scheduler Parameters
```yaml
# config/grid_search.yaml
grid_search:
  parameters:
    learning_rate: [0.001, 0.0001]
    scheduler_type: [null, reduce_on_plateau]
    scheduler_patience: [50, 100, 200]
```

Run: `python grid_search.py --yes`

#### 3. No Scheduler (Default)
```yaml
train:
  scheduler:
    type: null  # Disabled
```

### Expected Behavior

- **Without scheduler**: LR stays constant at initial value
- **With ReduceLROnPlateau**: 
  - LR reduces when loss plateaus
  - Typical reductions: 1e-3 → 5e-4 → 2.5e-4 → ... → 1e-7
  - Should see step-like improvements in loss after each reduction

### Notes

- Scheduler is **optional**: Set `type: null` to disable
- All scheduler parameters are logged to MLflow for reproducibility
- The implementation prioritizes **training loss** over eval L² for scheduler decisions (traditional approach)
- For PINN-specific scheduling (e.g., based on eval L²), you can modify `engine.py` line 653 to pass a different metric

## Files Modified

- `src/config_loader.py`: Added `SchedulerConfig` dataclass
- `src/train/engine.py`: Added scheduler creation, stepping, and logging
- `config/train.yaml`: Added scheduler configuration section
- `config/grid_search.yaml`: Added scheduler parameters for grid search
- `grid_search.py`: Added scheduler parameter handling
- `config/test_scheduler.yaml`: Test configuration with scheduler enabled

## Next Steps

1. Test with `config/test_scheduler.yaml` to verify functionality
2. Run grid search with scheduler variations to find optimal configuration
3. Compare runs with/without scheduler using MLflow UI


