# NCC Analysis for Schrödinger PINN

## Overview

This module analyzes the hidden-layer smoothness of trained Schrödinger PINN models using the **Nearest Class Center (NCC)** metric from neural collapse theory.

## What is NCC?

The NCC metric measures how well hidden-layer representations cluster according to their ground-truth amplitude classes. Lower mismatch rates indicate smoother, more organized representations.

## Usage

### Basic Usage

```bash
python src/evaluate/ncc_analysis.py --run_id <MLFLOW_RUN_ID> --dataset data/processed/dataset.npz
```

### With Custom Parameters

```bash
python src/evaluate/ncc_analysis.py \
    --run_id 0cb7cac0b09043408ae5444eb7d19aa3 \
    --dataset data/processed/dataset_eval.npz \
    --bins 200 \
    --amp_range 0 4 \
    --batch_size 2048
```

### Command-Line Arguments

- `--run_id` (required): MLflow run ID of the trained model to analyze
- `--dataset`: Path to dataset .npz file (default: `data/processed/dataset.npz`)
- `--bins`: Number of bins for amplitude classes (default: 200)
- `--amp_range`: Min and max amplitude for binning (default: 0 4)
- `--batch_size`: Batch size for forward passes (default: 1024)

## Finding Run IDs

You have several ways to find the run ID:

1. **From output folders:**
   ```bash
   ls outputs/plots/
   ls outputs/evaluation/
   ```

2. **From grid search results:**
   ```bash
   # View the CSV file
   cat outputs/grid_search_results.csv
   ```

3. **Using the utility script:**
   ```bash
   python get_run_params.py <run_id>
   ```

4. **From `run_info.json` files:**
   ```bash
   cat outputs/plots/<run_id>/run_info.json
   ```

## Outputs

The analysis creates a folder `outputs/plots/ncc_<run_id>/` containing:

### 1. **ncc_layer_smoothness.png**
Bar chart showing mismatch rate per layer. Lower is better.

### 2. **ncc_confusion_all_layers.png**
Confusion matrices for all 5 layers in one image (1×5 grid).
Shows how well each layer's representations cluster by amplitude class.

### 3. **ncc_distance_hist_all_layers.png**
Overlaid histograms of distances to nearest centers for all layers.
Tighter distributions indicate more compact clusters.

### 4. **ncc_metrics.json**
JSON file with mismatch rates for each layer:
```json
{
  "layer_1": 0.234,
  "layer_2": 0.189,
  "layer_3": 0.156,
  "layer_4": 0.134,
  "layer_5": 0.112
}
```

## Example Workflow

### 1. Train a model and get its run ID
```bash
python train_full.py --config config/train.yaml
# Output: MLflow run ID: 0cb7cac0b09043408ae5444eb7d19aa3
```

### 2. Run NCC analysis on training data
```bash
python src/evaluate/ncc_analysis.py \
    --run_id 0cb7cac0b09043408ae5444eb7d19aa3 \
    --dataset data/processed/dataset.npz
```

### 3. Run NCC analysis on evaluation data
```bash
python src/evaluate/ncc_analysis.py \
    --run_id 0cb7cac0b09043408ae5444eb7d19aa3 \
    --dataset data/processed/dataset_eval.npz
```

### 4. Compare results
Check `outputs/plots/ncc_<run_id>/` for both runs to compare training vs. evaluation trends.

## Implementation Details

### GPU Acceleration
The NCC computation is accelerated with PyTorch on GPU when available. This provides significant speedups for large datasets:
- CPU: ~5-10 seconds per layer
- GPU: ~0.5-1 seconds per layer

### Class Labels
Amplitude values |h| = √(u² + v²) are binned into classes:
- Default: 200 bins over range [0, 4]
- Adjustable via `--bins` and `--amp_range`

### Layer Activation Collection
Forward hooks capture activations from:
- `layer_1`, `layer_2`, `layer_3`, `layer_4`, `layer_5`

The hooks capture outputs after the activation function (tanh).

## Files

- `src/evaluate/ncc_analysis.py`: Main analysis script
- `src/utils/ncc.py`: NCC metric computation (GPU-accelerated)
- `src/utils/ncc_plotting.py`: Plotting functions

## Requirements

- PyTorch
- NumPy
- scikit-learn (for confusion_matrix)
- Matplotlib
- Seaborn
- MLflow

