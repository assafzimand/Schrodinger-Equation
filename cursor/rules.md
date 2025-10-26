# Repository Rules and Standards

## 1. Coding Style
- Python ≥ 3.10  
- PyTorch | NumPy | Matplotlib | MLflow  
- Format with Black + isort + flake8  
- Type annotations required  
- Google-style docstrings

## 2. Naming
- Layers: `layer_1`…`layer_5`, `output`  
- Config files: lowercase hyphenated  
- Functions snake_case, Classes PascalCase  
- Descriptive variable names (`u_true`, `loss_total`)

## 3. Reproducibility
- Global seeds (Python, NumPy, Torch)  
- Log seed & device in MLflow  
- Optional deterministic CuDNN

## 4. Experiment Tracking
- Local MLflow (`mlruns/`)  
- Log: config snapshot, loss components, L² error, plots  
- Optional tags (commit hash, date, author)

## 5. Testing
- Basic pytest suite:  
  - dataset shapes  
  - model forward dims  
  - loss outputs non-NaN

## 6. GPU/CPU Handling
- Auto-detect CUDA  
- All tensors on configured device

## 7. Plots & Outputs
- Save to `outputs/plots/`  
- Use Matplotlib/Seaborn (`seaborn-whitegrid`)  
- Label axes & legends

## 8. Future-Proofing
- Access layers via `named_children()`  
- Allow forward hook registration for Step 2 introspection
