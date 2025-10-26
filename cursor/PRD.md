# Project Requirements Document (PRD)
## Title
**Deep Learning Approximation of the Schrödinger Equation – Step 1**

---

## 1. Objective
Approximate the solution of the focusing nonlinear Schrödinger equation (NLSE) from §3.1.1 of the reference paper using a neural network.

\[
i\,h_t+\tfrac12 h_{xx}+|h|^2h=0,\qquad x\in[-5,5],\ t\in[0,\pi/2],
\]
with periodic boundary conditions and
\[
h(x,0)=2\,\mathrm{sech}(x).
\]

Goal: train a neural network \(h_\theta(x,t)=u_\theta+i\,v_\theta\) that approximates \(h(x,t)\) on this domain.

---

## 2. Ground-Truth Generation
- Generate **20 000 collocation samples** \((x,t)\) via **Latin Hypercube Sampling** on the domain.  
- Use a **split-step Fourier NLSE solver** (Python package *NLSE*, or equivalent) for the ground-truth solution.  
- Save the dataset as pairs \(((x,t),(u,v))\) where \(h(x,t)=u+iv\).  
- Keep the same sample counts as the paper:
  - \(N_f=20 000\) (collocation)
  - \(N_0=50\) (initial)
  - \(N_b=50\) (boundary)

---

## 3. Neural Network Model
- **Architecture:** 5 hidden layers × 100 neurons, fully connected (MLP).  
- **Activation:** tanh.  
- **Input:** (x, t). **Output:** (u, v).  
- **Layer names:** `layer_1` … `layer_5`, `output`.  
- **Framework:** PyTorch.  
- **Device:** CUDA with CPU fallback.

### Step 2 foresight — introspection hooks
Design so we can later probe neuron activations:
- Each layer can register **forward hooks** that capture intermediate tensors during forward passes.  
- Hooks store activations for later visualization without changing training code.

---

## 4. Loss Function (explicit formulation)

The total loss combines three mean-squared-error (MSE) terms:

\[
\mathcal{L}=\mathrm{MSE}_0+\mathrm{MSE}_b+\mathrm{MSE}_f.
\]

### 4.1 Initial Condition Term (MSE₀)
\[
\mathrm{MSE}_0=\frac{1}{N_0}\sum_{i=1}^{N_0}
\left|h_\theta(x_i,0)-2\,\mathrm{sech}(x_i)\right|^2.
\]

### 4.2 Boundary Condition Term (MSE_b)
\[
\mathrm{MSE}_b=\frac{1}{N_b}\sum_{i=1}^{N_b}
\Big(
|h_\theta(-5,t_i)-h_\theta(5,t_i)|^2+
|\partial_xh_\theta(-5,t_i)-\partial_xh_\theta(5,t_i)|^2
\Big).
\]

### 4.3 PDE Residual Term (MSE_f)
Residual:
\[
r_\theta(x,t)=
i\,\partial_t h_\theta(x,t)
+\tfrac12\,\partial_{xx}h_\theta(x,t)
+|h_\theta(x,t)|^2h_\theta(x,t),
\]
then
\[
\mathrm{MSE}_f=\frac{1}{N_f}\sum_{i=1}^{N_f}|r_\theta(x_i,t_i)|^2.
\]

### 4.4 Implementation Notes
- Compute derivatives with PyTorch autograd (`create_graph=True`).  
- Handle real and imag parts separately, then sum.  
- Back-propagate through all terms jointly.

---

## 5. Evaluation Metric
- **Relative L² error** versus ground-truth solver outputs.  
- Optional magnitude-only error to ignore global phase.

---

## 6. Training & Experiment Management
- Config-driven hyperparameters.  
- MLflow tracking (local `mlruns/`).  
- Plots: \|h(x,t)\| heatmap, time-slice overlays, loss curves, error vs time.  
- Deterministic seeds; float32 default, float64 optional.

---

## 7. Configuration System
- Plain YAML files + argparse loader (`--config path.yaml`).  
- Dataclass schema validation.  
- Environment-variable overrides allowed.

---

## 8. Deliverables (Step 1)
1. Config files (`config/`)  
2. Dataset generator using NLSE solver  
3. MLP model class (5×100 tanh)  
4. Physics-informed loss function (3 terms above)  
5. Training engine + evaluation script  
6. MLflow tracking + plots  
7. Sanity test run verifying batch prediction vs solver

---

## 9. Step 2 Preview
Later we’ll:
- Attach forward hooks to layers to capture hidden activations.  
- Analyze their temporal and spatial behavior.  
- Correlate internal representations with physical quantities.  
This architecture must allow such inspection without code changes.
