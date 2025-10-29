"""Generate evaluation dataset for the Schrödinger equation solver.

This script creates a smaller, separate evaluation dataset to be used only for
periodic evaluation during training. The sizes are:
- F (collocation): 5000 points
- IC (initial): 15 points
- BC (boundary): 15 time points

Usage:
    python -m src.data.generate_eval_dataset
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config_loader import load_config
from src.solver.nlse_solver import solve_nlse_full_grid


def generate_eval_dataset(
    config_path: str = "config/dataset.yaml",
    output_path: str = "data/processed/dataset_eval.npz",
):
    """Generate evaluation dataset using Latin Hypercube Sampling.

    Args:
        config_path: Path to dataset configuration file
        output_path: Path to save the evaluation dataset
    """
    print("=" * 70)
    print("Generating Evaluation Dataset")
    print("=" * 70)

    # Load configurations
    config = load_config(config_path)

    # Evaluation dataset sizes
    n_collocation = 5000  # reduced from 20000
    n_initial = 15  # reduced from 50
    n_boundary = 15  # reduced from 50 (times)

    print(f"\nConfiguration:")
    print(f"  Spatial domain: [{config.solver.x_min}, {config.solver.x_max}]")
    print(f"  Temporal domain: [{config.solver.t_min}, {config.solver.t_max}]")
    print(f"  Grid: {config.solver.nx} × {config.solver.nt}")
    print(f"  Collocation points: {n_collocation}")
    print(f"  Initial points: {n_initial}")
    print(f"  Boundary time points: {n_boundary}")

    # Solve NLSE on fine grid FIRST to get ground truth
    print("\nSolving NLSE on fine grid...")
    print(f"  Grid size: {config.solver.nx} × {config.solver.nt}")
    x_grid, t_grid, h_solution = solve_nlse_full_grid(
        x_min=config.solver.x_min,
        x_max=config.solver.x_max,
        t_min=config.solver.t_min,
        t_max=config.solver.t_max,
        nx=config.solver.nx,
        nt=config.solver.nt,
        alpha=config.solver.alpha,
    )
    print(f"  ✓ Solution computed, shape: {h_solution.shape}")

    # Set random seed
    np.random.seed(config.dataset.seed)

    # Generate collocation points using LHS
    print("\nGenerating collocation points (LHS)...")
    sampler = qmc.LatinHypercube(d=2, seed=config.dataset.seed)
    samples = sampler.random(n=n_collocation)

    # Scale to domain
    x_f = qmc.scale(
        samples[:, 0:1],
        l_bounds=[config.solver.x_min],
        u_bounds=[config.solver.x_max],
    ).flatten()

    t_f = qmc.scale(
        samples[:, 1:2],
        l_bounds=[config.solver.t_min],
        u_bounds=[config.solver.t_max],
    ).flatten()

    # Generate initial condition points (t=0) with Gaussian bias
    # IC samples are drawn from a Gaussian near x=0 to emphasize the region
    # where |h(x,0)| is large. This improves phase anchoring at t=0.
    print("Generating initial condition points...")
    print(f"  Gaussian bias: σ={config.dataset.ic_sigma}, mix={config.dataset.ic_mix*100:.0f}% Gaussian")
    
    rng = np.random.RandomState(config.dataset.seed + 1)
    n_gaussian = int(n_initial * config.dataset.ic_mix)
    n_uniform = n_initial - n_gaussian
    
    # Gaussian samples centered at x=0
    x_gaussian = rng.normal(loc=0.0, scale=config.dataset.ic_sigma, size=n_gaussian)
    x_gaussian = np.clip(x_gaussian, config.solver.x_min, config.solver.x_max)
    
    # Uniform samples across full domain
    x_uniform = rng.uniform(config.solver.x_min, config.solver.x_max, size=n_uniform)
    
    # Combine and shuffle
    x_0 = np.concatenate([x_gaussian, x_uniform])
    rng.shuffle(x_0)
    t_0 = np.zeros(n_initial)

    # Generate boundary condition points (t>0)
    print("Generating boundary condition points...")
    t_b_samples = np.linspace(
        config.solver.t_min, config.solver.t_max, n_boundary
    )
    x_b_left = np.full(n_boundary, config.solver.x_min)
    t_b_left = t_b_samples
    x_b_right = np.full(n_boundary, config.solver.x_max)
    t_b_right = t_b_samples
    
    print(f"  ✓ {n_boundary} left points, {n_boundary} right points generated")

    # Combine all points (for interpolation)
    x_all = np.concatenate([x_f, x_0, x_b_left, x_b_right])
    t_all = np.concatenate([t_f, t_0, t_b_left, t_b_right])
    
    # Interpolate solution
    print("\nInterpolating to sampled points...")

    # Helper for interpolation
    def interpolate_solution(
        x_query: np.ndarray, 
        t_query: np.ndarray, 
        x_grid: np.ndarray, 
        t_grid: np.ndarray, 
        h_solution: np.ndarray
    ):
        """Interpolate from fine grid to query points."""
        t_idx = np.searchsorted(t_grid, t_query) - 1
        x_idx = np.searchsorted(x_grid, x_query) - 1
        
        # Clip indices to be within bounds
        t_idx = np.clip(t_idx, 0, h_solution.shape[0] - 2)
        x_idx = np.clip(x_idx, 0, h_solution.shape[1] - 2)
        
        # Bilinear interpolation
        t1 = t_grid[t_idx]
        t2 = t_grid[t_idx + 1]
        x1 = x_grid[x_idx]
        x2 = x_grid[x_idx + 1]

        f_q11 = h_solution[t_idx, x_idx]
        f_q21 = h_solution[t_idx, x_idx + 1]
        f_q12 = h_solution[t_idx + 1, x_idx]
        f_q22 = h_solution[t_idx + 1, x_idx + 1]

        term1 = f_q11 * (x2 - x_query) * (t2 - t_query)
        term2 = f_q21 * (x_query - x1) * (t2 - t_query)
        term3 = f_q12 * (x2 - x_query) * (t_query - t1)
        term4 = f_q22 * (x_query - x1) * (t_query - t1)

        h_interp = (term1 + term2 + term3 + term4) / ((x2 - x1) * (t2 - t1))
        return h_interp

    # Interpolate for each set of points
    h_f = interpolate_solution(x_f, t_f, x_grid, t_grid, h_solution)
    h_0 = interpolate_solution(x_0, t_0, x_grid, t_grid, h_solution)
    h_b_left = interpolate_solution(x_b_left, t_b_left, x_grid, t_grid, h_solution)
    h_b_right = interpolate_solution(x_b_right, t_b_right, x_grid, t_grid, h_solution)
    
    # Extract real and imag parts (for saving)
    u_f = h_f.real.astype(np.float32)
    v_f = h_f.imag.astype(np.float32)

    # Interpolate for initial points
    u_0 = h_0.real.astype(np.float32)
    v_0 = h_0.imag.astype(np.float32)

    # Interpolate for boundary points
    u_b_left = h_b_left.real.astype(np.float32)
    v_b_left = h_b_left.imag.astype(np.float32)

    u_b_right = h_b_right.real.astype(np.float32)
    v_b_right = h_b_right.imag.astype(np.float32)

    # Save dataset
    print(f"\nSaving evaluation dataset...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        # Collocation
        x_f=x_f.astype(np.float32),
        t_f=t_f.astype(np.float32),
        u_f=u_f,
        v_f=v_f,
        # Initial condition
        x_0=x_0.astype(np.float32),
        t_0=t_0.astype(np.float32),
        u_0=u_0,
        v_0=v_0,
        # Boundary condition (split)
        x_b_left=x_b_left.astype(np.float32),
        t_b_left=t_b_left.astype(np.float32),
        x_b_right=x_b_right.astype(np.float32),
        t_b_right=t_b_right.astype(np.float32),
    )

    print(f"\n✓ Evaluation dataset saved to: {output_path}")
    print(f"\nDataset Summary:")
    print(f"  Collocation: {len(x_f)} points")
    print(f"  Initial: {len(x_0)} points")
    print(f"  Boundary: {len(x_b_left) + len(x_b_right)} points")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    return output_path


if __name__ == "__main__":
    generate_eval_dataset()

