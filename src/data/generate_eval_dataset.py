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

    # Generate boundary condition points (x = x_min and x = x_max)
    print("Generating boundary condition points...")
    t_b_samples = np.linspace(
        config.solver.t_min, config.solver.t_max, n_boundary, endpoint=False
    )

    # For each time point, sample both boundaries
    x_b = np.concatenate(
        [
            np.full(n_boundary, config.solver.x_min),
            np.full(n_boundary, config.solver.x_max),
        ]
    )
    t_b = np.concatenate([t_b_samples, t_b_samples])

    # Solve NLSE on fine grid
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

    print(f"  Solution shape: {h_solution.shape}")

    # Interpolate to sampled points
    print("\nInterpolating to sampled points...")

    def interpolate_solution(x_query, t_query):
        """Bilinear interpolation of the solution.
        
        Note: h_solution has shape (nt, nx) i.e. h_solution[t, x]
        """
        # Find indices
        x_idx = np.searchsorted(x_grid, x_query) - 1
        t_idx = np.searchsorted(t_grid, t_query) - 1

        # Clip indices
        x_idx = np.clip(x_idx, 0, len(x_grid) - 2)
        t_idx = np.clip(t_idx, 0, len(t_grid) - 2)

        # Get weights
        x_weight = (x_query - x_grid[x_idx]) / (x_grid[x_idx + 1] - x_grid[x_idx])
        t_weight = (t_query - t_grid[t_idx]) / (t_grid[t_idx + 1] - t_grid[t_idx])

        # Handle periodic boundary for x
        x_idx_p1 = (x_idx + 1) % len(x_grid)

        # Bilinear interpolation (note: h_solution is indexed as [t, x])
        h_00 = h_solution[t_idx, x_idx]
        h_10 = h_solution[t_idx, x_idx_p1]
        h_01 = h_solution[t_idx + 1, x_idx]
        h_11 = h_solution[t_idx + 1, x_idx_p1]

        h_interp = (
            (1 - x_weight) * (1 - t_weight) * h_00
            + x_weight * (1 - t_weight) * h_10
            + (1 - x_weight) * t_weight * h_01
            + x_weight * t_weight * h_11
        )

        return h_interp

    # Interpolate for collocation points
    h_f = interpolate_solution(x_f, t_f)
    u_f = h_f.real.astype(np.float32)
    v_f = h_f.imag.astype(np.float32)

    # Interpolate for initial points
    h_0 = interpolate_solution(x_0, t_0)
    u_0 = h_0.real.astype(np.float32)
    v_0 = h_0.imag.astype(np.float32)

    # Interpolate for boundary points
    h_b = interpolate_solution(x_b, t_b)
    u_b = h_b.real.astype(np.float32)
    v_b = h_b.imag.astype(np.float32)

    # Save dataset
    print(f"\nSaving evaluation dataset...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        x_f=x_f.astype(np.float32),
        t_f=t_f.astype(np.float32),
        u_f=u_f,
        v_f=v_f,
        x_0=x_0.astype(np.float32),
        t_0=t_0.astype(np.float32),
        u_0=u_0,
        v_0=v_0,
        x_b=x_b.astype(np.float32),
        t_b=t_b.astype(np.float32),
        u_b=u_b,
        v_b=v_b,
    )

    print(f"  ✓ Saved to: {output_path}")
    print(f"\nDataset Summary:")
    print(f"  Collocation: {len(x_f)} points")
    print(f"  Initial: {len(x_0)} points")
    print(f"  Boundary: {len(x_b)} points")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    return output_path


if __name__ == "__main__":
    generate_eval_dataset()

