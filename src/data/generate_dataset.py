"""Dataset generation for NLSE physics-informed neural network.

Generates training data by:
1. Solving the NLSE on a fine grid using the split-step Fourier solver
2. Sampling collocation points via Latin Hypercube Sampling (LHS)
3. Sampling initial condition points (t=0)
4. Sampling boundary condition points (x=±5)
5. Interpolating solver outputs to sample locations
6. Saving as ((x,t), (u,v)) pairs where h = u + i*v
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import qmc

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config_loader import Config, DatasetConfig, SolverConfig
from src.solver.nlse_solver import solve_nlse_full_grid


def latin_hypercube_sampling(
    n_samples: int,
    x_range: Tuple[float, float],
    t_range: Tuple[float, float],
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Latin Hypercube samples in (x, t) domain.

    Args:
        n_samples: Number of samples to generate
        x_range: (x_min, x_max) tuple
        t_range: (t_min, t_max) tuple
        seed: Random seed for reproducibility

    Returns:
        Tuple of (x_samples, t_samples) arrays of shape (n_samples,)
    """
    # Create LHS sampler
    sampler = qmc.LatinHypercube(d=2, seed=seed)

    # Generate samples in [0, 1]^2
    samples_unit = sampler.random(n=n_samples)

    # Scale to domain
    x_min, x_max = x_range
    t_min, t_max = t_range

    x_samples = x_min + samples_unit[:, 0] * (x_max - x_min)
    t_samples = t_min + samples_unit[:, 1] * (t_max - t_min)

    return x_samples, t_samples


def sample_initial_condition(
    n_samples: int,
    x_range: Tuple[float, float],
    t_value: float = 0.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points on the initial condition boundary (t=0).

    Args:
        n_samples: Number of samples
        x_range: (x_min, x_max) tuple
        t_value: Time value (default 0.0)
        seed: Random seed

    Returns:
        Tuple of (x_samples, t_samples) arrays
    """
    rng = np.random.RandomState(seed)
    x_min, x_max = x_range

    x_samples = rng.uniform(x_min, x_max, size=n_samples)
    t_samples = np.full(n_samples, t_value)

    return x_samples, t_samples


def sample_boundary_conditions(
    n_times: int,
    x_boundaries: Tuple[float, float],
    t_range: Tuple[float, float],
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample paired boundary points at identical times for x = x_min and x = x_max.

    For periodic BC, we need h(x_min, t) = h(x_max, t) and
    h_x(x_min, t) = h_x(x_max, t) for the same set of times t. This function
    samples N_b time values and returns 2*N_b boundary samples: one at each
    boundary for every sampled time.

    Args:
        n_times: Number of distinct time values (N_b)
        x_boundaries: (x_min, x_max) tuple
        t_range: (t_min, t_max) tuple
        seed: Random seed

    Returns:
        Tuple of (x_samples, t_samples) arrays of length 2*n_times, ordered as
        all left boundary points first, followed by all right boundary points,
        with identical time ordering across both halves.
    """
    rng = np.random.RandomState(seed)
    x_min, x_max = x_boundaries
    t_min, t_max = t_range

    # Sample N_b time values uniformly
    t_samples = rng.uniform(t_min, t_max, size=n_times)

    # Build paired boundary coordinates with identical time ordering
    x_left = np.full(n_times, x_min)
    x_right = np.full(n_times, x_max)

    x_all = np.concatenate([x_left, x_right])
    t_all = np.concatenate([t_samples, t_samples])

    return x_all, t_all


def solve_on_grid(
    solver_config: SolverConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve NLSE on a fine grid.

    Args:
        solver_config: Solver configuration

    Returns:
        Tuple of (x_grid, t_grid, h_solution)
        - x_grid: 1D spatial grid (nx,)
        - t_grid: 1D temporal grid (nt,)
        - h_solution: 2D complex solution (nt, nx)
    """
    print(f"  Solving NLSE on {solver_config.nx}×{solver_config.nt} grid...")

    x_grid, t_grid, h_solution = solve_nlse_full_grid(
        x_min=solver_config.x_min,
        x_max=solver_config.x_max,
        t_min=solver_config.t_min,
        t_max=solver_config.t_max,
        nx=solver_config.nx,
        nt=solver_config.nt,
        alpha=solver_config.alpha,
    )

    print(f"  ✓ Solution computed: shape = {h_solution.shape}")
    return x_grid, t_grid, h_solution


def interpolate_solution(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    h_solution: np.ndarray,
    x_samples: np.ndarray,
    t_samples: np.ndarray,
) -> np.ndarray:
    """Interpolate solution to sample points.

    Args:
        x_grid: Grid x coordinates (nx,)
        t_grid: Grid t coordinates (nt,)
        h_solution: Complex solution on grid (nt, nx)
        x_samples: Sample x coordinates (n_samples,)
        t_samples: Sample t coordinates (n_samples,)

    Returns:
        Interpolated complex values (n_samples,)
    """
    # Create interpolator for real and imaginary parts
    # Note: RegularGridInterpolator expects (t, x) order
    real_interp = RegularGridInterpolator(
        (t_grid, x_grid),
        h_solution.real,
        method="cubic",
        bounds_error=False,
        fill_value=None,
    )

    imag_interp = RegularGridInterpolator(
        (t_grid, x_grid),
        h_solution.imag,
        method="cubic",
        bounds_error=False,
        fill_value=None,
    )

    # Interpolate at sample points
    points = np.column_stack([t_samples, x_samples])
    u_samples = real_interp(points)
    v_samples = imag_interp(points)

    h_samples = u_samples + 1j * v_samples

    return h_samples


def generate_dataset(
    config: Config, verbose: bool = True
) -> Dict[str, np.ndarray]:
    """Generate complete dataset for PINN training.

    Args:
        config: Full configuration object
        verbose: Print progress information

    Returns:
        Dictionary with keys:
            'x_f', 't_f': Collocation points (n_collocation,)
            'x_0', 't_0': Initial condition points (n_initial,)
            'x_b', 't_b': Boundary condition points (n_boundary,)
            'u_f', 'v_f': Solution at collocation (n_collocation,)
            'u_0', 'v_0': Solution at initial (n_initial,)
            'u_b', 'v_b': Solution at boundary (n_boundary,)
    """
    if verbose:
        print("=" * 70)
        print("Generating NLSE Dataset")
        print("=" * 70)

    dataset_cfg = config.dataset
    solver_cfg = config.solver

    # 1. Solve on fine grid
    if verbose:
        print("\n1. Computing ground truth solution:")

    x_grid, t_grid, h_solution = solve_on_grid(solver_cfg)

    # 2. Generate collocation samples (interior domain)
    if verbose:
        print(f"\n2. Sampling collocation points (N_f={dataset_cfg.n_collocation}):")
        print(f"   Method: {dataset_cfg.sampling_method.upper()}")

    x_f, t_f = latin_hypercube_sampling(
        n_samples=dataset_cfg.n_collocation,
        x_range=(solver_cfg.x_min, solver_cfg.x_max),
        t_range=(solver_cfg.t_min, solver_cfg.t_max),
        seed=dataset_cfg.seed,
    )

    if verbose:
        print(f"   ✓ Generated {len(x_f)} collocation points")

    # 3. Generate initial condition samples
    if verbose:
        print(f"\n3. Sampling initial condition (N_0={dataset_cfg.n_initial}):")

    x_0, t_0 = sample_initial_condition(
        n_samples=dataset_cfg.n_initial,
        x_range=(solver_cfg.x_min, solver_cfg.x_max),
        t_value=solver_cfg.t_min,
        seed=dataset_cfg.seed + 1,
    )

    if verbose:
        print(f"   ✓ Generated {len(x_0)} initial condition points")

    # 4. Generate boundary condition samples
    if verbose:
        print(f"\n4. Sampling boundary conditions (N_b={dataset_cfg.n_boundary}):")

    x_b, t_b = sample_boundary_conditions(
        n_times=dataset_cfg.n_boundary,
        x_boundaries=(solver_cfg.x_min, solver_cfg.x_max),
        t_range=(solver_cfg.t_min, solver_cfg.t_max),
        seed=dataset_cfg.seed + 2,
    )

    if verbose:
        print(f"   ✓ Generated {len(x_b)} boundary condition points")

    # 5. Interpolate solution to all sample points
    if verbose:
        print("\n5. Interpolating solution to sample points:")

    h_f = interpolate_solution(x_grid, t_grid, h_solution, x_f, t_f)
    h_0 = interpolate_solution(x_grid, t_grid, h_solution, x_0, t_0)
    h_b = interpolate_solution(x_grid, t_grid, h_solution, x_b, t_b)

    if verbose:
        print(f"   ✓ Interpolated collocation points")
        print(f"   ✓ Interpolated initial condition points")
        print(f"   ✓ Interpolated boundary condition points")

    # 6. Extract real and imaginary parts
    dataset = {
        # Collocation
        "x_f": x_f.astype(np.float32),
        "t_f": t_f.astype(np.float32),
        "u_f": h_f.real.astype(np.float32),
        "v_f": h_f.imag.astype(np.float32),
        # Initial condition
        "x_0": x_0.astype(np.float32),
        "t_0": t_0.astype(np.float32),
        "u_0": h_0.real.astype(np.float32),
        "v_0": h_0.imag.astype(np.float32),
        # Boundary condition
        "x_b": x_b.astype(np.float32),
        "t_b": t_b.astype(np.float32),
        "u_b": h_b.real.astype(np.float32),
        "v_b": h_b.imag.astype(np.float32),
    }

    if verbose:
        print("\n6. Dataset summary:")
        print(f"   Collocation:  {len(x_f)} points")
        print(f"   Initial:      {len(x_0)} points")
        print(f"   Boundary:     {len(x_b)} points")
        print(f"   Total:        {len(x_f) + len(x_0) + len(x_b)} points")

    return dataset


def save_dataset(
    dataset: Dict[str, np.ndarray], save_path: str, verbose: bool = True
) -> None:
    """Save dataset to .npz file.

    Args:
        dataset: Dictionary of arrays
        save_path: Path to save file
        verbose: Print confirmation
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(save_path, **dataset)

    if verbose:
        file_size = save_path.stat().st_size / 1024 / 1024  # MB
        print(f"\n✓ Dataset saved: {save_path}")
        print(f"  File size: {file_size:.2f} MB")


def load_dataset(load_path: str) -> Dict[str, np.ndarray]:
    """Load dataset from .npz file.

    Args:
        load_path: Path to dataset file

    Returns:
        Dictionary of arrays
    """
    data = np.load(load_path)
    return {key: data[key] for key in data.files}


if __name__ == "__main__":
    """Generate dataset with default configuration."""
    import argparse

    from src.config_loader import load_config

    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate NLSE dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config/dataset.yaml",
        help="Config file path",
    )
    args = parser.parse_args()

    # Load configuration (loads all three configs: solver, dataset, train)
    config = load_config(args.config)

    # Generate dataset
    dataset = generate_dataset(config, verbose=True)

    # Save dataset
    save_dataset(dataset, config.dataset.save_path, verbose=True)

    print("\n" + "=" * 70)
    print("✓ Dataset generation completed successfully!")
    print("=" * 70)

