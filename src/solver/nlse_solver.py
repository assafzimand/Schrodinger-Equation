"""Split-step Fourier solver for the focusing nonlinear Schrödinger
equation.

This module implements a numerical solver for the NLSE from the
reference paper: i*h_t + 0.5*h_xx + |h|^2*h = 0

with periodic boundary conditions on x ∈ [-5, 5] and t ∈ [0, π/2].
Initial condition: h(x, 0) = 2*sech(x)
"""

import numpy as np
from typing import Callable, Optional, Tuple


def sech(x: np.ndarray) -> np.ndarray:
    """Hyperbolic secant function.

    Args:
        x: Input array

    Returns:
        sech(x) = 1/cosh(x)
    """
    return 1.0 / np.cosh(x)


def initial_condition(x: np.ndarray) -> np.ndarray:
    """Initial condition for the NLSE: h(x, 0) = 2*sech(x).

    Args:
        x: Spatial grid points

    Returns:
        Complex array of initial values
    """
    return 2.0 * sech(x)


def solve_nlse(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    h0: Optional[np.ndarray] = None,
    alpha: float = 1.0,
) -> np.ndarray:
    """Solve the focusing NLSE using split-step Fourier method.

    Solves: i*h_t + 0.5*h_xx + alpha*|h|^2*h = 0

    The equation is split into:
    - Linear part: i*h_t + 0.5*h_xx = 0 (solved in Fourier space)
    - Nonlinear part: i*h_t + alpha*|h|^2*h = 0 (solved analytically)

    Args:
        x_grid: 1D array of spatial points (periodic domain)
        t_grid: 1D array of temporal points
        h0: Initial condition h(x, 0). If None, uses 2*sech(x)
        alpha: Nonlinearity coefficient (default 1.0)

    Returns:
        Complex array of shape (len(t_grid), len(x_grid)) containing h(x, t)

    Notes:
        - Uses periodic boundary conditions in space
        - Second-order split-step method (Strang splitting)
        - FFT-based spatial derivatives
    """
    # Grid parameters
    nx = len(x_grid)
    nt = len(t_grid)
    dx = x_grid[1] - x_grid[0]
    dt = t_grid[1] - t_grid[0]

    # Set initial condition
    if h0 is None:
        h0 = initial_condition(x_grid)
    else:
        h0 = h0.copy()

    # Fourier space wave numbers (for periodic domain)
    k = 2.0 * np.pi * np.fft.fftfreq(nx, dx)

    # Linear operator in Fourier space: exp(-i * 0.5 * k^2 * dt)
    # From i*h_t + 0.5*h_xx = 0 -> h_t = i*0.5*h_xx
    # In Fourier space: h_t = -i*0.5*k^2*h_k
    linear_operator_full = np.exp(-1j * 0.5 * k**2 * dt)

    # Storage for solution
    h_solution = np.zeros((nt, nx), dtype=complex)
    h_solution[0, :] = h0

    # Current field
    h_current = h0.copy()

    # Time stepping with Strang splitting (second-order accurate)
    for n in range(1, nt):
        # Step 1: Half-step nonlinear (N/2)
        # From i*h_t + alpha*|h|^2*h = 0
        # -> h_t = i*alpha*|h|^2*h -> h(t+dt) = h(t)*exp(+i*alpha*|h|^2*dt)
        nonlinear_phase = (
            +1j * alpha * np.abs(h_current) ** 2 * dt / 2.0
        )
        h_current = h_current * np.exp(nonlinear_phase)

        # Step 2: Full-step linear in Fourier space (L)
        h_fft = np.fft.fft(h_current)
        h_fft = h_fft * linear_operator_full
        h_current = np.fft.ifft(h_fft)

        # Step 3: Half-step nonlinear (N/2)
        nonlinear_phase = +1j * alpha * np.abs(h_current) ** 2 * dt / 2.0
        h_current = h_current * np.exp(nonlinear_phase)

        # Store result
        h_solution[n, :] = h_current

    return h_solution


def solve_nlse_full_grid(
    x_min: float = -5.0,
    x_max: float = 5.0,
    t_min: float = 0.0,
    t_max: float = np.pi / 2,
    nx: int = 256,
    nt: int = 100,
    alpha: float = 1.0,
    h0_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience function to solve NLSE with automatic grid generation.

    Args:
        x_min: Minimum spatial coordinate
        x_max: Maximum spatial coordinate
        t_min: Minimum time
        t_max: Maximum time
        nx: Number of spatial grid points
        nt: Number of temporal grid points
        alpha: Nonlinearity coefficient
        h0_func: Function to compute initial condition. If None, uses 2*sech(x)

    Returns:
        Tuple of (x_grid, t_grid, h_solution)
        - x_grid: 1D spatial grid
        - t_grid: 1D temporal grid
        - h_solution: 2D complex array of shape (nt, nx)
    """
    # Create grids (note: periodic, so don't include endpoint)
    x_grid = np.linspace(x_min, x_max, nx, endpoint=False)
    t_grid = np.linspace(t_min, t_max, nt)

    # Initial condition
    if h0_func is None:
        h0 = initial_condition(x_grid)
    else:
        h0 = h0_func(x_grid)

    # Solve
    h_solution = solve_nlse(x_grid, t_grid, h0, alpha=alpha)

    return x_grid, t_grid, h_solution


def compute_energy(h: np.ndarray, dx: float) -> float:
    """Compute the L2 norm (energy) of the solution.

    Args:
        h: Complex field values
        dx: Spatial grid spacing

    Returns:
        Energy integral: ∫|h|^2 dx
    """
    return np.sum(np.abs(h) ** 2) * dx


def compute_mass(h: np.ndarray, dx: float) -> float:
    """Compute the mass (L1 norm) of the solution.

    Args:
        h: Complex field values
        dx: Spatial grid spacing

    Returns:
        Mass integral: ∫|h| dx
    """
    return np.sum(np.abs(h)) * dx


def verify_periodic_bc(
    h_solution: np.ndarray, x_grid: np.ndarray, tol: float = 0.1
) -> Tuple[bool, float]:
    """Verify that periodic boundary conditions are satisfied.

    The FFT-based solver automatically enforces periodic BCs. We check
    the maximum relative difference at boundaries across all time steps.

    Args:
        h_solution: Solution array of shape (nt, nx)
        x_grid: Spatial grid
        tol: Relative tolerance for boundary mismatch (default 0.1 = 10%)

    Returns:
        Tuple of (is_periodic, max_rel_diff)
        - is_periodic: True if max relative difference < tolerance
        - max_rel_diff: Maximum relative difference at boundaries

    Notes:
        For localized initial conditions like sech(x), boundaries are
        near zero, so small absolute differences are acceptable.
    """
    max_rel_diff = 0.0

    for n in range(h_solution.shape[0]):
        left_val = h_solution[n, 0]
        right_val = h_solution[n, -1]

        # Compute relative difference
        abs_diff = np.abs(left_val - right_val)
        avg_magnitude = (np.abs(left_val) + np.abs(right_val)) / 2.0

        if avg_magnitude > 1e-10:  # Avoid division by near-zero
            rel_diff = abs_diff / avg_magnitude
        else:
            rel_diff = abs_diff  # Both values are essentially zero

        max_rel_diff = max(max_rel_diff, rel_diff)

    return max_rel_diff < tol, max_rel_diff


if __name__ == "__main__":
    """Quick verification that solver runs without errors."""
    print("Testing NLSE solver...")
    x_grid, t_grid, h_solution = solve_nlse_full_grid(nx=128, nt=50)
    print(f"✓ Solution computed: shape = {h_solution.shape}")
    dx = x_grid[1] - x_grid[0]
    energy = compute_energy(h_solution[0, :], dx)
    print(f"✓ Initial energy: {energy:.6f}")
    print("Solver module OK. Run tests/test_solver.py for full validation.")
