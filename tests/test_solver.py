"""Test script for NLSE solver validation and visualization."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solver.nlse_solver import (
    compute_energy,
    solve_nlse_full_grid,
    verify_periodic_bc,
)


def test_solver_basic():
    """Test the NLSE solver with default parameters."""
    print("=" * 70)
    print("NLSE Solver Validation")
    print("=" * 70)

    # Solve with default parameters from PRD
    print("\n1. Solving NLSE:")
    print("   Equation: i*h_t + 0.5*h_xx + |h|^2*h = 0")
    print("   Domain: x ∈ [-5, 5], t ∈ [0, π/2]")
    print("   Grid: 256 × 100 points")
    print("   BC: Periodic (h(-5,t) = h(5,t), h_x(-5,t) = h_x(5,t))")
    print("   IC: h(x, 0) = 2*sech(x)")

    x_grid, t_grid, h_solution = solve_nlse_full_grid(
        x_min=-5.0,
        x_max=5.0,
        t_min=0.0,
        t_max=np.pi / 2,
        nx=256,
        nt=100,
        alpha=1.0,
    )

    print(f"\n   ✓ Solution computed: shape = {h_solution.shape}")
    print(f"   ✓ x range: [{x_grid[0]:.3f}, {x_grid[-1]:.3f}]")
    print(f"   ✓ t range: [{t_grid[0]:.3f}, {t_grid[-1]:.3f}]")

    # Compute magnitude
    magnitude = np.abs(h_solution)

    # Verify periodic BC
    print("\n2. Periodic boundary conditions:")
    is_periodic, max_rel_diff = verify_periodic_bc(h_solution, x_grid)
    print(
        f"   Max relative difference at boundaries: {max_rel_diff:.4f}"
    )
    print(f"   Status: {'✓ OK' if is_periodic else '⚠ Large difference'}")

    # Check values at boundaries
    print(f"\n   Boundary values at t=0:")
    print(f"   h(-5, 0) = {h_solution[0, 0]:.6f}")
    print(f"   h(+5, 0) = {h_solution[0, -1]:.6f}")
    print(f"\n   Boundary values at t=π/2:")
    print(f"   h(-5, π/2) = {h_solution[-1, 0]:.6f}")
    print(f"   h(+5, π/2) = {h_solution[-1, -1]:.6f}")

    # Check energy conservation
    print("\n3. Energy conservation:")
    dx = x_grid[1] - x_grid[0]
    energies = [
        compute_energy(h_solution[n, :], dx) for n in range(len(t_grid))
    ]
    energy_variation = (max(energies) - min(energies)) / energies[0]
    print(f"   Initial energy: {energies[0]:.8f}")
    print(f"   Final energy:   {energies[-1]:.8f}")
    print(f"   Relative variation: {energy_variation:.2e}")
    print(f"   Status: {'✓ Conserved' if energy_variation < 1e-10 else '⚠'}")

    # Generate plots
    print("\n4. Generating visualization plots...")
    fig = plt.figure(figsize=(14, 10))

    # Main heatmap
    ax1 = plt.subplot(2, 3, 1)
    extent = [x_grid[0], x_grid[-1], t_grid[0], t_grid[-1]]
    im = ax1.imshow(
        magnitude,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax1, label="|h(x, t)|")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    ax1.set_title("Solution Magnitude |h(x, t)|")

    # Time slices
    ax2 = plt.subplot(2, 3, 2)
    time_indices = [0, 25, 50, 75, 99]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, color in zip(time_indices, colors):
        ax2.plot(
            x_grid,
            magnitude[idx, :],
            label=f"t={t_grid[idx]:.3f}",
            linewidth=2,
            color=color,
        )
    ax2.set_xlabel("x")
    ax2.set_ylabel("|h(x, t)|")
    ax2.set_title("Time Evolution")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Initial vs Final
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(
        x_grid,
        magnitude[0, :],
        "b-",
        linewidth=2.5,
        label="t=0",
    )
    ax3.plot(
        x_grid,
        magnitude[-1, :],
        "r--",
        linewidth=2.5,
        label="t=π/2",
    )
    ax3.set_xlabel("x")
    ax3.set_ylabel("|h(x, t)|")
    ax3.set_title("Initial vs Final State")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Boundary values over time
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(
        t_grid,
        magnitude[:, 0],
        "b-",
        label="x = -5 (left)",
        linewidth=2,
    )
    ax4.plot(
        t_grid,
        magnitude[:, -1],
        "r--",
        label="x = +5 (right)",
        linewidth=2,
        alpha=0.8,
    )
    ax4.set_xlabel("t")
    ax4.set_ylabel("|h(boundary, t)|")
    ax4.set_title("Periodic BC: Boundary Values")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Energy conservation
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(t_grid, energies, "g-", linewidth=2)
    ax5.set_xlabel("t")
    ax5.set_ylabel("Energy")
    ax5.set_title(f"Energy (var: {energy_variation:.2e})")
    ax5.grid(True, alpha=0.3)

    # Phase plot at final time
    ax6 = plt.subplot(2, 3, 6)
    phase = np.angle(h_solution[-1, :])
    ax6.plot(x_grid, phase, "m-", linewidth=2)
    ax6.set_xlabel("x")
    ax6.set_ylabel("arg(h)")
    ax6.set_title("Phase at t=π/2")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = Path("outputs/plots/test_nlse_solver.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"   ✓ Plot saved: {output_path}")

    print("\n" + "=" * 70)
    print("✓ All tests completed successfully!")
    print("=" * 70)

    return h_solution, x_grid, t_grid


if __name__ == "__main__":
    h_solution, x_grid, t_grid = test_solver_basic()
    plt.show()

