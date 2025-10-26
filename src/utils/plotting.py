"""Plotting utilities for Schrödinger equation results.

Provides functions to create:
- Heatmaps of |h(x,t)|
- Time-slice overlays
- Loss curves
- Error vs time plots
- Model vs solver comparisons
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize


# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10


def plot_solution_heatmap(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    h_solution: np.ndarray,
    title: str = "Solution Magnitude |h(x,t)|",
    ax: Optional[plt.Axes] = None,
    colorbar: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Axes:
    """Plot heatmap of solution magnitude.

    Args:
        x_grid: Spatial grid (1D array)
        t_grid: Temporal grid (1D array)
        h_solution: Complex solution (nt, nx)
        title: Plot title
        ax: Axes to plot on (creates new if None)
        colorbar: Whether to add colorbar
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    magnitude = np.abs(h_solution)
    extent = [x_grid.min(), x_grid.max(), t_grid.min(), t_grid.max()]

    im = ax.imshow(
        magnitude,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(title)

    if colorbar:
        plt.colorbar(im, ax=ax, label="|h(x,t)|")

    return ax


def plot_time_slices(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    h_solution: np.ndarray,
    time_indices: Optional[List[int]] = None,
    title: str = "Time Slices of |h(x,t)|",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot multiple time slices of solution magnitude.

    Args:
        x_grid: Spatial grid
        t_grid: Temporal grid
        h_solution: Complex solution (nt, nx)
        time_indices: Indices to plot (default: 5 evenly spaced)
        title: Plot title
        ax: Axes to plot on

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if time_indices is None:
        # Default: 5 evenly spaced time points
        nt = len(t_grid)
        time_indices = [0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]

    magnitude = np.abs(h_solution)
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

    for i, (idx, color) in enumerate(zip(time_indices, colors)):
        ax.plot(
            x_grid,
            magnitude[idx, :],
            label=f"t={t_grid[idx]:.3f}",
            linewidth=2,
            color=color,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("|h(x,t)|")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_comparison(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    h_model: np.ndarray,
    h_solver: np.ndarray,
    time_indices: Optional[List[int]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create comprehensive model vs solver comparison plot.

    Args:
        x_grid: Spatial grid
        t_grid: Temporal grid
        h_model: Model prediction (nt, nx)
        h_solver: Ground truth from solver (nt, nx)
        time_indices: Time indices for slices
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(16, 12))

    if time_indices is None:
        nt = len(t_grid)
        time_indices = [0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]

    magnitude_model = np.abs(h_model)
    magnitude_solver = np.abs(h_solver)
    error = np.abs(h_model - h_solver)

    # Global min/max for consistent colormaps
    vmin = 0
    vmax = max(magnitude_model.max(), magnitude_solver.max())

    # Row 1: Heatmaps
    ax1 = plt.subplot(3, 3, 1)
    plot_solution_heatmap(
        x_grid, t_grid, h_solver, "Ground Truth (Solver)", ax=ax1, vmin=vmin, vmax=vmax
    )

    ax2 = plt.subplot(3, 3, 2)
    plot_solution_heatmap(
        x_grid, t_grid, h_model, "Model Prediction", ax=ax2, vmin=vmin, vmax=vmax
    )

    ax3 = plt.subplot(3, 3, 3)
    im = ax3.imshow(
        error,
        aspect="auto",
        origin="lower",
        extent=[x_grid.min(), x_grid.max(), t_grid.min(), t_grid.max()],
        cmap="hot",
    )
    ax3.set_xlabel("x")
    ax3.set_ylabel("t")
    ax3.set_title("Absolute Error |h_model - h_solver|")
    plt.colorbar(im, ax=ax3, label="Error")

    # Row 2: Time slices comparison
    n_slices = min(3, len(time_indices))
    for i in range(n_slices):
        ax = plt.subplot(3, 3, 4 + i)
        idx = time_indices[i]
        t_val = t_grid[idx]

        ax.plot(
            x_grid,
            magnitude_solver[idx, :],
            "b-",
            linewidth=2.5,
            label="Solver",
        )
        ax.plot(
            x_grid,
            magnitude_model[idx, :],
            "r--",
            linewidth=2,
            label="Model",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("|h|")
        ax.set_title(f"t = {t_val:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Row 3: Error analysis
    ax7 = plt.subplot(3, 3, 7)
    error_vs_time = error.mean(axis=1)
    ax7.plot(t_grid, error_vs_time, "g-", linewidth=2)
    ax7.set_xlabel("t")
    ax7.set_ylabel("Mean |Error|")
    ax7.set_title("Error vs Time")
    ax7.grid(True, alpha=0.3)

    ax8 = plt.subplot(3, 3, 8)
    error_vs_space = error.mean(axis=0)
    ax8.plot(x_grid, error_vs_space, "m-", linewidth=2)
    ax8.set_xlabel("x")
    ax8.set_ylabel("Mean |Error|")
    ax8.set_title("Error vs Space")
    ax8.grid(True, alpha=0.3)

    ax9 = plt.subplot(3, 3, 9)
    relative_errors = []
    for i in range(len(t_grid)):
        diff_norm = np.sqrt(np.sum(np.abs(h_model[i, :] - h_solver[i, :]) ** 2))
        true_norm = np.sqrt(np.sum(np.abs(h_solver[i, :]) ** 2))
        relative_errors.append((diff_norm / true_norm).item())

    ax9.plot(t_grid, relative_errors, "c-", linewidth=2)
    ax9.set_xlabel("t")
    ax9.set_ylabel("Relative L² Error")
    ax9.set_title("Relative Error vs Time")
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_loss_curves(
    losses: Dict[str, List[float]],
    title: str = "Training Loss Curves",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot training loss curves.

    Args:
        losses: Dictionary with loss history {name: [values]}
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total loss
    if "total_loss" in losses:
        axes[0, 0].plot(losses["total_loss"], "b-", linewidth=2)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Total Loss")
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale("log")

    # MSE components
    ax = axes[0, 1]
    if "mse_0" in losses:
        ax.plot(losses["mse_0"], label="MSE₀ (initial)", linewidth=2)
    if "mse_b" in losses:
        ax.plot(losses["mse_b"], label="MSE_b (boundary)", linewidth=2)
    if "mse_f" in losses:
        ax.plot(losses["mse_f"], label="MSE_f (residual)", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Components")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Relative L2 error
    if "relative_l2_error" in losses:
        axes[1, 0].plot(losses["relative_l2_error"], "g-", linewidth=2)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Relative L² Error")
        axes[1, 0].set_title("Relative L² Error")
        axes[1, 0].grid(True, alpha=0.3)

    # Loss breakdown pie chart (final epoch)
    ax = axes[1, 1]
    if all(k in losses for k in ["mse_0", "mse_b", "mse_f"]):
        final_values = [
            losses["mse_0"][-1],
            losses["mse_b"][-1],
            losses["mse_f"][-1],
        ]
        labels = ["MSE₀\n(initial)", "MSE_b\n(boundary)", "MSE_f\n(residual)"]
        colors = plt.cm.Set3(range(3))
        ax.pie(final_values, labels=labels, autopct="%1.1f%%", colors=colors)
        ax.set_title("Final Loss Breakdown")

    plt.suptitle(title, fontsize=14, y=1.00)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_phase_comparison(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    h_model: np.ndarray,
    h_solver: np.ndarray,
    time_idx: int = -1,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot phase comparison at a specific time.

    Args:
        x_grid: Spatial grid
        t_grid: Temporal grid
        h_model: Model prediction
        h_solver: Ground truth
        time_idx: Time index to plot (default: -1, final time)
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    t_val = t_grid[time_idx]

    # Magnitude
    axes[0].plot(
        x_grid, np.abs(h_solver[time_idx, :]), "b-", linewidth=2.5, label="Solver"
    )
    axes[0].plot(
        x_grid, np.abs(h_model[time_idx, :]), "r--", linewidth=2, label="Model"
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("|h|")
    axes[0].set_title(f"Magnitude at t={t_val:.3f}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Phase
    axes[1].plot(
        x_grid, np.angle(h_solver[time_idx, :]), "b-", linewidth=2.5, label="Solver"
    )
    axes[1].plot(
        x_grid, np.angle(h_model[time_idx, :]), "r--", linewidth=2, label="Model"
    )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("arg(h)")
    axes[1].set_title(f"Phase at t={t_val:.3f}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Real and imaginary parts
    axes[2].plot(x_grid, h_solver[time_idx, :].real, "b-", linewidth=2, label="Re(solver)")
    axes[2].plot(x_grid, h_solver[time_idx, :].imag, "b--", linewidth=2, label="Im(solver)")
    axes[2].plot(x_grid, h_model[time_idx, :].real, "r-", linewidth=1.5, label="Re(model)", alpha=0.7)
    axes[2].plot(x_grid, h_model[time_idx, :].imag, "r--", linewidth=1.5, label="Im(model)", alpha=0.7)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("h")
    axes[2].set_title(f"Real & Imaginary at t={t_val:.3f}")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


if __name__ == "__main__":
    """Test plotting utilities."""
    print("Testing plotting utilities...")

    # Create dummy data
    x_grid = np.linspace(-5, 5, 100)
    t_grid = np.linspace(0, np.pi / 2, 50)
    X, T = np.meshgrid(x_grid, t_grid)

    # Dummy solution
    h_solution = 2.0 / np.cosh(X) * np.exp(1j * T)

    # Test heatmap
    fig, ax = plt.subplots()
    plot_solution_heatmap(x_grid, t_grid, h_solution, ax=ax)
    plt.savefig("outputs/plots/test_heatmap.png")
    print("✓ Heatmap test saved")

    # Test time slices
    fig, ax = plt.subplots()
    plot_time_slices(x_grid, t_grid, h_solution, ax=ax)
    plt.savefig("outputs/plots/test_slices.png")
    print("✓ Time slices test saved")

    plt.close("all")
    print("✓ Plotting utilities test completed!")

