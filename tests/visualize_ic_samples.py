"""Visualize initial condition (IC) sample distribution.

This script shows how IC samples are distributed across the spatial domain
with Gaussian bias toward the center where |h(x,0)| = |2*sech(x)| is large.

Usage:
    python tests/visualize_ic_samples.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config


def sech(x):
    """Hyperbolic secant function."""
    return 1.0 / np.cosh(x)


def visualize_ic_distribution(
    n_samples: int = 1000,
    ic_sigma: float = 1.2,
    ic_mix: float = 0.7,
    x_min: float = -5.0,
    x_max: float = 5.0,
    seed: int = 42,
    save_path: str = "outputs/diagnostics/ic_sample_distribution.png",
):
    """Visualize IC sample distribution compared to the true IC function.
    
    Args:
        n_samples: Number of samples to generate
        ic_sigma: Standard deviation for Gaussian distribution
        ic_mix: Fraction of Gaussian samples
        x_min: Minimum x value
        x_max: Maximum x value
        seed: Random seed
        save_path: Path to save figure
    """
    rng = np.random.RandomState(seed)
    
    # Generate samples with Gaussian bias
    n_gaussian = int(n_samples * ic_mix)
    n_uniform = n_samples - n_gaussian
    
    x_gaussian = rng.normal(loc=0.0, scale=ic_sigma, size=n_gaussian)
    x_gaussian = np.clip(x_gaussian, x_min, x_max)
    
    x_uniform = rng.uniform(x_min, x_max, size=n_uniform)
    
    x_samples = np.concatenate([x_gaussian, x_uniform])
    rng.shuffle(x_samples)
    
    # Also generate uniform samples for comparison
    x_samples_uniform = rng.uniform(x_min, x_max, size=n_samples)
    
    # True IC function
    x_dense = np.linspace(x_min, x_max, 1000)
    ic_true = 2.0 * sech(x_dense)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Histogram of Gaussian-biased samples
    ax = axes[0, 0]
    ax.hist(x_samples, bins=50, alpha=0.7, color='C0', edgecolor='black', density=True)
    ax.set_xlabel('x')
    ax.set_ylabel('Sample Density')
    ax.set_title(f'Gaussian-Biased IC Samples\n({ic_mix*100:.0f}% Gaussian, σ={ic_sigma})')
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='x=0 (center)')
    ax.legend()
    
    # Plot 2: Histogram of uniform samples for comparison
    ax = axes[0, 1]
    ax.hist(x_samples_uniform, bins=50, alpha=0.7, color='C1', edgecolor='black', density=True)
    ax.set_xlabel('x')
    ax.set_ylabel('Sample Density')
    ax.set_title('Uniform IC Samples (for comparison)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Samples vs IC function
    ax = axes[1, 0]
    ax.plot(x_dense, ic_true, 'b-', linewidth=2, label='|h(x,0)| = 2·sech(x)')
    ax.scatter(x_samples, np.zeros_like(x_samples), 
              alpha=0.3, s=20, color='C0', label=f'Gaussian-biased samples (n={n_samples})')
    ax.set_xlabel('x')
    ax.set_ylabel('|h(x, 0)|')
    ax.set_title('IC Samples vs True IC Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative distribution comparison
    ax = axes[1, 1]
    x_sorted_biased = np.sort(x_samples)
    x_sorted_uniform = np.sort(x_samples_uniform)
    cdf_biased = np.arange(1, len(x_sorted_biased) + 1) / len(x_sorted_biased)
    cdf_uniform = np.arange(1, len(x_sorted_uniform) + 1) / len(x_sorted_uniform)
    
    ax.plot(x_sorted_biased, cdf_biased, 'C0-', linewidth=2, label='Gaussian-biased')
    ax.plot(x_sorted_uniform, cdf_uniform, 'C1--', linewidth=2, label='Uniform')
    ax.set_xlabel('x')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='red', linestyle=':', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved IC distribution plot to: {save_path}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("IC Sample Distribution Statistics")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Total samples: {n_samples}")
    print(f"  Gaussian samples: {n_gaussian} ({ic_mix*100:.0f}%)")
    print(f"  Uniform samples: {n_uniform} ({(1-ic_mix)*100:.0f}%)")
    print(f"  Gaussian σ: {ic_sigma}")
    print(f"  Domain: [{x_min}, {x_max}]")
    
    print(f"\nGaussian-Biased Samples:")
    print(f"  Mean: {np.mean(x_samples):.4f}")
    print(f"  Std: {np.std(x_samples):.4f}")
    print(f"  Min: {np.min(x_samples):.4f}")
    print(f"  Max: {np.max(x_samples):.4f}")
    print(f"  Median: {np.median(x_samples):.4f}")
    
    # Count samples in center region |x| < 2σ
    center_mask = np.abs(x_samples) < (2 * ic_sigma)
    print(f"\nSamples within |x| < {2*ic_sigma:.1f}: {np.sum(center_mask)} ({np.sum(center_mask)/n_samples*100:.1f}%)")
    
    print(f"\nUniform Samples (for comparison):")
    print(f"  Mean: {np.mean(x_samples_uniform):.4f}")
    print(f"  Std: {np.std(x_samples_uniform):.4f}")
    
    center_mask_uniform = np.abs(x_samples_uniform) < (2 * ic_sigma)
    print(f"  Samples within |x| < {2*ic_sigma:.1f}: {np.sum(center_mask_uniform)} ({np.sum(center_mask_uniform)/n_samples*100:.1f}%)")
    
    print("\n" + "=" * 70)
    
    plt.show()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize IC sample distribution with Gaussian bias"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/dataset.yaml",
        help="Path to dataset config file",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples to generate (default: 1000)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="outputs/diagnostics/ic_sample_distribution.png",
        help="Path to save the plot",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Visualize
    visualize_ic_distribution(
        n_samples=args.n_samples,
        ic_sigma=config.dataset.ic_sigma,
        ic_mix=config.dataset.ic_mix,
        x_min=config.solver.x_min,
        x_max=config.solver.x_max,
        seed=config.dataset.seed,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()

