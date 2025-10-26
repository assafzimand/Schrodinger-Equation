"""Test script to load and validate the generated dataset."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generate_dataset import load_dataset


def test_dataset(dataset_path: str = "data/processed/dataset.npz"):
    """Load dataset and print statistics."""
    print("=" * 70)
    print("Dataset Validation")
    print("=" * 70)

    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)

    print("\n1. Dataset Keys:")
    for key in sorted(dataset.keys()):
        print(f"   {key}: shape = {dataset[key].shape}, dtype = {dataset[key].dtype}")

    # Verify shapes
    print("\n2. Shape Verification:")
    n_f = len(dataset["x_f"])
    n_0 = len(dataset["x_0"])
    n_b = len(dataset["x_b"])

    print(f"   Collocation points (N_f): {n_f}")
    print(f"   Initial points (N_0):     {n_0}")
    print(f"   Boundary points (N_b):    {n_b}")
    print(f"   Total:                    {n_f + n_0 + n_b}")

    # Check consistency
    print("\n3. Consistency Checks:")
    checks_passed = True

    # Check collocation
    if (
        len(dataset["t_f"]) == n_f
        and len(dataset["u_f"]) == n_f
        and len(dataset["v_f"]) == n_f
    ):
        print("   ✓ Collocation arrays consistent")
    else:
        print("   ✗ Collocation arrays inconsistent!")
        checks_passed = False

    # Check initial
    if (
        len(dataset["t_0"]) == n_0
        and len(dataset["u_0"]) == n_0
        and len(dataset["v_0"]) == n_0
    ):
        print("   ✓ Initial condition arrays consistent")
    else:
        print("   ✗ Initial condition arrays inconsistent!")
        checks_passed = False

    # Check boundary
    if (
        len(dataset["t_b"]) == n_b
        and len(dataset["u_b"]) == n_b
        and len(dataset["v_b"]) == n_b
    ):
        print("   ✓ Boundary condition arrays consistent")
    else:
        print("   ✗ Boundary condition arrays inconsistent!")
        checks_passed = False

    # Check initial condition is at t=0
    if np.allclose(dataset["t_0"], 0.0):
        print("   ✓ Initial condition at t=0")
    else:
        print(f"   ✗ Initial condition not at t=0! Range: [{dataset['t_0'].min():.3f}, {dataset['t_0'].max():.3f}]")
        checks_passed = False

    # Check boundary points
    x_min, x_max = dataset["x_b"].min(), dataset["x_b"].max()
    print(f"   ✓ Boundary points at x ∈ [{x_min:.3f}, {x_max:.3f}]")

    # Statistics
    print("\n4. Data Statistics:")
    print(f"   Collocation:")
    print(f"     x range: [{dataset['x_f'].min():.3f}, {dataset['x_f'].max():.3f}]")
    print(f"     t range: [{dataset['t_f'].min():.3f}, {dataset['t_f'].max():.3f}]")
    print(f"     |h| range: [{np.sqrt(dataset['u_f']**2 + dataset['v_f']**2).min():.3f}, {np.sqrt(dataset['u_f']**2 + dataset['v_f']**2).max():.3f}]")

    print(f"   Initial condition:")
    print(f"     x range: [{dataset['x_0'].min():.3f}, {dataset['x_0'].max():.3f}]")
    print(f"     |h| range: [{np.sqrt(dataset['u_0']**2 + dataset['v_0']**2).min():.3f}, {np.sqrt(dataset['u_0']**2 + dataset['v_0']**2).max():.3f}]")

    print(f"   Boundary condition:")
    print(f"     t range: [{dataset['t_b'].min():.3f}, {dataset['t_b'].max():.3f}]")
    print(f"     |h| range: [{np.sqrt(dataset['u_b']**2 + dataset['v_b']**2).min():.3f}, {np.sqrt(dataset['u_b']**2 + dataset['v_b']**2).max():.3f}]")

    # Visualize
    print("\n5. Generating visualization...")
    fig = plt.figure(figsize=(14, 10))

    # Collocation points
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(
        dataset["x_f"],
        dataset["t_f"],
        c=np.sqrt(dataset["u_f"] ** 2 + dataset["v_f"] ** 2),
        s=1,
        cmap="viridis",
        alpha=0.5,
    )
    plt.colorbar(scatter, ax=ax1, label="|h|")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    ax1.set_title(f"Collocation Points (N={n_f})")

    # Initial condition points
    ax2 = plt.subplot(2, 3, 2)
    h_0_mag = np.sqrt(dataset["u_0"] ** 2 + dataset["v_0"] ** 2)
    ax2.scatter(dataset["x_0"], h_0_mag, s=20, alpha=0.7, label="Sampled")
    ax2.set_xlabel("x")
    ax2.set_ylabel("|h(x, 0)|")
    ax2.set_title(f"Initial Condition (N={n_0})")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Boundary condition points
    ax3 = plt.subplot(2, 3, 3)
    h_b_mag = np.sqrt(dataset["u_b"] ** 2 + dataset["v_b"] ** 2)
    # Separate left and right boundaries
    left_mask = dataset["x_b"] < -4
    right_mask = dataset["x_b"] > 4
    ax3.scatter(
        dataset["t_b"][left_mask],
        h_b_mag[left_mask],
        s=20,
        alpha=0.7,
        label="x=-5",
    )
    ax3.scatter(
        dataset["t_b"][right_mask],
        h_b_mag[right_mask],
        s=20,
        alpha=0.7,
        label="x=+5",
    )
    ax3.set_xlabel("t")
    ax3.set_ylabel("|h(±5, t)|")
    ax3.set_title(f"Boundary Condition (N={n_b})")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Real part distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(dataset["u_f"], bins=50, alpha=0.7, label="Collocation")
    ax4.set_xlabel("u (real part)")
    ax4.set_ylabel("Count")
    ax4.set_title("Real Part Distribution")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Imaginary part distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(dataset["v_f"], bins=50, alpha=0.7, color="orange", label="Collocation")
    ax5.set_xlabel("v (imag part)")
    ax5.set_ylabel("Count")
    ax5.set_title("Imaginary Part Distribution")
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # Magnitude distribution
    ax6 = plt.subplot(2, 3, 6)
    mag_f = np.sqrt(dataset["u_f"] ** 2 + dataset["v_f"] ** 2)
    ax6.hist(mag_f, bins=50, alpha=0.7, color="green")
    ax6.set_xlabel("|h|")
    ax6.set_ylabel("Count")
    ax6.set_title("Magnitude Distribution")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = Path("outputs/plots/dataset_validation.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"   ✓ Plot saved: {output_path}")

    print("\n" + "=" * 70)
    if checks_passed:
        print("✓ All validation checks passed!")
    else:
        print("⚠ Some validation checks failed!")
    print("=" * 70)

    return dataset


if __name__ == "__main__":
    dataset = test_dataset()
    plt.show()

