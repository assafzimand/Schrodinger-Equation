import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def sech(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.cosh(x)


def main():
    parser = argparse.ArgumentParser(
        description="Plot IC check: 2*sech(x) vs dataset u(x, t=0)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/dataset.npz",
        help="Path to training dataset .npz",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="outputs/diagnostics/ic_check.png",
        help="Path to save the output plot",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window in addition to saving",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = np.load(dataset_path)
    required_keys = {"x_0", "t_0", "u_0"}
    missing = required_keys - set(data.files)
    if missing:
        raise KeyError(f"Dataset is missing required keys: {sorted(missing)}")

    x_0 = data["x_0"].astype(np.float64)
    t_0 = data["t_0"].astype(np.float64)
    u_0 = data["u_0"].astype(np.float64)

    # Validate that IC samples are at t=0
    max_abs_t0 = float(np.max(np.abs(t_0))) if t_0.size > 0 else 0.0
    if max_abs_t0 > 1e-10:
        print(f"[IC WARNING] t_0 values deviate from 0 by up to {max_abs_t0:.3e}")
    else:
        print("[IC OK] All t_0 are effectively 0.")

    # Build a dense x grid spanning the IC sample domain
    x_min = float(np.min(x_0))
    x_max = float(np.max(x_0))
    x_dense = np.linspace(x_min, x_max, 1000)
    u_ic = 2.0 * sech(x_dense)

    # Plot analytic IC vs dataset samples
    plt.figure(figsize=(8, 5))
    plt.plot(x_dense, u_ic, label="2·sech(x) (analytic IC)", color="C0", lw=2)
    plt.scatter(x_0, u_0, label="Dataset u(x, t=0)", color="C1", s=16, alpha=0.8)
    plt.title("IC check: 2·sech(x) vs dataset u at t=0")
    plt.xlabel("x")
    plt.ylabel("u(x, 0)")
    plt.grid(True, ls=":", alpha=0.5)
    plt.legend()

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to: {save_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()


