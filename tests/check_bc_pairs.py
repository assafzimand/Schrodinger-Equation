import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_boundary(dataset_path: Path):
    d = np.load(dataset_path)
    x_b = d["x_b"].reshape(-1)
    t_b = d["t_b"].reshape(-1)
    return x_b, t_b

def match_times(left_t, right_t, tol=1e-8):
    # Two-pointer matching on sorted arrays with tolerance
    l = np.sort(left_t.copy())
    r = np.sort(right_t.copy())
    i = j = 0
    matches = []
    unmatched_left = []
    unmatched_right = []

    while i < len(l) and j < len(r):
        diff = l[i] - r[j]
        if abs(diff) <= tol:
            matches.append((l[i], r[j]))
            i += 1
            j += 1
        elif diff < 0:
            unmatched_left.append(l[i])
            i += 1
        else:
            unmatched_right.append(r[j])
            j += 1

    while i < len(l):
        unmatched_left.append(l[i]); i += 1
    while j < len(r):
        unmatched_right.append(r[j]); j += 1

    return np.array(matches), np.array(unmatched_left), np.array(unmatched_right)

def main():
    parser = argparse.ArgumentParser(description="Check BC pairing (±L, same t).")
    parser.add_argument("--dataset", type=str, default="data/processed/dataset.npz")
    parser.add_argument("--xmin", type=float, default=-5.0)
    parser.add_argument("--xmax", type=float, default=5.0)
    parser.add_argument("--xtol", type=float, default=1e-6, help="Tolerance for identifying boundary x")
    parser.add_argument("--ttol", type=float, default=1e-8, help="Tolerance for matching times")
    parser.add_argument("--out", type=str, default="outputs/diagnostics/bc_pairs_train.png")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_b, t_b = load_boundary(dataset_path)

    left_mask = np.isclose(x_b, args.xmin, atol=args.xtol)
    right_mask = np.isclose(x_b, args.xmax, atol=args.xtol)

    left_t = t_b[left_mask]
    right_t = t_b[right_mask]

    matches, left_only, right_only = match_times(left_t, right_t, tol=args.ttol)

    # Print summary
    print("=" * 70)
    print(f"Dataset: {dataset_path}")
    print(f"Left boundary x={args.xmin}:  N_left  = {len(left_t)}")
    print(f"Right boundary x={args.xmax}: N_right = {len(right_t)}")
    print(f"Matched pairs (|Δt| <= {args.ttol}): {len(matches)}")
    print(f"Unmatched left-only times:  {len(left_only)}")
    print(f"Unmatched right-only times: {len(right_only)}")
    if len(matches) > 0:
        max_dt = np.max(np.abs(matches[:,0] - matches[:,1]))
        print(f"Max |Δt| among matched pairs: {max_dt:.3e}")
    print("=" * 70)

    # Plot: hist of times and unmatched diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Left: overlaid histograms of left/right times
    bins = max(10, int(np.sqrt(max(len(left_t), len(right_t)))) )
    axes[0].hist(left_t, bins=bins, alpha=0.6, label=f"left x={args.xmin}")
    axes[0].hist(right_t, bins=bins, alpha=0.6, label=f"right x={args.xmax}")
    axes[0].set_title("Boundary time distributions")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("count"); axes[0].legend()

    # Right: unmatched times as scatter along y-lines
    y_left = np.zeros_like(left_only)
    y_right = np.ones_like(right_only)
    if len(left_only) > 0:
        axes[1].scatter(left_only, y_left, c="tab:blue", s=12, label="unmatched left t")
    if len(right_only) > 0:
        axes[1].scatter(right_only, y_right, c="tab:orange", s=12, label="unmatched right t")
    axes[1].set_yticks([0,1]); axes[1].set_yticklabels([f"x={args.xmin}", f"x={args.xmax}"])
    axes[1].set_xlabel("t"); axes[1].set_title("Unmatched boundary times")
    axes[1].legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")

if __name__ == "__main__":
    main()
