# src/evaluate/rfwfc_smoothness.py
"""
Run RF-Wavelet (RFWFC) smoothness analysis on our trained Schrödinger model.

This script mirrors the example_mnist_1d.ipynb flow:

1) Build args (like loaded_example_args)
2) Load model from MLflow run_id
3) Load dataset and wrap as Dataset/DataLoader
4) Call RFWFC's run_smoothness_analysis(args, model, dataset, test_dataset, layers, data_loader)
5) Save JSONs/plots under outputs/evaluation/rfwfc/<run_id>/
"""

import os
import sys
import argparse
import pathlib
import torch
from src.utils.rfwfc_utils import WrappedSchrodingerNet

# Ensure the local RFWFC repo is importable
# (project_root/RFWFC should exist)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
RFWFC_ROOT = PROJECT_ROOT / "RFWFC"
if str(RFWFC_ROOT) not in sys.path:
    sys.path.insert(0, str(RFWFC_ROOT))

# Import the RFWFC analysis API exactly like their notebook:
from DL_Layer_Analysis.DL_smoothness import init_params, run_smoothness_analysis  # type: ignore

# Our utilities
from src.utils.rfwfc_utils import (
    build_args,
    load_model_from_mlflow,
    load_npz_dataset,
    make_dataloaders,
    get_named_hidden_layers,
    infer_feature_dimension_from_model,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", required=True, type=str, help="MLflow run ID to load model weights")
    p.add_argument("--dataset", default="data/processed/dataset.npz", type=str, help="NPZ with (x,t,u,v)")
    p.add_argument("--num_samples", default=100000, type=int, help="Subsample size from dataset for analysis")
    p.add_argument("--batch_size", default=4096, type=int)
    p.add_argument("--trees", default=100, type=int)
    p.add_argument("--low_eps", default=0.4, type=float)
    p.add_argument("--high_eps", default=0.1, type=float)
    p.add_argument("--use_clustering", default=False, type=lambda s: s.lower()=="true")
    p.add_argument("--create_umap", default=False, type=lambda s: s.lower()=="true")
    return p.parse_args()


def main():
    args_cli = parse_args()

    # Output/checkpoints folder for RFWFC (like the notebook)
    base_out = PROJECT_ROOT / "outputs" / "evaluation" / "rfwfc" / args_cli.run_id
    base_out.mkdir(parents=True, exist_ok=True)
    checkpoints_folder = str(base_out)

    # 1) Build RFWFC-args “namespace” (matches notebook usage)
    rfw_args = build_args(
        checkpoints_folder=checkpoints_folder,
        trees=args_cli.trees,
        low_eps=args_cli.low_eps,
        high_eps=args_cli.high_eps,
        use_clustering=args_cli.use_clustering,
        create_umap=args_cli.create_umap,
    )

    # 2) Load model from MLflow
    model = load_model_from_mlflow(args_cli.run_id)
    model = WrappedSchrodingerNet(model)
    
    # Make sure RFWFC looks where we will save the shim:
    rfw_args.checkpoint_file_name = "best_model.pt"  # already set in your dataclass
    ckpt_path = os.path.join(checkpoints_folder, rfw_args.checkpoint_file_name)

    # Save a checkpoint in the format RFWFC expects: {'checkpoint': state_dict}
    os.makedirs(checkpoints_folder, exist_ok=True)
    torch.save({"checkpoint": model.state_dict()}, ckpt_path)
    print(f"  ✓ Wrote shim checkpoint for RFWFC: {ckpt_path}")

    # (Optional) also add a params dict, some versions look for it
    setattr(rfw_args, "params", vars(rfw_args))
    
    # 3) Load data & make dataloaders
    x, t, u, v = load_npz_dataset(args_cli.dataset, num_samples=args_cli.num_samples)
    data_loader, train_ds, test_ds = make_dataloaders(x, t, u, v, batch_size=args_cli.batch_size)

    # 4) Get layers to analyze (must match your model naming)
    layers = get_named_hidden_layers(model)
    if not layers:
        print("No named layers found. Expected layer_1..layer_5 in your model.")
        sys.exit(1)

    # 5) Initialize params via RFWFC (as in example)
    # NOTE: Their init_params builds a richer args + maybe returns (args, model, dataset, test_dataset, layers, data_loader).
    # We already have model/datasets/layers; we’ll pass ours *into* run_smoothness_analysis like the notebook does.
    #
    # If your clone requires calling init_params to enrich args with defaults, do it now.
    # The notebook calls:
    #   args, model, dataset, test_dataset, layers, data_loader = init_params(args=loaded_example_args)
    #
    # Here we try to mimic only the arg-enrichment:
    # --- 5) Initialize params via RFWFC (args enrichment only) ---
    rfw_args.feature_dimension = infer_feature_dimension_from_model(model, fallback=100)
    
    try:
        res = init_params(args=rfw_args)  # enriches defaults/paths etc.
        # Some forks return a tuple; we only want the enriched args.
        if isinstance(res, tuple) and len(res) == 6:
            rfw_args_enriched, _, _, _, _, _ = res
            # keep our own (model, layers, data_loader) — DO NOT overwrite them
            for k, v in vars(rfw_args_enriched).items():
                setattr(rfw_args, k, v)
    except TypeError:
        # Different signature — it's fine; we already have what we need.
        pass

    # Recompute layers AFTER any enrichment, from our wrapped model only.
    layers = get_named_hidden_layers(model)
    if not layers:
        print("No named layers found. Expected layer_1..layer_5 in your model.")
        sys.exit(1)

    # 6) Run smoothness analysis exactly like notebook:
    # run_smoothness_analysis(args, model, dataset, test_dataset, layers, data_loader)
    #
    # We pass our train/test datasets and loader; run_smoothness_analysis will handle
    # RF training and will write JSONs/plots to rfw_args.checkpoints_folder.
    print("Running RFWFC smoothness analysis...")
    sparsity_run_output = run_smoothness_analysis(
        rfw_args, model, train_ds, test_ds, layers, data_loader  # type: ignore
    )
    print("Done. Results saved to:", checkpoints_folder)

    # Optional: you can plot epochs as in notebook; we just call their helper if available.
    try:
        from DL_Layer_Analysis.plot_DL_json_results import plot_epochs  # type: ignore
        plot_epochs(rfw_args.checkpoints_folder, plot_test=True, add_fill=False, use_clustering=rfw_args.use_clustering)
        print("Saved plots via RFWFC plot_epochs().")
    except Exception as e:
        print("Skipping plot_epochs() (not available):", e)


if __name__ == "__main__":
    main()
