#!/usr/bin/env python3
"""
Export per-bus voltage magnitude and angle predictions on the test split.

Writes three artifacts per model/case:
  - JSONL (one sample per line, arrays for vm/va pred and true)
  - CSV (long format: one row per sample x bus)
  - PNG plots (scatter, temporal MAE, optional network maps)

Usage:
    python scripts/export_inference_predictions.py --case 118 --models PIResnetGCLSTM
    python scripts/export_inference_predictions.py --case all --models all
    python scripts/export_inference_predictions.py --case 118 --models all --max-samples 20
"""

import argparse
import glob
import json
import logging
import os
import sys
import warnings
from typing import Optional

os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("LIGHTNING_PYTORCH_DISABLE_TIP", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.preprocess_data import denormalize_predictions
from src.constants import PredictionIndices, TargetIndices
from src.export.sample_metadata import load_test_sample_metadata, select_coverage_indices
from src.models import RECURRENT_MODELS, SPATIAL_MODELS, PowerFlowDataModule, get_model_registry
from src.visualization.plot_inference_predictions import (
    plot_combined_fraction_temporal_mae,
    plot_combined_pred_vs_true_scatter,
    plot_sample_network_comparison_panel,
)

MODEL_REGISTRY = get_model_registry()
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "training.yaml")

# Final complete cluster runs (one session per bus case, all 7 models, save_top_k=3).
FINAL_TRAINING_SESSIONS = {
    "case33": "session_20260312_223644",
    "case57": "session_20260316_070512",
    "case118": "session_20260319_223350",
}

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


def load_training_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_cases(case_arg: str):
    if case_arg.lower() == "all":
        prep_root = os.path.join(PROJECT_ROOT, "data", "prep")
        return sorted(
            (
                name
                for name in os.listdir(prep_root)
                if os.path.isdir(os.path.join(prep_root, name)) and name.startswith("case")
            ),
            key=lambda x: int(x.replace("case", ""))
        )
    return [
        c.strip() if c.strip().startswith("case") else f"case{c.strip()}"
        for c in case_arg.split(",")
    ]


def _best_checkpoint(paths):
    def val_loss(path):
        name = os.path.basename(path)
        marker = "val_loss="
        if marker not in name:
            return float("inf")
        try:
            return float(name.split(marker, 1)[1].replace(".ckpt", ""))
        except ValueError:
            return float("inf")

    return min(paths, key=val_loss)


def discover_checkpoints(checkpoint_root: str, session_name: Optional[str] = None):
    search_root = (
        os.path.join(checkpoint_root, session_name)
        if session_name
        else checkpoint_root
    )
    pattern = os.path.join(search_root, "**", "*.ckpt")
    found = {}
    for path in glob.glob(pattern, recursive=True):
        parent = os.path.basename(os.path.dirname(path))
        if "_" not in parent:
            continue
        model_name, case_name = parent.rsplit("_", 1)
        found.setdefault((model_name, case_name), []).append(path)
    return {key: _best_checkpoint(paths) for key, paths in found.items()}


def resolve_models(model_arg: str, checkpoint_root: str, cases):
    if model_arg.lower() != "all":
        return [m.strip() for m in model_arg.split(",")]

    available = discover_checkpoints(checkpoint_root)
    models = sorted({model for (model, case) in available.keys() if case in cases})
    if not models:
        models = sorted(MODEL_REGISTRY.keys())
    return models


def get_checkpoint(
    model_name: str,
    case_name: str,
    checkpoint_root: str,
    explicit: Optional[str],
    use_final_sessions: bool,
):
    if explicit:
        return explicit

    session_name = FINAL_TRAINING_SESSIONS.get(case_name) if use_final_sessions else None
    available = discover_checkpoints(checkpoint_root, session_name=session_name)
    path = available.get((model_name, case_name))
    if path:
        return path

    if session_name:
        available = discover_checkpoints(checkpoint_root)
        path = available.get((model_name, case_name))
        if path:
            return path

    pattern = os.path.join(checkpoint_root, "**", f"{model_name}_{case_name}", "*.ckpt")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return _best_checkpoint(files)


def denorm_vm_va_from_preds(preds: torch.Tensor, meta: dict):
    pred_np = preds.detach().cpu().numpy()
    if pred_np.ndim == 4:
        vm_norm = pred_np[:, -1, :, PredictionIndices.VM]
        va_norm = pred_np[:, -1, :, PredictionIndices.VA]
    else:
        vm_norm = pred_np[:, :, PredictionIndices.VM]
        va_norm = pred_np[:, :, PredictionIndices.VA]

    mock = np.zeros((*vm_norm.shape, TargetIndices.NUM_TARGETS))
    mock[..., TargetIndices.VM] = vm_norm
    mock[..., TargetIndices.VA] = va_norm
    denorm = denormalize_predictions(mock, meta)
    return denorm[..., TargetIndices.VM], denorm[..., TargetIndices.VA]


def denorm_vm_va_from_targets(targets: torch.Tensor, meta: dict):
    target_np = targets.detach().cpu().numpy()
    return denormalize_predictions(target_np, meta)[..., TargetIndices.VM], denormalize_predictions(target_np, meta)[..., TargetIndices.VA]


def run_export(
    model_name: str,
    case_name: str,
    ckpt_path: str,
    device: torch.device,
    output_dir: str,
    max_samples: Optional[int],
    sample_strategy: str,
    plot_samples: int,
    skip_plots: bool,
    batch_size: int,
    seq_len: int,
    verbose: bool = False,
):
    is_recurrent = model_name in RECURRENT_MODELS
    effective_seq_len = 1 if model_name in SPATIAL_MODELS else seq_len

    dm = PowerFlowDataModule(
        data_dir=os.path.join(PROJECT_ROOT, "data", "prep"),
        case_name=case_name,
        batch_size=batch_size,
        seq_len=effective_seq_len,
    )
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    meta = dm.meta

    metadata = load_test_sample_metadata(case_name, PROJECT_ROOT)
    if is_recurrent:
        # SpatioTemporalDataset predicts the last step of each sequence.
        metadata = metadata[effective_seq_len - 1 :]
    if len(metadata) != len(dm.test_dataset):
        raise RuntimeError(
            f"Metadata length ({len(metadata)}) != test dataset length ({len(dm.test_dataset)}) "
            f"for {case_name}. Ensure data/prep matches the checkpoint training run."
        )

    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass.load_from_checkpoint(ckpt_path, strict=False)
    model.to(device)
    model.eval()

    vm_pred_batches = []
    va_pred_batches = []
    vm_true_batches = []
    va_true_batches = []

    sample_offset = 0
    with torch.no_grad():
        desc = f"{model_name} {case_name}"
        desc = f"{desc:<25}"
        for batch in tqdm(test_loader, desc=desc, leave=True,
                          bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} batches",
                          unit="batch"):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            preds = model(
                batch["features"],
                batch["edge_index_seq"] if is_recurrent else batch["edge_index"],
            )

            vm_pred, va_pred = denorm_vm_va_from_preds(preds, meta)
            vm_true, va_true = denorm_vm_va_from_targets(batch["targets"], meta)

            if max_samples is not None and sample_offset >= max_samples:
                break
            if max_samples is not None and sample_offset + vm_pred.shape[0] > max_samples:
                keep = max_samples - sample_offset
                vm_pred, va_pred = vm_pred[:keep], va_pred[:keep]
                vm_true, va_true = vm_true[:keep], va_true[:keep]

            vm_pred_batches.append(vm_pred)
            va_pred_batches.append(va_pred)
            vm_true_batches.append(vm_true)
            va_true_batches.append(va_true)
            sample_offset += vm_pred.shape[0]

    vm_pred_all = np.concatenate(vm_pred_batches, axis=0)
    va_pred_all = np.concatenate(va_pred_batches, axis=0)
    vm_true_all = np.concatenate(vm_true_batches, axis=0)
    va_true_all = np.concatenate(va_true_batches, axis=0)
    n_samples = vm_pred_all.shape[0]
    metadata = metadata[:n_samples]

    if sample_strategy == "coverage":
        keep = select_coverage_indices(metadata)
        vm_pred_all = vm_pred_all[keep]
        va_pred_all = va_pred_all[keep]
        vm_true_all = vm_true_all[keep]
        va_true_all = va_true_all[keep]
        metadata = [metadata[i] for i in keep]
        n_samples = len(metadata)

    case_out = os.path.join(output_dir, case_name, model_name)
    if os.path.exists(case_out):
        import shutil
        shutil.rmtree(case_out)
    os.makedirs(case_out, exist_ok=True)

    jsonl_path = os.path.join(case_out, f"{case_name}_predictions.jsonl")
    csv_path = os.path.join(case_out, f"{case_name}_predictions.csv")
    summary_path = os.path.join(case_out, f"{case_name}_summary.json")

    json_records = []
    csv_rows = []
    for i, meta_row in enumerate(metadata):
        record = {
            **meta_row,
            "model": model_name,
            "checkpoint": os.path.relpath(ckpt_path, PROJECT_ROOT),
            "vm_pred": vm_pred_all[i].tolist(),
            "va_pred": va_pred_all[i].tolist(),
            "vm_true": vm_true_all[i].tolist(),
            "va_true": va_true_all[i].tolist(),
            "mae_vm": float(np.mean(np.abs(vm_pred_all[i] - vm_true_all[i]))),
            "mae_va": float(np.mean(np.abs(va_pred_all[i] - va_true_all[i]))),
        }
        json_records.append(record)
        for bus_id in range(vm_pred_all.shape[1]):
            csv_rows.append(
                {
                    "sample_id": meta_row["sample_id"],
                    "case_name": case_name,
                    "model": model_name,
                    "timestep": meta_row["timestep"],
                    "renewable_fraction": meta_row["renewable_fraction"],
                    "topology_id": meta_row["topology_id"],
                    "bus_id": bus_id,
                    "vm_pred_pu": float(vm_pred_all[i, bus_id]),
                    "va_pred_rad": float(va_pred_all[i, bus_id]),
                    "vm_true_pu": float(vm_true_all[i, bus_id]),
                    "va_true_rad": float(va_true_all[i, bus_id]),
                    "vm_abs_error_pu": float(abs(vm_pred_all[i, bus_id] - vm_true_all[i, bus_id])),
                    "va_abs_error_rad": float(abs(va_pred_all[i, bus_id] - va_true_all[i, bus_id])),
                }
            )

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in json_records:
            f.write(json.dumps(record) + "\n")

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    summary = {
        "model": model_name,
        "case_name": case_name,
        "checkpoint": os.path.relpath(ckpt_path, PROJECT_ROOT),
        "num_test_samples": n_samples,
        "num_buses": int(vm_pred_all.shape[1]),
        "csv_rows": len(csv_rows),
        "mean_mae_vm_pu": float(np.mean(np.abs(vm_pred_all - vm_true_all))),
        "mean_mae_va_rad": float(np.mean(np.abs(va_pred_all - va_true_all))),
        "outputs": {
            "jsonl": os.path.relpath(jsonl_path, PROJECT_ROOT),
            "csv": os.path.relpath(csv_path, PROJECT_ROOT),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not skip_plots:
        plot_dir = os.path.join(case_out, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        if plot_samples > 0:
            chosen = []
            seen_fracs = set()
            for record in json_records:
                frac = record["renewable_fraction"]
                if frac not in seen_fracs:
                    chosen.append(record)
                    seen_fracs.add(frac)
                if len(chosen) >= plot_samples:
                    break
            if len(chosen) < plot_samples:
                for record in json_records:
                    if record not in chosen:
                        chosen.append(record)
                    if len(chosen) >= plot_samples:
                        break
            
            # Save a single combined 2x2 comparison panel per sample
            net_maps_dir = os.path.join(plot_dir, "network_maps")
            os.makedirs(net_maps_dir, exist_ok=True)
            for record in chosen:
                sid = record["sample_id"]
                plot_sample_network_comparison_panel(
                    case_name=case_name,
                    record=record,
                    output_path=os.path.join(net_maps_dir, f"{sid}_comparison.png"),
                    model_name=model_name,
                )

    if verbose:
        print(
            f"[OK] {model_name} {case_name}: {n_samples} samples, "
            f"MAE Vm={summary['mean_mae_vm_pu']:.6f} p.u., "
            f"MAE Va={summary['mean_mae_va_rad']:.6f} rad"
        )
        print(f"     JSONL -> {jsonl_path}")
        print(f"     CSV   -> {csv_path}")
    return summary, vm_pred_all, va_pred_all, vm_true_all, va_true_all, metadata


def main():
    parser = argparse.ArgumentParser(description="Export test-set voltage predictions to CSV/JSON/plots.")
    parser.add_argument("--case", "--cases", dest="case", default="all", help="case33, 118, or all")
    parser.add_argument("--models", default="all", help="Comma-separated model names or 'all'")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path (single model/case run)")
    parser.add_argument("--checkpoint-root", default=os.path.join(PROJECT_ROOT, "checkpoints"))
    parser.add_argument(
        "--use-final-sessions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the canonical final session per case (see FINAL_TRAINING_SESSIONS in script)",
    )
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "reports", "predictions"))
    parser.add_argument("--max-samples", type=int, default=None, help="Cap exported test samples (first N only; prefer --sample-strategy coverage)")
    parser.add_argument(
        "--sample-strategy",
        choices=["all", "coverage"],
        default="coverage",
        help="all=every test sample; coverage=one per renewable fraction plus topology contingencies",
    )
    parser.add_argument("--plot-samples", type=int, default=6, help="Network map plots (0 to skip)")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--verbose", action="store_true", help="Print detailed export details")
    args = parser.parse_args()

    config = load_training_config()
    batch_size = args.batch_size or config["data"].get("batch_size", 32)
    seq_len = config["data"].get("seq_len", 4)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cases = resolve_cases(args.case)
    models = resolve_models(args.models, args.checkpoint_root, cases)

    if args.verbose:
        print(f"Device: {device}")
        print(f"Cases: {', '.join(cases)}")
        print(f"Models: {', '.join(models)}")
        print(f"Output: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    summaries = []
    for case_name in cases:
        case_model_data = []
        for model_name in models:
            ckpt = get_checkpoint(
                model_name,
                case_name,
                args.checkpoint_root,
                args.checkpoint,
                args.use_final_sessions,
            )
            if not ckpt:
                if args.verbose:
                    print(f"[SKIP] No checkpoint for {model_name} on {case_name}")
                continue
            prep_case = os.path.join(PROJECT_ROOT, "data", "prep", case_name)
            if not os.path.isdir(prep_case):
                if args.verbose:
                    print(f"[SKIP] Missing prep data for {case_name}")
                continue
            try:
                summary, vm_pred, va_pred, vm_true, va_true, metadata = run_export(
                    model_name=model_name,
                    case_name=case_name,
                    ckpt_path=ckpt,
                    device=device,
                    output_dir=args.output_dir,
                    max_samples=args.max_samples,
                    sample_strategy=args.sample_strategy,
                    plot_samples=args.plot_samples,
                    skip_plots=args.skip_plots,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    verbose=args.verbose,
                )
                summaries.append(summary)
                case_model_data.append({
                    "model_name": model_name,
                    "vm_pred": vm_pred,
                    "va_pred": va_pred,
                    "vm_true": vm_true,
                    "va_true": va_true,
                    "metadata": metadata
                })
            except Exception as exc:
                print(f"[ERROR] {model_name} {case_name}: {exc}")

        # Generate grouped plots across models for this case
        if not args.skip_plots and case_model_data:
            case_plots_dir = os.path.join(args.output_dir, case_name)
            os.makedirs(case_plots_dir, exist_ok=True)

            # Divide into Group 1 (first 4 models) and Group 2 (remaining models)
            groups = [
                (case_model_data[:4], "group1"),
                (case_model_data[4:], "group2")
            ]

            for group_data, group_name in groups:
                if not group_data:
                    continue

                # Plot combined scatter
                plot_combined_pred_vs_true_scatter(
                    model_data=group_data,
                    output_path=os.path.join(case_plots_dir, f"{case_name}_scatter_plots_{group_name}.png"),
                    case_name=case_name
                )

                # Plot combined temporal MAE
                plot_combined_fraction_temporal_mae(
                    model_data=group_data,
                    output_path=os.path.join(case_plots_dir, f"{case_name}_temporal_mae_{group_name}.png"),
                    case_name=case_name
                )

    index_path = os.path.join(args.output_dir, "export_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    if args.verbose:
        print(f"\nWrote index -> {index_path}")


if __name__ == "__main__":
    main()
