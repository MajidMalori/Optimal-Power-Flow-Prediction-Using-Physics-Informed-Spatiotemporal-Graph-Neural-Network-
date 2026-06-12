import os
import sys
import argparse
import glob
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

PROJECT_ROOT = os.getcwd()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.benchmarks.benchmark_dataset import load_states_jsonl
from src.benchmarks.speed_runtime import run_all_methods_for_state
from src.benchmarks.benchmark_state import BenchmarkState
from src.benchmarks.model_inference import predict_voltages_for_state
from src.constants import FeatureIndices
from src.models import get_model_registry
from src.visualization.plot_ws_common import apply_ws_style, METHOD_COLORS, save_pub_figure

MODEL_PALETTE = {
    "StandardGCN": "#E63946",       # Vibrant Red
    "DynamicGCN": "#457B9D",        # Steel Blue
    "PIGCN": "#2A9D8F",             # Teal
    "PIGCLSTM": "#F4A261",          # Sandy Orange
    "PIGCGRU": "#9B5DE5",           # Purple
    "PIResnetGCLSTM": "#F15BB5",     # Pink
    "PIResnetGCGRU": "#00F5D4",      # Neon Turquoise
    "flat": "#ADB5BD",              # Neutral Gray
    "dc": "#6C757D"                 # Dark Gray
}

MODEL_MARKERS = {
    "StandardGCN": "^",
    "DynamicGCN": "s",
    "PIGCN": "o",
    "PIGCLSTM": "D",
    "PIGCGRU": "P",
    "PIResnetGCLSTM": "*",
    "PIResnetGCGRU": "v"
}

def main():
    parser = argparse.ArgumentParser(description="Stressed Grid Loadability & Solver Rescue Benchmark")
    parser.add_argument("--case", type=str, default="case33", help="Case name (e.g., case33 or 'all')")
    parser.add_argument("--model", type=str, default="StandardGCN", help="Model name or 'all'")
    parser.add_argument("--max-samples", type=int, default=10, help="Number of benchmark samples to evaluate")
    parser.add_argument("--max-iter", type=int, default=4, help="Strict real-time iteration budget")
    args = parser.parse_args()

    # Determine cases to evaluate
    if args.case.lower() == "all":
        benchmark_dir = os.path.join(PROJECT_ROOT, "data", "benchmark")
        if os.path.exists(benchmark_dir):
            cases = [d for d in os.listdir(benchmark_dir) if os.path.isdir(os.path.join(benchmark_dir, d)) and d.startswith("case")]
        else:
            cases = ["case33"]
    else:
        cases = [args.case if args.case.startswith("case") else f"case{args.case}"]

    # Determine models to evaluate
    config_path = os.path.join(PROJECT_ROOT, "configs", "warmstart.yaml")
    configured_models = []
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            configured_models = cfg.get("models", [])
    if not configured_models:
        configured_models = list(MODEL_PALETTE.keys())[:-2] # Exclude flat and dc

    if args.model.lower() == "all":
        models_to_run = configured_models
    else:
        models_to_run = [args.model]

    for case in cases:
        states_path = os.path.join(PROJECT_ROOT, "data", "benchmark", case, "states.jsonl")
        if not os.path.exists(states_path):
            print(f"[WARNING] Benchmark states not found for {case} at: {states_path}. Skipping.")
            continue
            
        states = load_states_jsonl(states_path)[:args.max_samples]
        
        # Load normalization metadata
        norm_path = os.path.join(PROJECT_ROOT, "data", "prep", case, "normalization.json")
        with open(norm_path, 'r') as f:
            meta = json.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_model_records = []

        for model_name in models_to_run:
            # Resolve model checkpoint
            pattern = os.path.join(PROJECT_ROOT, "checkpoints", "**", f"{model_name}_{case}", "*.ckpt")
            files = glob.glob(pattern, recursive=True)
            if not files:
                print(f"[WARNING] No checkpoint found for model {model_name} and case {case}. Skipping.")
                continue
            ckpt_path = max(files, key=os.path.getmtime)
            print(f"[INFO] Loaded {model_name} checkpoint: {ckpt_path}")
            
            try:
                model = get_model_registry()[model_name].load_from_checkpoint(ckpt_path, strict=False)
                model.to(device)
                model.eval()
            except Exception as e:
                print(f"[ERROR] Failed to load model {model_name}: {e}. Skipping.")
                continue

            print(f"\n--- Stressed Solver Rescue Benchmark: {case.upper()} ({model_name}) ---")
            print(f"Evaluating {len(states)} grid states under extreme scaling multipliers (1.0x to 3.5x)")
            print(f"Strict Iteration Budget: {args.max_iter} iterations")

            from tqdm import tqdm
            multipliers = np.arange(1.0, 3.6, 0.25)
            records = []

            desc = f"Stressed {case.upper()} ({model_name})"
            desc = f"{desc:<25}"
            total_scenarios = len(states) * len(multipliers)
            pbar = tqdm(total=total_scenarios, desc=desc,
                        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} scenarios",
                        unit="scenario")

            for state_idx, state in enumerate(states):
                for lam in multipliers:
                    # Scale active and reactive load power injections
                    stressed_features = [list(row) for row in state.features]
                    for row in stressed_features:
                        row[FeatureIndices.P_LOAD] *= lam
                        row[FeatureIndices.Q_LOAD] *= lam
                        
                    stressed_state = BenchmarkState(
                        sample_id=state.sample_id + f"_lam_{lam:.2f}",
                        case_name=state.case_name,
                        timestep=state.timestep,
                        renewable_fraction=state.renewable_fraction,
                        topology_id=state.topology_id,
                        features=stressed_features,
                        active_edges=state.active_edges,
                        metadata={**state.metadata, "load_multiplier": float(lam)},
                    )
                    
                    try:
                        pred_vm, pred_va = predict_voltages_for_state(stressed_state, model, model_name, device, meta)
                        results = run_all_methods_for_state(
                            stressed_state, max_iter=args.max_iter, tolerance=1e-5, pred_vm=pred_vm, pred_va=pred_va
                        )
                        
                        records.append({
                            "model": model_name,
                            "sample_id": state.sample_id,
                            "load_multiplier": lam,
                            "flat_converged": results["flat"].converged,
                            "dc_converged": results["dc"].converged,
                            "warmstart_converged": results["warmstart"].converged
                        })
                    except Exception as e:
                        records.append({
                            "model": model_name,
                            "sample_id": state.sample_id,
                            "load_multiplier": lam,
                            "flat_converged": False,
                            "dc_converged": False,
                            "warmstart_converged": False
                        })
                    pbar.update(1)
            pbar.close()
            
            # Save single model CSV logs
            model_output_dir = os.path.join(PROJECT_ROOT, "reports", "warmstart", "rescue", case, model_name)
            os.makedirs(model_output_dir, exist_ok=True)
            df_model = pd.DataFrame(records)
            df_model.to_csv(os.path.join(model_output_dir, f"{case}_stressed_loadability_records.csv"), index=False)
            all_model_records.extend(records)

        if not all_model_records:
            print(f"[ERROR] No models successfully evaluated for case {case}.")
            continue

        # Save global logs
        global_output_dir = os.path.join(PROJECT_ROOT, "reports", "warmstart", "rescue", case)
        os.makedirs(global_output_dir, exist_ok=True)
        df_all = pd.DataFrame(all_model_records)
        df_all.to_csv(os.path.join(global_output_dir, f"{case}_all_models_stressed_loadability_records.csv"), index=False)

        # Plot comparison curves
        apply_ws_style()
        fig, ax = plt.subplots(figsize=(10, 6.5))

        # Plot Flat Start and DC Start (take average across all model runs since they are identical)
        summary_flat_dc = df_all.groupby("load_multiplier")[["flat_converged", "dc_converged"]].mean() * 100.0
        summary_flat_dc = summary_flat_dc.reset_index()

        ax.plot(
            summary_flat_dc["load_multiplier"], 
            summary_flat_dc["flat_converged"], 
            color=MODEL_PALETTE["flat"], 
            label="Generic Flat Start (1.0 p.u., 0°)", 
            linestyle=":", 
            marker="o", 
            markersize=6, 
            alpha=0.8, 
            linewidth=1.5
        )
        ax.plot(
            summary_flat_dc["load_multiplier"], 
            summary_flat_dc["dc_converged"], 
            color=MODEL_PALETTE["dc"], 
            label="Linearized DC Start", 
            linestyle="--", 
            marker="s", 
            markersize=6, 
            alpha=0.8, 
            linewidth=1.5
        )

        # Plot each model's warmstart curve
        unique_models = df_all["model"].unique()
        best_model_name = None
        best_model_auc = -1.0

        for model_name in unique_models:
            df_m = df_all[df_all["model"] == model_name]
            summary_m = df_m.groupby("load_multiplier")["warmstart_converged"].mean() * 100.0
            summary_m = summary_m.reset_index()

            color = MODEL_PALETTE.get(model_name, "#1D3557")
            marker = MODEL_MARKERS.get(model_name, "d")
            label = f"Physics-Informed GNN ({model_name})"

            ax.plot(
                summary_m["load_multiplier"], 
                summary_m["warmstart_converged"], 
                color=color, 
                label=label, 
                linestyle="-", 
                marker=marker, 
                markersize=7, 
                linewidth=2.0
            )

            # Keep track of best model by area under the curve
            auc = summary_m["warmstart_converged"].sum()
            if auc > best_model_auc:
                best_model_auc = auc
                best_model_name = model_name

        # Highlight "Rescue Envelope" for the best performing GNN
        if best_model_name is not None:
            df_best = df_all[df_all["model"] == best_model_name]
            summary_best = df_best.groupby("load_multiplier")["warmstart_converged"].mean() * 100.0
            summary_best = summary_best.reset_index()

            ax.fill_between(
                summary_flat_dc["load_multiplier"], 
                summary_flat_dc["dc_converged"], 
                summary_best["warmstart_converged"], 
                color=MODEL_PALETTE.get(best_model_name, "#E63946"), 
                alpha=0.06, 
                label=f"Solver Rescue Envelope ({best_model_name})"
            )

        ax.set_ylim(-5, 105)
        ax.set_title(f"{case.upper()} - AC Power Flow Solver Loadability Limits under Extreme Stress", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Operational Load Multiplier (Relative to Grid Nominal Base)", fontsize=10, fontweight="bold")
        ax.set_ylabel("AC Power Flow Convergence Rate (%)", fontsize=10, fontweight="bold")
        ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

        sns.despine(ax=ax)
        
        # Save consolidated plot
        if len(models_to_run) > 1:
            save_pub_figure(fig, global_output_dir, f"{case}_all_models_stressed_loadability_rescue_trajectory")
            print(f"[SUCCESS] Stressed loadability trajectory plot saved to: {global_output_dir}/{case}_all_models_stressed_loadability_rescue_trajectory.png")
            
            # Save single-model copy if it was requested specifically
            for model_name in unique_models:
                single_dir = os.path.join(PROJECT_ROOT, "reports", "warmstart", "rescue", case, model_name)
                # Filter df_all for that model
                df_single = df_all[(df_all["model"] == model_name)]
                summary_single = df_single.groupby("load_multiplier")[["flat_converged", "dc_converged", "warmstart_converged"]].mean() * 100.0
                summary_single = summary_single.reset_index()

                fig_s, ax_s = plt.subplots(figsize=(10, 6.0))
                ax_s.plot(summary_single["load_multiplier"], summary_single["flat_converged"], color=MODEL_PALETTE["flat"], label="Generic Flat Start (1.0 p.u., 0°)", linestyle=":", marker="o", markersize=6, alpha=0.8, linewidth=1.5)
                ax_s.plot(summary_single["load_multiplier"], summary_single["dc_converged"], color=MODEL_PALETTE["dc"], label="Linearized DC Start", linestyle="--", marker="s", markersize=6, alpha=0.8, linewidth=1.5)
                ax_s.plot(summary_single["load_multiplier"], summary_single["warmstart_converged"], color=MODEL_PALETTE.get(model_name, "#E63946"), label=f"Physics-Informed GNN ({model_name})", linestyle="-", marker=MODEL_MARKERS.get(model_name, "^"), markersize=8, linewidth=2.5)
                ax_s.fill_between(summary_single["load_multiplier"], summary_single["dc_converged"], summary_single["warmstart_converged"], color=MODEL_PALETTE.get(model_name, "#E63946"), alpha=0.08, label="Solver Rescue Envelope")
                ax_s.set_ylim(-5, 105)
                ax_s.set_title(f"{case.upper()} - AC Power Flow Solver Loadability Limits under Extreme Stress", fontsize=12, fontweight="bold", pad=12)
                ax_s.set_xlabel("Operational Load Multiplier (Relative to Grid Nominal Base)", fontsize=10, fontweight="bold")
                ax_s.set_ylabel("AC Power Flow Convergence Rate (%)", fontsize=10, fontweight="bold")
                ax_s.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none")
                ax_s.grid(True, which="both", linestyle=":", alpha=0.5)
                sns.despine(ax=ax_s)
                save_pub_figure(fig_s, single_dir, f"{case}_stressed_loadability_rescue_trajectory")
                plt.close(fig_s)
        else:
            single_dir = os.path.join(PROJECT_ROOT, "reports", "warmstart", "rescue", case, models_to_run[0])
            save_pub_figure(fig, single_dir, f"{case}_stressed_loadability_rescue_trajectory")
            print(f"[SUCCESS] Stressed loadability trajectory plot saved to: {single_dir}/{case}_stressed_loadability_rescue_trajectory.png")

        plt.close(fig)

if __name__ == "__main__":
    main()
