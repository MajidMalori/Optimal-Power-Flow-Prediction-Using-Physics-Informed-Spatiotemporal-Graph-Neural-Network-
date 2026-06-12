#!/usr/bin/env python3
import argparse
import json
import os
import sys
import glob
import torch
import yaml
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.benchmarks.benchmark_dataset import load_states_jsonl
from src.benchmarks.feasibility_runner import run_feasibility_benchmark
from src.benchmarks.feasibility_runtime import run_all_methods_for_state_feasibility
from src.benchmarks.model_inference import predict_voltages_for_state
from src.models import get_model_registry
from src.visualization.plot_ws_feasibility import plot_ws_feasibility, plot_ws_feasibility_comparison

MODEL_PALETTE = {
    "StandardGCN": "#E63946",       # Vibrant Red
    "DynamicGCN": "#457B9D",        # Steel Blue
    "PIGCN": "#2A9D8F",             # Teal
    "PIGCLSTM": "#F4A261",          # Sandy Orange
    "PIGCGRU": "#9B5DE5",           # Purple
    "PIResnetGCLSTM": "#F15BB5",     # Pink
    "PIResnetGCGRU": "#00F5D4",      # Neon Turquoise
}

def main() -> None:
    parser = argparse.ArgumentParser(description="Warm-start feasibility benchmark")
    parser.add_argument("--case", type=str, required=True, help="case33 or 'all'")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--model", type=str, default="StandardGCN", help="Model name (e.g. StandardGCN) or 'all'")
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
        configured_models = list(MODEL_PALETTE.keys())

    if args.model.lower() == "all":
        models_to_run = configured_models
    else:
        models_to_run = [args.model]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for case in cases:
        states_path = os.path.join(PROJECT_ROOT, "data", "benchmark", case, "states.jsonl")
        if not os.path.exists(states_path):
            print(f"[WARNING] Benchmark states not found for {case} at: {states_path}. Skipping.")
            continue
            
        states = load_states_jsonl(states_path)[: args.max_samples]
        
        # Load normalization metadata
        norm_path = os.path.join(PROJECT_ROOT, "data", "prep", case, "normalization.json")
        with open(norm_path, 'r') as f:
            meta = json.load(f)

        all_records = []
        global_summary = {}

        for model_name in models_to_run:
            # Resolve model checkpoint
            pattern = os.path.join(PROJECT_ROOT, "checkpoints", "**", f"{model_name}_{case}", "*.ckpt")
            files = glob.glob(pattern, recursive=True)
            if not files:
                print(f"[WARNING] No checkpoint found for model {model_name} and case {case}. Skipping.")
                continue
            ckpt_path = max(files, key=os.path.getmtime)
            print(f"[INFO] Loaded checkpoint for {model_name}: {ckpt_path}")
            
            try:
                model = get_model_registry()[model_name].load_from_checkpoint(ckpt_path, strict=False)
                model.to(device)
                model.eval()
            except Exception as e:
                print(f"[ERROR] Failed to load model {model_name}: {e}. Skipping.")
                continue

            def model_run_fn(state, *args_fn, **kwargs_fn):
                pred_vm, pred_va = predict_voltages_for_state(state, model, model_name, device, meta)
                return run_all_methods_for_state_feasibility(state, *args_fn, pred_vm=pred_vm, pred_va=pred_va, **kwargs_fn)

            print(f"\n--- Running Feasibility Benchmark: {case.upper()} ({model_name}) ---")
            records, summary = run_feasibility_benchmark(states, model_run_fn)
            
            # Save single model feasibility logs
            out_dir = os.path.join(PROJECT_ROOT, "reports", "warmstart", "feasibility", case, model_name)
            os.makedirs(out_dir, exist_ok=True)
            
            summary["model_name"] = model_name
            with open(os.path.join(out_dir, f"{case}_feasibility_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            with open(os.path.join(out_dir, f"{case}_feasibility_records.json"), "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
                
            df_model = pd.DataFrame(records)
            df_model.to_csv(os.path.join(out_dir, f"{case}_feasibility_records.csv"), index=False)
            
            plot_ws_feasibility(records, summary, case_name=case, output_dir=out_dir)

            # Append model field to records for multi-model comparison
            for r in records:
                r["model"] = model_name
            all_records.extend(records)
            global_summary[model_name] = summary

        if not all_records:
            print(f"[ERROR] No models successfully evaluated for case {case}.")
            continue

        # If multiple models were run, save consolidated outputs and call global comparison plot
        if len(models_to_run) > 1:
            global_out_dir = os.path.join(PROJECT_ROOT, "reports", "warmstart", "feasibility", case, "all_comparison")
            os.makedirs(global_out_dir, exist_ok=True)

            df_all = pd.DataFrame(all_records)
            df_all.to_csv(os.path.join(global_out_dir, f"{case}_all_models_feasibility_records.csv"), index=False)

            plot_ws_feasibility_comparison(df_all, case, global_out_dir)
            print(f"[SUCCESS] Global feasibility comparison plots generated to: {global_out_dir}")

if __name__ == "__main__":
    main()
