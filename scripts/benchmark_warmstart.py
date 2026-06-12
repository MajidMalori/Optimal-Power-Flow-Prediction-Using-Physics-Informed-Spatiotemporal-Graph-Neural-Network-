#!/usr/bin/env python3
import os
import sys
import yaml
import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import pandapower as pp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models import PowerFlowDataModule, get_model_registry

MODEL_REGISTRY = get_model_registry()
from src.constants import TargetIndices, FeatureIndices
from src.processing.topology import load_network
from src.benchmarks.warm_start_evaluator import WarmStartEvaluator
from src.visualization.plot_warmstart import plot_warmstart_metrics, plot_cross_case_scaling
from scripts.preprocess_data import denormalize_predictions
import glob

def get_latest_checkpoint(model_name, case_name):
    pattern = os.path.join(PROJECT_ROOT, "checkpoints", "**", f"{model_name}_{case_name}", "*.ckpt")
    files = glob.glob(pattern, recursive=True)
    return max(files, key=os.path.getmtime) if files else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="case33")
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--samples", type=int, default=None, help="Overrides max_samples in config if provided")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config_path = os.path.join(PROJECT_ROOT, "configs", "warmstart.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if args.case.lower() == "all":
        cases = config['evaluation']['benchmark']['cases']
    else:
        cases = [args.case if args.case.startswith("case") else f"case{args.case}"]
        
    model_list = config['models'] if args.model.lower() == "all" else [args.model]

    global_results = []

    for case in cases:
        print(f"\nWarm-Start Benchmark: {case}")
        output_dir = os.path.join(PROJECT_ROOT, "reports", "benchmarks", "warm_start", case)
        
        # Ensure a clean start by clearing the case-specific report directory
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        net = load_network(case)
        
        # Warm-up run to eliminate first-call overhead (Numba compilation, etc)
        try:
            pp.runpp(net, algorithm="nr", init="flat", max_iteration=5)
        except Exception:
            pass
            
        bench_cfg = config['evaluation']['benchmark']
        evaluator = WarmStartEvaluator(
            net=net,
            case_name=case,
            max_iter=bench_cfg.get('max_iterations', 100),
            tolerance=float(bench_cfg.get('tolerance', 1e-5))
        )
        
        dm = PowerFlowDataModule(
            data_dir=os.path.join(PROJECT_ROOT, "data", "prep"),
            case_name=case,
            batch_size=config['data']['batch_size'],
            seq_len=config['data']['seq_len']
        )
        dm.setup(stage="test")
        test_loader = dm.test_dataloader()
        
        all_results = []
        
        desc = f"Evaluating {case}"
        desc = f"{desc:<25}"
        pbar = tqdm(model_list, desc=desc, leave=True,
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} models",
                    unit="model")
        for model_name in pbar:
            ckpt = get_latest_checkpoint(model_name, case)
            if not ckpt:
                continue
                
            model = MODEL_REGISTRY[model_name].load_from_checkpoint(ckpt, strict=False)
            model.to(device)
            model.eval()
            
            # Limit samples for speed (since this is sequential per-sample solver evaluation)
            MAX_SAMPLES = args.samples if args.samples is not None else bench_cfg.get('max_samples', 50)
            sample_count = 0
            
            with torch.no_grad():
                batch_size = dm.batch_size
                total_batches = min(len(test_loader), int(np.ceil(MAX_SAMPLES / batch_size)))
                desc_batch = f"  {model_name}"
                desc_batch = f"{desc_batch:<25}"
                for batch in tqdm(test_loader, desc=desc_batch, total=total_batches, leave=False,
                                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} batches",
                                  unit="batch"):
                    if sample_count >= MAX_SAMPLES:
                        break
                        
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    x = batch["features"]
                    targets = batch["targets"]
                    edge_index = batch.get("edge_index_seq", batch.get("edge_index"))
                    
                    edges_tensor = edge_index
                    if edge_index is not None:
                        while isinstance(edges_tensor, list):
                            edges_tensor = edges_tensor[-1]
                        if torch.is_tensor(edges_tensor):
                            while edges_tensor.dim() > 2:
                                edges_tensor = edges_tensor[0, -1] if edges_tensor.shape[1] > 1 else edges_tensor[0]
                            if edges_tensor.shape[0] != 2 and edges_tensor.shape[1] == 2:
                                edges_tensor = edges_tensor.t()
                    
                    # Recurrent models require [B, Seq, N, F]
                    is_recurrent = any(m in model_name for m in ["LSTM", "GRU"])
                    if is_recurrent and x.dim() == 3:
                         x = x.unsqueeze(1) # [B, 1, N, F]
                         # Ensure recurrent models receive a nested list for time iteration if they didn't get one
                         if torch.is_tensor(edge_index):
                             model_edge_index = [[edge_index] for _ in range(x.shape[0])]
                         elif isinstance(edge_index, list) and torch.is_tensor(edge_index[0]):
                             model_edge_index = [[edge] for edge in edge_index]
                         else:
                             model_edge_index = edge_index
                    else:
                         model_edge_index = edge_index

                    preds = model(x, model_edge_index)
                    
                    # Feature extraction (handles whether a sequence dimension exists or not)
                    if x.dim() == 4: # [B, Seq, N, F]
                        p_load = x[0, -1, :, FeatureIndices.P_LOAD].cpu().numpy()
                        q_load = x[0, -1, :, FeatureIndices.Q_LOAD].cpu().numpy()
                        p_gen = x[0, -1, :, FeatureIndices.P_CONV].cpu().numpy()
                        p_ren = x[0, -1, :, FeatureIndices.P_REN].cpu().numpy()
                        q_ren = x[0, -1, :, FeatureIndices.Q_REN].cpu().numpy()
                    else:
                        p_load = x[0, :, FeatureIndices.P_LOAD].cpu().numpy()
                        q_load = x[0, :, FeatureIndices.Q_LOAD].cpu().numpy()
                        p_gen = x[0, :, FeatureIndices.P_CONV].cpu().numpy()
                        p_ren = x[0, :, FeatureIndices.P_REN].cpu().numpy()
                        q_ren = x[0, :, FeatureIndices.Q_REN].cpu().numpy()
                        
                    # Get bus-index ordered predictions to ensure alignment with pandapower
                    if preds.dim() == 4:
                        pred_vm = preds[0, -1, :, 0].cpu().numpy()
                        pred_va = preds[0, -1, :, 1].cpu().numpy()
                    else:
                        pred_vm = preds[0, :, 0].cpu().numpy()
                        pred_va = preds[0, :, 1].cpu().numpy()
                        
                    if targets.dim() == 4:
                        target_vm = targets[0, -1, :, TargetIndices.VM].cpu().numpy()
                        target_va = targets[0, -1, :, TargetIndices.VA].cpu().numpy()
                    else:
                        target_vm = targets[0, :, TargetIndices.VM].cpu().numpy()
                        target_va = targets[0, :, TargetIndices.VA].cpu().numpy()
                    
                    
                    # Denormalize predictions back to physical parameters so the solver doesn't crash on 0.0 p.u voltages
                    norm_path = os.path.join(PROJECT_ROOT, "data", "prep", case, "normalization.json")
                    with open(norm_path, 'r') as f:
                         meta = json.load(f)
                    
                    mock_full = np.zeros((x.shape[1] if x.dim() == 3 else x.shape[2], 10))
                    mock_full[..., TargetIndices.VM] = pred_vm
                    mock_full[..., TargetIndices.VA] = pred_va
                    denorm = denormalize_predictions(mock_full[np.newaxis, ...], meta)[0]
                    
                    # Also denorm targets to ensure valid metric comparisons
                    mock_target = np.zeros_like(mock_full)
                    mock_target[..., TargetIndices.VM] = target_vm
                    mock_target[..., TargetIndices.VA] = target_va
                    denorm_t = denormalize_predictions(mock_target[np.newaxis, ...], meta)[0]
                    
                    pred_vm_phys = denorm[..., TargetIndices.VM]
                    pred_va_phys = denorm[..., TargetIndices.VA]
                    target_vm_phys = denorm_t[..., TargetIndices.VM]
                    target_va_phys = denorm_t[..., TargetIndices.VA]

                    active_edges = None
                    if edges_tensor is not None and torch.is_tensor(edges_tensor):
                         if edges_tensor.dim() == 2 and edges_tensor.shape[0] == 2:
                              u_arr = edges_tensor[0].cpu().numpy()
                              v_arr = edges_tensor[1].cpu().numpy()
                              active_edges = set(zip(u_arr, v_arr))
                    
                    try:
                        res = evaluator.evaluate_sample(
                            p_load=p_load, q_load=q_load, 
                            p_gen=p_gen, p_ren=p_ren, q_ren=q_ren, active_edges=active_edges,
                            pred_vm=pred_vm_phys, pred_va=pred_va_phys,
                            target_vm=target_vm_phys, target_va=target_va_phys
                        )
                        
                        # Map internal keys to concise professional labels
                        METHOD_MAP = {
                            "FLAT": "Generic Flat Start (1.0 p.u., 0°)",
                            "DC": "Linearized DC Start",
                            "RESULTS": "Physics-Informed GNN"
                        }
                        
                        for init_method, metrics in res.items():
                            all_results.append({
                                "Case": case,
                                "Model": model_name,
                                "InitMethod": METHOD_MAP.get(init_method.upper(), init_method.upper()),
                                "Time_ms": metrics['time_ms'],
                                "Iterations": metrics['iterations'],
                                "MAE_VM": metrics['mae_vm'],
                                "Success": metrics['success']
                            })
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"Evaluation error: {e}")
                        
                    sample_count += 1
                    
        if all_results:
            df = pd.DataFrame(all_results)
            # Filter failed solves
            df = df[df['Success'] == True]
            
            # Create a summary table by averaging across samples
            df_summary = df.groupby(["Model", "InitMethod"]).agg({
                "Time_ms": "mean",
                "Iterations": "mean",
                "MAE_VM": "mean"
            }).reset_index()
            
            print("\n" + "="*80)
            print(f"WARM-START BENCHMARK SUMMARY: {case.upper()}")
            print("="*80)
            # Formatting to make it beautiful
            print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print("="*80 + "\n")
            
            csv_path = os.path.join(output_dir, f"{case}_warmstart_metrics.csv")
            df.to_csv(csv_path, index=False)
            
            # Save the summary table to CSV as well
            summary_csv_path = os.path.join(output_dir, f"{case}_warmstart_summary.csv")
            df_summary.to_csv(summary_csv_path, index=False)
            
            plot_warmstart_metrics(df, case, output_dir)
            print(f"[{case}] Detailed full-sample results and plots saved to {output_dir}")
            
            global_results.extend(all_results)

    if global_results:
        global_df = pd.DataFrame(global_results)
        cross_case_dir = os.path.join(PROJECT_ROOT, "reports", "benchmarks", "warm_start")
        plot_cross_case_scaling(global_df, cross_case_dir)

if __name__ == "__main__":
    main()
