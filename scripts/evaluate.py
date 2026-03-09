#!/usr/bin/env python3
import os
import sys
import warnings

# Absolute silence for environment warnings
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_CONSOLE"] = "off"
os.environ["LIGHTNING_PYTORCH_DISABLE_TIP"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

warnings.filterwarnings("ignore") # Global ignore for all warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import time
import glob
import torch
import yaml
import argparse
import numpy as np
import logging
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models import MODEL_REGISTRY, PowerFlowDataModule, SPATIAL_MODELS, RECURRENT_MODELS
import pandapower as pp
from src.processing.topology import load_network
from src.visualization.plot_benchmarks import plot_benchmark_results

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

def get_latest_checkpoint(model_name, case_name):
    pattern = os.path.join(PROJECT_ROOT, "checkpoints", "**", f"{model_name}_{case_name}", "*.ckpt")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def run_evaluation(model_name, case_name, ckpt_path, device):
    print(f"Evaluating {model_name} on {case_name}")
    
    # 1. Load Data
    config_path = os.path.join(PROJECT_ROOT, "configs", "training.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize DataModule correctly
    data_dir = os.path.join(PROJECT_ROOT, "data", "03_processed")
    dm = PowerFlowDataModule(
        data_dir=data_dir,
        case_name=case_name,
        batch_size=config['data'].get("batch_size", 32),
        seq_len=1 if model_name in SPATIAL_MODELS else config['data'].get("seq_len", 1)
    )
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()

    # 2. Load Model
    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass.load_from_checkpoint(ckpt_path, strict=False)
    model.to(device)
    model.eval()

    # 3. Metrics accumulators
    mae_vm = []
    mae_va = []
    
    p_satisfied = 0
    q_satisfied = 0
    v_satisfied = 0
    s_satisfied = 0
    total_samples = 0
    total_nodes = 0
    total_branches = 0
    feasible_count = 0

    inference_times = []

    # 4. Evaluation Loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference", leave=False, ascii=True):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            x = batch["features"]
            targets = batch["targets"]
            topo_ids = batch["topology_ids"]

            # Handle Spatial vs Recurrent batch keys
            is_recurrent = model_name in RECURRENT_MODELS
            edge_index = batch["edge_index_seq"] if is_recurrent else batch["edge_index"]

            # Time inference
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            # Forward pass
            preds = model(x, edge_index)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            inference_times.append((end - start) / x.size(0))

            # MAE (VM is index 0 in preds, VA is index 1)
            targets_vm_va = targets[..., 8:10]
            err = torch.abs(preds - targets_vm_va)
            mae_vm.append(err[..., 0].mean().item())
            mae_va.append(err[..., 1].mean().item())

            # Constraints
            physics = model._get_physics_loss(batch)
            
            # For recurrent models, we evaluate physics on the final timestep of the sequence
            eval_topo_ids = topo_ids[:, -1] if is_recurrent else topo_ids
            
            res = physics.evaluate_constraints(
                vm_pred=preds[..., 0],
                va_pred=preds[..., 1],
                targets=targets,
                topology_ids=eval_topo_ids
            )

            p_satisfied += res["p_satisfied"]
            q_satisfied += res["q_satisfied"]
            v_satisfied += res["v_satisfied"]
            s_satisfied += res["s_satisfied"]
            total_nodes += res["total_p_q"]
            total_branches += res["total_s"]
            feasible_count += res["feasible_samples"]
            total_samples += res["total_samples"]

    # 5. Pandapower Baseline (Speed Factor)
    net = load_network(case_name)
    pp_times = []
    for _ in range(5):
        start = time.perf_counter()
        try:
            pp.runopp(net)
        except:
            pp.runpp(net)
        end = time.perf_counter()
        pp_times.append(end - start)
    
    avg_pp_time = np.mean(pp_times)
    avg_inf_time = np.mean(inference_times)
    speed_factor = avg_pp_time / avg_inf_time

    # 6. Report
    print(f"\n{'-'*50}")
    print(f"RESULTS: {model_name} on {case_name}")
    print(f"{'-'*50}")
    print(f"Accuracy:")
    print(f"  MAE (Voltage Mag):    {np.mean(mae_vm):.6f} p.u.")
    print(f"  MAE (Voltage Angle):  {np.mean(mae_va):.6f} rad")
    print(f"\nPhysical Feasibility:")
    print(f"  Overall Feasibility:  {feasible_count/total_samples*100:6.2f}%")
    print(f"  Constraint Sat. Rate:")
    print(f"    - Power P:          {p_satisfied/total_nodes*100:6.2f}%")
    print(f"    - Power Q:          {q_satisfied/total_nodes*100:6.2f}%")
    print(f"    - Voltage Limits:   {v_satisfied/total_nodes*100:6.2f}%")
    print(f"    - Branch Limits:    {s_satisfied/total_branches*100:6.2f}%")
    print(f"\nEfficiency:")
    print(f"  Avg Inference Time:   {avg_inf_time*1000:8.3f} ms/sample")
    print(f"  Avg Pandapower Time:  {avg_pp_time*1000:8.3f} ms/sample")
    print(f"  Speedup Factor:       {speed_factor:8.1f}x")
    print(f"{'-'*50}\n")

    return {
        "case": case_name,
        "model": model_name,
        "mae_vm": np.mean(mae_vm),
        "mae_va": np.mean(mae_va),
        "feasibility": feasible_count / total_samples,
        "p_sat": p_satisfied / total_nodes,
        "q_sat": q_satisfied / total_nodes,
        "v_sat": v_satisfied / total_nodes,
        "s_sat": s_satisfied / total_branches,
        "avg_inf_ms": avg_inf_time * 1000,
        "avg_pp_ms": avg_pp_time * 1000,
        "speedup": speed_factor
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="case33")
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.case.lower() == "all":
        cases = ["case33", "case57", "case118"]
    else:
        cases = [args.case if args.case.startswith("case") else f"case{args.case}"]

    model_list = list(MODEL_REGISTRY.keys()) if args.model.lower() == "all" else [args.model]

    results = []
    for case in cases:
        for model_name in model_list:
            ckpt = args.checkpoint if args.checkpoint else get_latest_checkpoint(model_name, case)
            if not ckpt:
                print(f"No checkpoint found for {model_name} on {case}. Skipping.")
                continue
            
            try:
                res = run_evaluation(model_name, case, ckpt, device)
                results.append(res)
            except Exception as e:
                print(f"Error evaluating {model_name} on {case}: {e}")
                import traceback
                traceback.print_exc()

    # Integrated Plotting
    if results:
        # Group by case for plotting
        case_results = {}
        for r in results:
            c = r['case']
            if c not in case_results:
                case_results[c] = []
            case_results[c].append(r)
        
        for case, res_list in case_results.items():
            case_dir = os.path.join(PROJECT_ROOT, "reports", "benchmarks", case)
            plot_benchmark_results(res_list, case, case_dir)

if __name__ == "__main__":
    main()
