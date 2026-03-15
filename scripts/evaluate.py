#!/usr/bin/env python3
import os
import sys
import shutil
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
import warnings
from tqdm import tqdm

# Silence font-related warnings before they trigger
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*findfont: Generic family.*")
warnings.filterwarnings("ignore", message=".*findfont: Font family.*")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models import MODEL_REGISTRY, PowerFlowDataModule, SPATIAL_MODELS, RECURRENT_MODELS
from src.constants import TargetIndices
import pandapower as pp
from src.processing.topology import load_network
from src.visualization.plot_benchmarks import plot_benchmark_results

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pandapower").setLevel(logging.ERROR)
logging.getLogger("pypower").setLevel(logging.ERROR)

def get_latest_checkpoint(model_name, case_name):
    pattern = os.path.join(PROJECT_ROOT, "checkpoints", "**", f"{model_name}_{case_name}", "*.ckpt")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def run_evaluation(model_name, case_name, ckpt_path, device):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = os.path.join(PROJECT_ROOT, "data", "prep")
    dm = PowerFlowDataModule(
        data_dir=data_dir,
        case_name=case_name,
        batch_size=config['data'].get("batch_size", 32),
        seq_len=1 if model_name in SPATIAL_MODELS else config['data'].get("seq_len", 1)
    )
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()

    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass.load_from_checkpoint(ckpt_path, strict=False)
    model.to(device)
    model.eval()

    # 3. Metrics accumulators
    mae_vm = []
    mae_va = []
    mse_vm = []
    mse_va = []
    
    p_satisfied = 0
    q_satisfied = 0
    v_satisfied = 0
    s_satisfied = 0
    total_samples = 0
    total_nodes = 0
    total_branches = 0
    feasible_count = 0

    inference_times = []
    

    with torch.no_grad():
        for batch in test_loader:
            # Match device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            x = batch["features"]
            targets = batch["targets"]
            topo_ids = batch["topology_ids"]

            is_recurrent = model_name in RECURRENT_MODELS
            edge_index = batch["edge_index_seq"] if is_recurrent else batch["edge_index"]

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            preds = model(x, edge_index)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            inference_times.append((end - start) / x.size(0))

            from src.constants import TargetIndices
            targets_vm = targets[..., TargetIndices.VM]
            targets_va = targets[..., TargetIndices.VA]
            
            err_vm = torch.abs(preds[..., 0] - targets_vm)
            err_va = torch.abs(preds[..., 1] - targets_va)
            
            mae_vm.append(err_vm.mean().item())
            mae_va.append(err_va.mean().item())
            
            mse_vm.append((err_vm**2).mean().item())
            mse_va.append((err_va**2).mean().item())

            physics = model._get_physics_loss(batch)
            
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

    avg_inf_time = np.mean(inference_times)
    
    # No verbose per-model reporting here; we sum it all up at the end.

    return {
        "case": case_name,
        "model": model_name,
        "mae_vm": np.mean(mae_vm),
        "mae_va": np.mean(mae_va),
        "mse_vm": np.mean(mse_vm),
        "mse_va": np.mean(mse_va),
        "feasibility": feasible_count / total_samples,
        "p_sat": p_satisfied / total_nodes,
        "q_sat": q_satisfied / total_nodes,
        "v_sat": v_satisfied / total_nodes,
        "s_sat": s_satisfied / total_branches,
        "avg_inf_ms": avg_inf_time * 1000
    }

def run_solver_benchmark(case_name, config):
    """Run classical solvers once per case for speed/accuracy reference."""
    net = load_network(case_name)
    eval_cfg = config.get("evaluation", {}).get("benchmark", {})
    configured_solvers = eval_cfg.get("solvers", ["nr", "iwamoto_nr", "gs", "bfsw"])
    solver_trials = eval_cfg.get("solver_trials", 3)
    
    is_radial = case_name == "case33"
    solvers = [s for s in configured_solvers if (s != "bfsw" or is_radial)]
    
    solver_speeds = {}
    solver_vm = {}
    solver_va = {}
    
    devnull = open(os.devnull, 'w')
    for alg in solvers:
        times = []
        vm_result, va_result = None, None
        for _ in range(solver_trials):
            try:
                old_stdout = sys.stdout
                sys.stdout = devnull
                start = time.perf_counter()
                pp.runpp(net, algorithm=alg)
                end = time.perf_counter()
                sys.stdout = old_stdout
                times.append(end - start)
                vm_result = net.res_bus.vm_pu.values.copy()
                va_result = np.deg2rad(net.res_bus.va_degree.values.copy())
            except Exception:
                sys.stdout = old_stdout
        
        if times:
            solver_speeds[alg] = np.mean(times) * 1000  # ms
            solver_vm[alg] = vm_result
            solver_va[alg] = va_result
    devnull.close()
    
    solver_accuracy = {}
    if "nr" in solver_vm:
        nr_vm, nr_va = solver_vm["nr"], solver_va["nr"]
        for alg in solvers:
            if alg == "nr" or alg not in solver_vm: continue
            mae_vm = np.mean(np.abs(solver_vm[alg] - nr_vm))
            mae_va = np.mean(np.abs(solver_va[alg] - nr_va))
            solver_accuracy[alg] = {"mae_vm": mae_vm, "mae_va": mae_va}
            
    return solver_speeds, solver_accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="case33")
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join(PROJECT_ROOT, "configs", "training.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if args.case.lower() == "all":
        cases = ["case33", "case57", "case118"]
    else:
        cases = [args.case if args.case.startswith("case") else f"case{args.case}"]

    model_list = list(MODEL_REGISTRY.keys()) if args.model.lower() == "all" else [args.model]

    results = []
    
    for case in cases:
        print(f"\nEvaluation: {case}")
        print("-" * 40)
        
        # Reports go to reports/evaluation/standard/[case]
        benchmark_dir = os.path.join(PROJECT_ROOT, "reports", "evaluation", "standard", case)
        if os.path.exists(benchmark_dir):
            shutil.rmtree(benchmark_dir)
        os.makedirs(benchmark_dir, exist_ok=True)
        
        csv_dir = os.path.join(benchmark_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        
        print(f"Running benchmarks for {case}...")
        solver_speeds, solver_accuracy = run_solver_benchmark(case, config)
        
        pbar = tqdm(model_list, desc=f"Evaluating {case}", leave=True, dynamic_ncols=True)
        
        for model_name in pbar:
            pbar.set_postfix_str(f"Processing {model_name}... ")
            ckpt = args.checkpoint if args.checkpoint else get_latest_checkpoint(model_name, case)
            if not ckpt:
                continue
            try:
                res = run_evaluation(model_name, case, ckpt, device)
                res["solver_speeds"] = solver_speeds
                res["solver_accuracy"] = solver_accuracy
                # Calculate speedup vs NR
                nr_ms = solver_speeds.get("nr", 1.0)
                res["speedup"] = nr_ms / res["avg_inf_ms"]
                
                results.append(res)
            except Exception as e:
                pass # Silently continue on errors
                
        # Final Tabular Summary Report for the current case
        case_res = [r for r in results if r['case'] == case]
        if case_res:
            # Solver speed reference table
            first = case_res[0]
            solver_names = {"nr": "Newton-Raphson", "iwamoto_nr": "NR+Iwamoto", "gs": "Gauss-Seidel", "bfsw": "Backward/Fwd"}
            
            print("\n" + "="*80)
            print("SOLVER SPEED REFERENCE")
            print("="*80)
            for alg, ms in first['solver_speeds'].items():
                print(f"  {solver_names.get(alg, alg):<20}: {ms:8.3f} ms/sample")
            print("")
            
            # GNN Benchmark Table
            # Model Performance Comparison
            print("\nModel Performance Comparison")
            print("-" * 30)
            
            # Build header with per-solver speedups
            solver_keys = list(first['solver_speeds'].keys())
            solver_hdrs = [f"vs {solver_names.get(s, s)[:6]}" for s in solver_keys]
            
            header = f"{'Model':<18} | {'MAE vm/va':<15} | {'MSE vm/va':<15} | {'PQ/V/S Sat %':<15}"
            for sh in solver_hdrs:
                header += f" | {sh:<8}"
            print(header)
            print("-" * len(header))
            
            for res in case_res:
                row = f"{res['model']:<18} | "
                # Compact metrics to fit more
                row += f"{res['mae_vm']:.3f}/{res['mae_va']:.3f} | "
                row += f"{res['mse_vm']:.2e}/{res['mse_va']:.2e} | "
                
                avg_pq = (res['p_sat'] + res['q_sat']) / 2.0
                row += f"{avg_pq*100:4.1f}/{res['v_sat']*100:4.1f}/{res['s_sat']*100:4.1f} | "
                
                for alg in solver_keys:
                    spd_ms = res['solver_speeds'].get(alg, 1.0)
                    sp = spd_ms / res['avg_inf_ms']
                    row += f" {sp:7.1f}x |"
                print(row)
            
            # Solver accuracy vs NR (if available)
            if first.get('solver_accuracy'):
                print("\n" + "="*80)
                print("SOLVER ACCURACY vs NR GROUND TRUTH")
                print("="*80)
                header2 = f"{'Solver':<20} | {'MAE VM (p.u.)':<14} | {'MAE VA (rad)':<14}"
                print(header2)
                print("-" * len(header2))
                for alg, acc in first['solver_accuracy'].items():
                    print(f"{solver_names.get(alg, alg):<20} | {acc['mae_vm']:14.10f} | {acc['mae_va']:14.10f}")
                # Add GNN models for comparison
                for res in case_res:
                    print(f"{res['model']:<20} | {res['mae_vm']:14.10f} | {res['mae_va']:14.10f}")
            
            # Save to CSV
            import pandas as pd
            
            # 1. GNN Benchmark Summary CSV
            gnn_summary_data = []
            for res in case_res:
                row_data = {
                    "Model": res['model'],
                    "MAE_VM": res['mae_vm'],
                    "MAE_VA": res['mae_va'],
                    "MSE_VM": res['mse_vm'],
                    "MSE_VA": res['mse_va'],
                    "P_Sat_pct": res['p_sat'] * 100,
                    "Q_Sat_pct": res['q_sat'] * 100,
                    "V_Sat_pct": res['v_sat'] * 100,
                    "S_Sat_pct": res['s_sat'] * 100,
                    "Avg_Inf_ms": res['avg_inf_ms']
                }
                for alg in solver_keys:
                    spd_ms = res['solver_speeds'].get(alg, 1.0)
                    sp = spd_ms / res['avg_inf_ms']
                    row_data[f"Speedup_vs_{alg}"] = sp
                gnn_summary_data.append(row_data)
            
            df_gnn = pd.DataFrame(gnn_summary_data)
            df_gnn.to_csv(os.path.join(csv_dir, f"{case}_gnn_benchmark.csv"), index=False)
            
            # 2. Solver Accuracy CSV
            if first.get('solver_accuracy'):
                acc_data = []
                for alg, acc in first['solver_accuracy'].items():
                    acc_data.append({
                        "Solver_or_Model": solver_names.get(alg, alg),
                        "Type": "Classical Solver",
                        "MAE_VM": acc['mae_vm'],
                        "MAE_VA": acc['mae_va']
                    })
                for res in case_res:
                    acc_data.append({
                        "Solver_or_Model": res['model'],
                        "Type": "GNN",
                        "MAE_VM": res['mae_vm'],
                        "MAE_VA": res['mae_va']
                    })
                df_acc = pd.DataFrame(acc_data)
                df_acc.to_csv(os.path.join(csv_dir, f"{case}_solver_accuracy.csv"), index=False)
            
            # Plotting
            plot_benchmark_results(case_res, case, benchmark_dir)

if __name__ == "__main__":
    main()
