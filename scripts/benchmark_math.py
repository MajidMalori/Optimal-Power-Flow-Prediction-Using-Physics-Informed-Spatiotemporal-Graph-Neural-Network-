"""
Stage 1: Full Mathematical Benchmark Runner (Section 3.1).
Compares MoSOA vs 7 competitor metaheuristics on F1-F23.
Competitors use peer-reviewed implementations from mealpy (Van Thieu, 2023).
"""
import time
import sys
import os
import warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, List
import shutil
import yaml

from src.benchmarks.functions import BENCHMARKS
from src.optimizers.mosoa import MoSOA
from src.optimizers.mealpy_wrapper import run_mealpy_optimizer, MEALPY_ALGORITHMS
from src.visualization.plot_mosoa import plot_mosoa_ranks, plot_convergence_curves


# ============================================================
# MoSOA runner (uses our custom implementation)
# ============================================================
def run_mosoa(func_name: str, num_runs: int, n_trials: int, pop_size: int, dim: int, mosoa_params: Dict[str, Any]):
    benchmark = BENCHMARKS[func_name]
    _obj_fn = benchmark['fn']
    bounds = benchmark['bounds']

    # Handle per-dim bounds
    if isinstance(bounds[0], list):
        search_space = {f'x_{i}': bounds[i] for i in range(dim)}
    else:
        search_space = {f'x_{i}': bounds for i in range(dim)}

    def obj_fn(params: Dict[str, Any]) -> float:
        x_array = np.array([params[f'x_{i}'] for i in range(dim)])
        return _obj_fn(x_array)

    fitness_results = []
    histories = []
    for run in range(num_runs):
        opt = MoSOA(search_space=search_space, seed=run + 42, pop_size=pop_size, **mosoa_params)
        res = opt.optimize(obj_fn, n_trials=n_trials, verbose=False)
        fitness_results.append(res['best_fitness'])
        histories.append(res['history'])
    
    # Average history across runs
    avg_history = np.mean(histories, axis=0).tolist()
    return fitness_results, avg_history


# ============================================================
# Mealpy runner (uses library implementations)
# ============================================================
def run_mealpy(algo_name: str, func_name: str, num_runs: int, n_trials: int, pop_size: int, dim: int):
    benchmark = BENCHMARKS[func_name]
    obj_fn = benchmark['fn']
    bounds = benchmark['bounds']

    # Handle per-dim bounds for fixed-dimension functions
    if isinstance(bounds[0], list):
        bounds_list = bounds
    else:
        bounds_list = bounds

    fitness_results = []
    histories = []
    for run in range(num_runs):
        res = run_mealpy_optimizer(
            algo_name=algo_name,
            obj_fn=obj_fn,
            bounds=bounds_list,
            dim=dim,
            n_trials=n_trials,
            pop_size=pop_size,
            seed=run + 42,
        )
        fitness_results.append(res['best_fitness'])
        histories.append(res['history'])
        
    avg_history = np.mean(histories, axis=0).tolist()
    return fitness_results, avg_history


# ============================================================
# Main
# ============================================================
def main():
    # 1. Load Config
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "mosoa_benchmarks.yaml")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f).get('mathematical', {})
    
    num_runs = config.get('num_runs', 10)
    iterations = config.get('iterations', 200)
    pop_size = config.get('pop_size', 30)
    n_trials = iterations * pop_size
    default_dim = config.get('default_dim', 30)
    
    # Load MoSOA specific params
    with open(config_path, 'r') as f:
        mosoa_params = yaml.safe_load(f).get('mosoa_params', {})
    
    # 2. Clear previous results to avoid stale files
    base_dir = "reports/mosoa/mathematical"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    all_algorithms = ['MoSOA'] + list(MEALPY_ALGORITHMS.keys())
    func_names = list(BENCHMARKS.keys())

    tasks = [(fn, algo) for fn in func_names for algo in all_algorithms]

    all_results = []
    convergence_histories = {} # {fn_name: {algo_name: history}}
    
    # Functions for which we want convergence curves
    conf_subset = config.get('plot_subset', 'all')
    if conf_subset == 'all':
        plot_subset = [f'F{i}' for i in range(1, 24)]
    else:
        plot_subset = conf_subset

    print() # Spacing from command
    for fn_name, algo_name in tqdm(tasks, desc="Mathematical Function Benchmarks", leave=True, dynamic_ncols=True):
        benchmark = BENCHMARKS[fn_name]
        dim = benchmark['dim'] if benchmark['dim'] is not None else default_dim

        start_time = time.time()

        if algo_name == 'MoSOA':
            fitnesses, avg_hist = run_mosoa(fn_name, num_runs, n_trials, pop_size, dim, mosoa_params)
        else:
            fitnesses, avg_hist = run_mealpy(algo_name, fn_name, num_runs, n_trials, pop_size, dim)

        elapsed = time.time() - start_time

        all_results.append({
            'Algorithm': algo_name,
            'Function': fn_name,
            'Best': np.min(fitnesses),
            'Worst': np.max(fitnesses),
            'Mean': np.mean(fitnesses),
            'Std': np.std(fitnesses),
            'Time (s)': round(elapsed, 3),
        })
        
        if fn_name in plot_subset:
            if fn_name not in convergence_histories:
                convergence_histories[fn_name] = {}
            convergence_histories[fn_name][algo_name] = avg_hist

    df = pd.DataFrame(all_results)
    
    # Restructure paths
    base_dir = "reports/mosoa/mathematical"
    csv_dir = os.path.join(base_dir, "csv")
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Categorize results into three tables
    unimodal_fns = [f'F{i}' for i in range(1, 8)]
    multimodal_fns = [f'F{i}' for i in range(8, 14)]
    fixed_dim_fns = [f'F{i}' for i in range(14, 24)]

    def save_category(fns, filename):
        cat_df = df[df['Function'].isin(fns)].copy()
        if not cat_df.empty:
            cat_df.to_csv(os.path.join(csv_dir, filename), index=False)

    save_category(unimodal_fns, "results_unimodal_F1_F7.csv")
    save_category(multimodal_fns, "results_multimodal_F8_F13.csv")
    save_category(fixed_dim_fns, "results_fixed_dim_F14_F23.csv")
    df.to_csv(os.path.join(csv_dir, "results_full_benchmark.csv"), index=False)

    # 2. Average Rank Summary
    rank_df = df.pivot(index='Function', columns='Algorithm', values='Mean')
    ranks = rank_df.rank(axis=1, method='min')
    avg_ranks = ranks.mean().sort_values()
    
    print("\n================ AVERAGE RANK SUMMARY ================")
    summary_list = []
    for algo, rank in avg_ranks.items():
        line = f"  {algo:<12}: {rank:.2f}"
        print(line)
        summary_list.append({'Algorithm': algo, 'Avg Rank': rank})
    print("======================================================")
    avg_ranks.to_csv(os.path.join(csv_dir, "average_ranks_summary.csv"))

    # 3. Visualization
    try:
        plot_mosoa_ranks(avg_ranks, os.path.join(plot_dir, "math_rank_comparison.png"))
        
        plot_tasks = list(convergence_histories.items())
        for fn_name, histories in tqdm(plot_tasks, desc="Generating Convergence Plots", leave=True, dynamic_ncols=True):
            benchmark = BENCHMARKS[fn_name]
            dim = benchmark['dim'] if benchmark['dim'] is not None else default_dim
            plot_convergence_curves(histories, fn_name, os.path.join(plot_dir, f"convergence_{fn_name}.png"), 
                                    num_runs=num_runs, dim=dim)
    except Exception as e:
        print(f"Visualization error: {e}")

if __name__ == "__main__":
    main()
