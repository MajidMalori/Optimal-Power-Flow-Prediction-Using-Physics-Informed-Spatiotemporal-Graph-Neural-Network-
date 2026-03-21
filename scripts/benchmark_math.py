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
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, Any, List
import shutil
import yaml

from src.benchmarks.functions import BENCHMARKS
from src.optimizers.mosoa import MoSOA
from src.optimizers.mealpy_wrapper import run_mealpy_optimizer, MEALPY_ALGORITHMS
from src.visualization.plot_mosoa import plot_mosoa_ranks, plot_categorical_convergence


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
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "mosoa.yaml")
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

    # 2. Average Rank Summary + Speed (4 tables: per-category + overall)
    avg_times = df.groupby('Algorithm')['Time (s)'].mean().round(2)
    
    def print_rank_table(sub_df, title):
        if sub_df.empty: return None
        rank_df = sub_df.pivot(index='Function', columns='Algorithm', values='Mean')
        ranks = rank_df.rank(axis=1, method='min')
        avg = ranks.mean().sort_values()
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
        for algo, rank in avg.items():
            speed = avg_times.get(algo, 0)
            print(f"  {algo:<12}: Rank {rank:.2f}  |  Avg Time: {speed:.2f}s")
        return avg
    
    df_uni = df[df['Function'].isin(unimodal_fns)]
    df_multi = df[df['Function'].isin(multimodal_fns)]
    df_fixed = df[df['Function'].isin(fixed_dim_fns)]
    
    print_rank_table(df_uni, "UNIMODAL (F1-F7)")
    print_rank_table(df_multi, "MULTIMODAL (F8-F13)")
    print_rank_table(df_fixed, "FIXED-DIMENSION (F14-F23)")
    
    rank_df = df.pivot(index='Function', columns='Algorithm', values='Mean')
    ranks = rank_df.rank(axis=1, method='min')
    avg_ranks = ranks.mean().sort_values()
    print_rank_table(df, "OVERALL AVERAGE (F1-F23)")
    print(f"{'=' * 60}")
    
    avg_ranks.to_csv(os.path.join(csv_dir, "average_ranks_summary.csv"))

    # 3. Visualization
    try:
        plot_mosoa_ranks(avg_ranks, os.path.join(plot_dir, "math_rank_comparison.png"))
        
        # Speed comparison bar chart
        from src.visualization.plot_mosoa import set_premium_mosoa_aesthetics
        set_premium_mosoa_aesthetics()
        
        speed_df = avg_times.reset_index()
        speed_df.columns = ['Algorithm', 'Avg Time (s)']
        speed_df = speed_df.sort_values('Avg Time (s)')
        
        plt.figure(figsize=(10, 6))
        colors = ['#E91E63' if a == 'MoSOA' else '#3498db' for a in speed_df['Algorithm']]
        sns.barplot(data=speed_df, x='Algorithm', y='Avg Time (s)', palette=colors)
        plt.title('Average Execution Time per Function (F1-F23)', fontweight='bold', pad=20)
        plt.ylabel('Avg Time (s)', fontweight='bold')
        plt.xlabel('Algorithm', fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "math_speed_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nGenerating Categorical Convergence Plots...")
        
        # Unimodal
        unimodal_hist = {fn: convergence_histories[fn] for fn in unimodal_fns if fn in convergence_histories}
        if unimodal_hist:
            plot_categorical_convergence(unimodal_hist, "Unimodal Functions (F1-F7)", 
                                         os.path.join(plot_dir, "convergence_unimodal.png"), num_runs=num_runs, default_dim=default_dim)
                                         
        # Multimodal
        multimodal_hist = {fn: convergence_histories[fn] for fn in multimodal_fns if fn in convergence_histories}
        if multimodal_hist:
            plot_categorical_convergence(multimodal_hist, "Multimodal Functions (F8-F13)", 
                                         os.path.join(plot_dir, "convergence_multimodal.png"), num_runs=num_runs, default_dim=default_dim)
                                         
        # Fixed-dimension Multimodal (split into two for readability)
        fixed_dim_a = [f'F{i}' for i in range(14, 20)]
        fixed_dim_b = [f'F{i}' for i in range(20, 24)]
        
        fixed_hist_a = {fn: convergence_histories[fn] for fn in fixed_dim_a if fn in convergence_histories}
        if fixed_hist_a:
            plot_categorical_convergence(fixed_hist_a, "Fixed-Dimension Multimodal (F14-F19)", 
                                         os.path.join(plot_dir, "convergence_fixed_dim_F14_F19.png"), num_runs=num_runs, default_dim=default_dim)
        
        fixed_hist_b = {fn: convergence_histories[fn] for fn in fixed_dim_b if fn in convergence_histories}
        if fixed_hist_b:
            plot_categorical_convergence(fixed_hist_b, "Fixed-Dimension Multimodal (F20-F23)", 
                                         os.path.join(plot_dir, "convergence_fixed_dim_F20_F23.png"), num_runs=num_runs, default_dim=default_dim)
                                         
    except Exception as e:
        print(f"Visualization error: {e}")

if __name__ == "__main__":
    main()
