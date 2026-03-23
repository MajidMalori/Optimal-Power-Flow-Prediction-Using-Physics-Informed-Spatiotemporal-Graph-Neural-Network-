import time
import sys
import os
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any
import shutil

from src.benchmarks.functions import BENCHMARKS
from src.optimizers.mosoa import MoSOA
from src.visualization.plot_mosoa import (
    plot_perturbation_ablation, 
    plot_perturbation_convergence
)


class MoSOAPerturbationVariant(MoSOA):
    """MoSOA variant that allows overriding the perturbation decay strategy."""
    def __init__(self, search_space: Dict[str, Any], seed: int = 42, 
                 pop_size: int = 30, strategy: str = 'exponential', **kwargs):
        super().__init__(search_space, seed, pop_size=pop_size, **kwargs)
        self.strategy = strategy

    def optimize(self, objective_fn, n_trials: int, verbose: bool = True) -> Dict[str, Any]:
        t_max = n_trials // self.pop_size
        it = 0
        self.history = []
        
        while it < t_max:
            for i in range(self.pop_size):
                params = {name: self.positions[i, j] for j, name in enumerate(self.param_names)}
                current_fitness = objective_fn(params)
                self.fitness[i] = current_fitness
                
                if current_fitness < self.p_best_fitness[i]:
                    self.p_best_fitness[i] = current_fitness
                    self.p_best_positions[i] = np.copy(self.positions[i])
                
                if current_fitness < self.g_best_fitness:
                    self.g_best_fitness = current_fitness
                    self.g_best_position = np.copy(self.positions[i])

            sigma = 1.0 + (np.std(self.fitness) / (np.mean(self.fitness) + 1e-6))
            a = self.f_c * (1 - (it / t_max))**sigma
            w = 0.95 - (it / t_max) * (0.95 - 0.35)
            
            if self.strategy == 'exponential':
                # Clip to prevent overflow in exp
                beta = np.exp(-5.0 * min(1.0, it / t_max))
            elif self.strategy == 'linear':
                beta = 1.0 - (it / t_max)
            elif self.strategy == 'cosine':
                beta = 0.5 * (1 + np.cos(np.pi * it / t_max))
            elif self.strategy == 'quadratic':
                beta = (1 - it / t_max)**2
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            new_positions = np.zeros_like(self.positions)
            for i in range(self.pop_size):
                rd = np.random.random()
                b = 2 * (a**2) * rd
                k = np.random.uniform(0, 2 * np.pi)
                radius = np.tanh(1 - it / t_max) * np.exp(k * 0.1)
                x = radius * np.cos(k)
                y = radius * np.sin(k)
                z = radius * k
                dist = np.abs(a * self.positions[i] + b * (self.g_best_position - self.positions[i]))
                p_attack = dist * x * y * z + self.g_best_position
                r1, r2 = np.random.random(), np.random.random()
                p_learned = (w * self.positions[i] + 
                             self.c1 * r1 * (self.g_best_position - self.positions[i]) + 
                             self.c2 * r2 * (self.p_best_positions[i] - self.positions[i]))
                noise = np.random.uniform(-1, 1, self.dim) * (beta * (self.g_best_position - self.positions[i]))
                new_positions[i] = self._normalize_position(p_learned + noise + (p_attack - p_learned) * (it/t_max))

            self.positions = new_positions
            self.history.append(self.g_best_fitness)
            it += 1

        best_params = {name: self.g_best_position[j] for j, name in enumerate(self.param_names)}
        return best_params, self.history


def run_strategy_benchmark(strategy: str, func_name: str, num_runs: int = 15, 
                            n_trials: int = 300, default_dim: int = 10, pop_size: int = 30, **mosoa_params):
    benchmark = BENCHMARKS[func_name]
    _obj_fn = benchmark['fn']
    bounds = benchmark['bounds']
    
    # Honor fixed dimensions
    dim = benchmark['dim'] if benchmark['dim'] is not None else default_dim
    
    if isinstance(bounds[0], list):
        search_space = {f'x_{i}': bounds[i] for i in range(dim)}
    else:
        search_space = {f'x_{i}': bounds for i in range(dim)}
    
    def obj_fn(params: Dict[str, Any]) -> float:
        x_array = np.array([params[f'x_{i}'] for i in range(dim)])
        return _obj_fn(x_array)

    best_fitnesses = []
    all_histories = []

    for _ in range(num_runs):
        opt = MoSOAPerturbationVariant(search_space=search_space, strategy=strategy, pop_size=pop_size, **mosoa_params)
        best_params, history = opt.optimize(obj_fn, n_trials=n_trials, verbose=False)
        best_fitnesses.append(obj_fn(best_params))
        all_histories.append(history)

    avg_history = np.mean(all_histories, axis=0)

    return {
        'Strategy': strategy,
        'Function': func_name,
        'Best': np.min(best_fitnesses),
        'Worst': np.max(best_fitnesses),
        'Mean': np.mean(best_fitnesses),
        'Std': np.std(best_fitnesses)
    }, avg_history

def main():
    # 1. Load Config
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "mosoa.yaml")
    config = {}
    # Load mosoa_params from config
    with open(config_path, 'r') as f:
        full_conf = yaml.safe_load(f)
        mosoa_params = full_conf.get('mosoa_params', {})
        config = full_conf.get('perturbation', {})
    
    num_runs = config.get('num_runs', 10)
    iterations = config.get('iterations', 200)
    pop_size = config.get('pop_size', 30)
    n_trials = iterations * pop_size
    
    # 2. Clear    # Restructure paths
    base_dir = "reports/mosoa/perturbation"
    if os.path.exists(base_dir):
        import shutil
        shutil.rmtree(base_dir)
        
    csv_dir = os.path.join(base_dir, "csv")
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    test_funcs = [f'F{i}' for i in range(1, 24)] 
    strategies = ['linear', 'cosine', 'quadratic', 'exponential']
    tasks = [(fn, strat) for fn in test_funcs for strat in strategies]
    
    
    all_results = []
    convergence_histories = {} # {fn_name: {strategy: history}}
    
    # Subset of functions for convergence plots - Expanded to include ALL for research grade grids
    conv_plot_subset = config.get('conv_plot_subset', test_funcs)
    if conv_plot_subset == "all":
        conv_plot_subset = test_funcs
    
    print() # Spacing from command
    for fn_name, strat in tqdm(tasks, desc="Perturbation Strategy Ablation", leave=True, dynamic_ncols=True):
        res, avg_hist = run_strategy_benchmark(strat, fn_name, num_runs=num_runs, n_trials=n_trials, pop_size=pop_size, **mosoa_params)
        all_results.append(res)
        
        if fn_name in conv_plot_subset:
            if fn_name not in convergence_histories:
                convergence_histories[fn_name] = {}
            convergence_histories[fn_name][strat] = avg_hist
            
    df = pd.DataFrame(all_results)
    print("\n\n======= PERTURBATION STRATEGY COMPARISON =======")
    print(df.to_string(index=False))
    print("================================================")
    
    # 3. Save Results
    df.to_csv(os.path.join(csv_dir, "perturbation_results.csv"), index=False)
    
    try:
        # Categorize functions for separate plotting
        unimodal = [f'F{i}' for i in range(1, 8)]
        multimodal = [f'F{i}' for i in range(8, 14)]
        fixed_dim = [f'F{i}' for i in range(14, 24)]
        
        # 4. Strategy Convergence Plots (Grouped by category - LINE GRAPHS ONLY AS REQUESTED)
        print("\nGenerating Research-Grade Convergence Grids (Line Plots)...")
        from src.visualization.plot_mosoa import plot_categorical_convergence
        
        # Unimodal: split F1-F4 and F5-F7
        uni_a = {fn: convergence_histories[fn] for fn in [f'F{i}' for i in range(1, 5)] if fn in convergence_histories}
        uni_b = {fn: convergence_histories[fn] for fn in [f'F{i}' for i in range(5, 8)] if fn in convergence_histories}
        if uni_a:
            plot_categorical_convergence(uni_a, "Perturbation Convergence: Unimodal (F1-F4)",
                                         os.path.join(plot_dir, "convergence_ablation_unimodal_F1_F4.png"), num_runs=num_runs)
        if uni_b:
            plot_categorical_convergence(uni_b, "Perturbation Convergence: Unimodal (F5-F7)",
                                         os.path.join(plot_dir, "convergence_ablation_unimodal_F5_F7.png"), num_runs=num_runs)

        # Multimodal: 6 functions fits nicely in one 2x3 grid
        multimodal_conv = {fn: convergence_histories[fn] for fn in multimodal if fn in convergence_histories}
        if multimodal_conv:
            plot_categorical_convergence(multimodal_conv, "Perturbation Convergence: Multimodal (F8-F13)",
                                         os.path.join(plot_dir, "convergence_ablation_multimodal.png"), num_runs=num_runs)
        
        # Fixed-dim: split F14-F19 and F20-F23
        fix_a = {fn: convergence_histories[fn] for fn in [f'F{i}' for i in range(14, 20)] if fn in convergence_histories}
        fix_b = {fn: convergence_histories[fn] for fn in [f'F{i}' for i in range(20, 24)] if fn in convergence_histories}
        if fix_a:
            plot_categorical_convergence(fix_a, "Perturbation Convergence: Fixed-Dim (F14-F19)",
                                         os.path.join(plot_dir, "convergence_ablation_fixed_dim_F14_F19.png"), num_runs=num_runs)
        if fix_b:
            plot_categorical_convergence(fix_b, "Perturbation Convergence: Fixed-Dim (F20-F23)",
                                         os.path.join(plot_dir, "convergence_ablation_fixed_dim_F20_F23.png"), num_runs=num_runs)
            
    except Exception as e:
        print(f"Visualization error: {e}")

if __name__ == "__main__":
    main()
