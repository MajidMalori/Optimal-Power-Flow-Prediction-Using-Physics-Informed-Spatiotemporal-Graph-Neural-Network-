import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, Any, List
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.benchmarks.functions import BENCHMARKS
from src.optimizers.mosoa import MoSOA
from src.visualization.plot_mosoa import set_premium_mosoa_aesthetics

def run_sensitivity_test(fn_name: str, param_name: str, values: List[float], 
                         num_runs: int = 5, iterations: int = 200):
    benchmark = BENCHMARKS[fn_name]
    _obj_fn = benchmark['fn']
    bounds = benchmark['bounds']
    dim = benchmark['dim'] if benchmark['dim'] is not None else 30
    
    if isinstance(bounds[0], list):
        search_space = {f'x_{i}': bounds[i] for i in range(dim)}
    else:
        search_space = {f'x_{i}': bounds for i in range(dim)}
    
    def obj_fn(params: Dict[str, Any]) -> float:
        x_array = np.array([params[f'x_{i}'] for i in range(dim)])
        return _obj_fn(x_array)

    results = []
    
    for val in values:
        best_fits = []
        for _ in range(num_runs):
            params = {param_name: val}
            opt = MoSOA(search_space, **params)
            res = opt.optimize(obj_fn, n_trials=iterations*30, verbose=False)
            best_fits.append(res['best_fitness'])
        
        results.append({
            'Value': val,
            'Mean': np.mean(best_fits),
            'Std': np.std(best_fits)
        })
    
    return pd.DataFrame(results)

def main():
    out_dir = "reports/mosoa/sensitivity"
    os.makedirs(out_dir, exist_ok=True)
    
    # Study 1: Convergence Factor f_c
    f_c_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    test_funcs = ['F1', 'F9', 'F21'] # Representative ones
    
    set_premium_mosoa_aesthetics()
    
    for fn in test_funcs:
        print(f"Testing f_c sensitivity on {fn}...")
        df = run_sensitivity_test(fn, 'f_c', f_c_values)
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Value', y='Mean', marker='o', linewidth=2)
        plt.title(f'MoSOA Sensitivity Studio (f_c): {fn}', fontweight='bold')
        plt.xlabel('Convergence Factor (f_c)', fontweight='bold')
        plt.ylabel('Mean Fitness (Lower is Better)', fontweight='bold')
        
        if df['Mean'].min() > 0:
            plt.yscale('log')
            
        plt.savefig(os.path.join(out_dir, f'sensitivity_fc_{fn}.png'))
        plt.close()
        
    # Study 2: Perturbation Decay p_beta
    p_beta_values = [2.0, 5.0, 8.0, 10.0]
    for fn in test_funcs:
        print(f"Testing p_beta sensitivity on {fn}...")
        df = run_sensitivity_test(fn, 'p_beta', p_beta_values)
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Value', y='Mean', marker='o', linewidth=2)
        plt.title(f'MoSOA Sensitivity Studio (p_beta): {fn}', fontweight='bold')
        plt.xlabel('Perturbation Decay (p_beta)', fontweight='bold')
        plt.ylabel('Mean Fitness (Lower is Better)', fontweight='bold')
        
        if df['Mean'].min() > 0:
            plt.yscale('log')
            
        plt.savefig(os.path.join(out_dir, f'sensitivity_pbeta_{fn}.png'))
        plt.close()

if __name__ == "__main__":
    main()
