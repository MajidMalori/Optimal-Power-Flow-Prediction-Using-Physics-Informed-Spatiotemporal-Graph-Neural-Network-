"""
Compares MoSOA vs SOA on standard test functions.
"""
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any

from src.benchmarks.functions import BENCHMARKS
from src.optimizers.mosoa import MoSOA
from src.optimizers.soa import SOA


def run_benchmark(optimizer_class: type, func_name: str, num_runs: int = 30, 
                  n_trials: int = 500, pop_size: int = 30, dim: int = 30):
    """Runs an optimizer on a specific function multiple times."""
    benchmark = BENCHMARKS[func_name]
    _obj_fn = benchmark['fn']
    bounds = benchmark['bounds']
    search_space = {f'x_{i}': bounds for i in range(dim)}
    
    def obj_fn(params: Dict[str, Any]) -> float:
        x_array = np.array([params[f'x_{i}'] for i in range(dim)])
        return _obj_fn(x_array)
    
    results = []
    start_time = time.time()
    
    for run in range(num_runs):
        opt = optimizer_class(
            search_space=search_space, 
            seed=run + 42,
            pop_size=pop_size
        )
        best_params = opt.optimize(obj_fn, n_trials=n_trials, verbose=False)
        best_val = obj_fn(best_params)
        results.append(best_val)
        
    execution_time = time.time() - start_time
        
    return {
        'Algorithm': optimizer_class.__name__,
        'Function': func_name,
        'Best': np.min(results),
        'Worst': np.max(results),
        'Mean': np.mean(results),
        'Std': np.std(results),
        'Time (s)': round(execution_time, 3)
    }

def main():
    test_funcs = ['F1', 'F5', 'F9', 'F10'] 
    optimizers = [SOA, MoSOA]
    
    all_results = []
    tasks = [(fn, opt) for fn in test_funcs for opt in optimizers]
    
    for fn_name, opt_class in tqdm(tasks, desc="Math Benchmarks"):
        res = run_benchmark(opt_class, fn_name, num_runs=10, n_trials=300, dim=10)
        all_results.append(res)
            
    df = pd.DataFrame(all_results)
    
    print("\n\n================ BENCHMARK RESULTS ================")
    print(df.to_string(index=False))
    print("=====================================================")
    
    os.makedirs("reports/mosoa", exist_ok=True)
    df.to_csv("reports/mosoa/benchmark_math_results.csv", index=False)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df, x='Function', y='Mean', hue='Algorithm')
        plt.yscale('log')
        plt.title('MoSOA vs SOA: Mean Fitness on Math Benchmarks (Lower is Better)')
        plt.ylabel('Mean Fitness (Log Scale)')
        plt.tight_layout()
        plt.savefig("reports/mosoa/math_benchmark_comparison.png", dpi=300)
        plt.close()
    except ImportError:
        pass

    print("Results saved to reports/mosoa/")

if __name__ == "__main__":
    main()
