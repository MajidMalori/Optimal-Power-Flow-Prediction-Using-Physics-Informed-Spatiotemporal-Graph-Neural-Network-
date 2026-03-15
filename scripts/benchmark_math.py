"""
Runner script for Stage 1 Math Benchmarks.
Compares MoSOA vs SOA on standard test functions.
"""
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from typing import Dict, Any

from src.benchmarks.functions import BENCHMARKS
from src.optimizers.mosoa import MoSOA
from src.optimizers.soa import SOA


def run_benchmark(optimizer_class: type, func_name: str, num_runs: int = 30, 
                  n_trials: int = 500, pop_size: int = 30, dim: int = 30):
    """
    Runs an optimizer on a specific function multiple times.
    """
    benchmark = BENCHMARKS[func_name]
    _obj_fn = benchmark['fn']
    bounds = benchmark['bounds']
    
    # Create search space for the given dimension
    search_space = {f'x_{i}': bounds for i in range(dim)}
    
    # Wrapper to convert dict back to numpy array for math functions
    def obj_fn(params: Dict[str, Any]) -> float:
        x_array = np.array([params[f'x_{i}'] for i in range(dim)])
        return _obj_fn(x_array)
    
    results = []
    
    print(f"\nRunning {optimizer_class.__name__} on {func_name} ({num_runs} runs)...")
    start_time = time.time()
    
    for run in range(num_runs):
        opt = optimizer_class(
            search_space=search_space, 
            seed=run + 42, # Different seed per run
            pop_size=pop_size
        )
        best_params = opt.optimize(obj_fn, n_trials=n_trials)
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
        'Time (s)': execution_time
    }

def main():
    # Only test a few functions for the demo to save time
    test_funcs = ['F1', 'F5', 'F9', 'F10'] 
    
    all_results = []
    
    for fn_name in test_funcs:
        for opt_class in [SOA, MoSOA]:
            # Using 10 runs and 300 evaluations for faster testing
            res = run_benchmark(opt_class, fn_name, num_runs=10, n_trials=300, dim=10)
            all_results.append(res)
            
    df = pd.DataFrame(all_results)
    
    print("\n\n================ BENCHMARK RESULTS ================")
    print(df.to_string(index=False))
    print("=====================================================")
    
    # Save to CSV
    df.to_csv("reports/benchmarks/benchmark_math_results.csv", index=False)
    print("Results saved to reports/benchmarks/benchmark_math_results.csv")

if __name__ == "__main__":
    main()
