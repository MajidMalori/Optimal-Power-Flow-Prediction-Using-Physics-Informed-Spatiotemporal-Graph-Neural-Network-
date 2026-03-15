"""
Stage 2 Hyperparameter Tuning Benchmark Runner.
Compares MoSOA vs SOA vs TPE vs Random Search on PISTGNN model tuning.
"""
import yaml
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from typing import Dict, Any

from src.optimizers.mosoa import MoSOA
from src.optimizers.soa import SOA
from src.optimizers.tpe_wrapper import TPEOptimizer

def load_hpo_config(config_path="configs/hpo_space.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def mock_training_pipeline(params: Dict[str, Any]) -> float:
    """
    In a real run, this function imports your PowerSystemTrainer
    and returns `val_loss`. 
    
    For now, this is a simulated response surface that prefers:
    lr ~= 0.005, hidden_dim ~= 64, dropout ~= 0.2
    """
    # Penalty for moving away from "ideal" values
    lr_err = np.abs(params.get('learning_rate', 0.005) - 0.005) * 1000
    dim_err = np.abs(params.get('gcn_hidden', 64) - 64) / 10
    drop_err = np.abs(params.get('dropout', 0.2) - 0.2) * 50
    layers_err = np.abs(params.get('num_layers', 3) - 3) * 5
    
    # Add noise to simulate real training variance
    noise = np.random.normal(0, 0.5)
    
    # Simulate validation loss
    return max(0.1, lr_err + dim_err + drop_err + layers_err + noise)

def run_tuning(optimizer_class: type, name: str, search_space: Dict[str, Any], n_trials: int, pop_size: int = 10):
    print(f"\n--- Running {name} HPO ---")
    start_time = time.time()
    
    if optimizer_class is None: # Random Search baseline
        best_val = np.inf
        for i in range(n_trials):
            params = {}
            for k, bounds in search_space.items():
                if isinstance(bounds[0], int) and isinstance(bounds[1], int):
                    params[k] = np.random.randint(bounds[0], bounds[1] + 1)
                else:
                    params[k] = np.random.uniform(bounds[0], bounds[1])
            val = mock_training_pipeline(params)
            if val < best_val:
                best_val = val
        best_params = {"Result": "N/A"}
    else:
        # Pass pop_size to metaheuristics, not needed for TPE
        kwargs = {"pop_size": pop_size} if optimizer_class in [MoSOA, SOA] else {}
        opt = optimizer_class(search_space=search_space, seed=42, **kwargs)
        best_params = opt.optimize(mock_training_pipeline, n_trials=n_trials)
        best_val = mock_training_pipeline(best_params)
        
    execution_time = time.time() - start_time
    
    return {
        'Algorithm': name,
        'Best Val Loss': best_val,
        'Time (s)': execution_time,
    }

def main():
    config = load_hpo_config()
    search_space = config['hpo_space']
    settings = config['benchmark_settings']
    n_trials = settings['n_trials']
    pop_size = settings['pop_size']
    
    print(f"Starting Stage 2 HPO Benchmarks: {n_trials} trials each.")
    
    results = []
    
    # 1. Random Search
    results.append(run_tuning(None, "Random Search", search_space, n_trials))
    
    # 2. TPE (Optuna)
    results.append(run_tuning(TPEOptimizer, "TPE (Optuna)", search_space, n_trials))
    
    # 3. Standard SOA
    results.append(run_tuning(SOA, "SOA", search_space, n_trials, pop_size))
    
    # 4. MoSOA
    results.append(run_tuning(MoSOA, "MoSOA", search_space, n_trials, pop_size))
    
    df = pd.DataFrame(results)
    print("\n\n================ HPO PIPELINE RESULTS ================")
    print(df.to_string(index=False))
    print("======================================================")
    
    df.to_csv("reports/benchmarks/benchmark_hpo_results.csv", index=False)
    print("Results saved to reports/benchmarks/benchmark_hpo_results.csv")

if __name__ == "__main__":
    main()
