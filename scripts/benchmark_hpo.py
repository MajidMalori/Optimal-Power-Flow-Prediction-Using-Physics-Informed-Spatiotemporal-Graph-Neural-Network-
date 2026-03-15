"""
Compares MoSOA vs SOA vs TPE vs Random Search on PISTGNN model tuning.
"""
import yaml
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any

from src.optimizers.mosoa import MoSOA
from src.optimizers.soa import SOA
from src.optimizers.tpe_wrapper import TPEOptimizer

def load_hpo_config(config_path="configs/hpo_space.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def mock_training_pipeline(params: Dict[str, Any]) -> float:
    """
    Simulated response surface that prefers:
    lr ~= 0.005, gcn_hidden ~= 64, dropout ~= 0.2
    """
    lr_err = np.abs(params.get('learning_rate', 0.005) - 0.005) * 1000
    dim_err = np.abs(params.get('gcn_hidden', 64) - 64) / 10
    drop_err = np.abs(params.get('dropout', 0.2) - 0.2) * 50
    layers_err = np.abs(params.get('num_layers', 3) - 3) * 5
    noise = np.random.normal(0, 0.5)
    return max(0.1, lr_err + dim_err + drop_err + layers_err + noise)

def run_tuning(optimizer_class: type, name: str, search_space: Dict[str, Any], 
               n_trials: int, pop_size: int = 10):
    start_time = time.time()
    
    if optimizer_class is None:  # Random Search baseline
        best_val = np.inf
        for _ in range(n_trials):
            params = {}
            for k, bounds in search_space.items():
                if isinstance(bounds[0], int) and isinstance(bounds[1], int):
                    params[k] = np.random.randint(bounds[0], bounds[1] + 1)
                else:
                    params[k] = np.random.uniform(bounds[0], bounds[1])
            val = mock_training_pipeline(params)
            if val < best_val:
                best_val = val
    else:
        kwargs = {"pop_size": pop_size} if optimizer_class in [MoSOA, SOA] else {}
        opt = optimizer_class(search_space=search_space, seed=42, **kwargs)
        best_params = opt.optimize(mock_training_pipeline, n_trials=n_trials, verbose=False)
        best_val = mock_training_pipeline(best_params)
        
    execution_time = time.time() - start_time
    
    return {
        'Algorithm': name,
        'Best Val Loss': round(best_val, 4),
        'Time (s)': round(execution_time, 3),
    }

def main():
    config = load_hpo_config()
    search_space = config['hpo_space']
    settings = config['benchmark_settings']
    n_trials = settings['n_trials']
    pop_size = settings['pop_size']
    
    algorithms = [
        (None, "Random Search"),
        (TPEOptimizer, "TPE (Optuna)"),
        (SOA, "SOA"),
        (MoSOA, "MoSOA"),
    ]
    
    results = []
    for opt_class, name in tqdm(algorithms, desc="HPO Benchmarks"):
        results.append(run_tuning(opt_class, name, search_space, n_trials, pop_size))
    
    df = pd.DataFrame(results)
    print("\n\n================ HPO PIPELINE RESULTS ================")
    print(df.to_string(index=False))
    print("======================================================")
    
    os.makedirs("reports/mosoa", exist_ok=True)
    df.to_csv("reports/mosoa/benchmark_hpo_results.csv", index=False)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='Algorithm', y='Best Val Loss')
        plt.title('Stage 2: Hyperparameter Tuning Performance (Lower is Better)')
        plt.ylabel('Validation Loss')
        plt.tight_layout()
        plt.savefig("reports/mosoa/hpo_performance_comparison.png", dpi=300)
        plt.close()
    except ImportError:
        pass

    print("Results saved to reports/mosoa/")

if __name__ == "__main__":
    main()
