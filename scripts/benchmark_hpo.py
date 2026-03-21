import numpy as np
import pandas as pd
import os
import sys
import time
import argparse
import yaml
from tqdm import tqdm
from typing import Dict, Any, List
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.optimizers.mosoa import MoSOA
from src.optimizers.soa import SOA
from src.optimizers.tpe_wrapper import TPEOptimizer
from src.visualization.plot_mosoa import plot_hpo_performance

def load_hpo_config(config_path="configs/mosoa.yaml"):
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
               n_trials: int, pop_size: int = 10, verbose: bool = True, **mosoa_params):
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
        if optimizer_class == MoSOA:
            kwargs.update(mosoa_params)
        opt = optimizer_class(search_space=search_space, seed=42, **kwargs)
        best_params = opt.optimize(mock_training_pipeline, n_trials=n_trials, verbose=verbose)
        best_val = mock_training_pipeline(best_params)
        
    execution_time = time.time() - start_time
    
    return {
        'Algorithm': name,
        'Best Val Loss': round(best_val, 4),
        'Time (s)': round(execution_time, 3),
    }

def main():
    parser = argparse.ArgumentParser(description='Benchmark HPO for PISTGNN')
    parser.add_argument('--case', type=str, default='case33', help='Power system case')
    args = parser.parse_args()

    # 1. Load Execution Config
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "mosoa.yaml")
    exec_config = {}
    mosoa_params = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            full_conf = yaml.safe_load(f)
            exec_config = full_conf.get('hpo', {})
            mosoa_params = full_conf.get('mosoa_params', {})
            
    iterations = exec_config.get('iterations', 200)
    pop_size = exec_config.get('pop_size', 10)
    n_trials = iterations * pop_size
    
    # 2. Clear previous reports
    out_dir = "reports/mosoa/hpo"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 3. Load Search Space Config
    space_path = os.path.join(os.path.dirname(__file__), "..", "configs", "mosoa.yaml")
    with open(space_path, 'r') as f:
        space_config = yaml.safe_load(f)
    
    search_space = space_config['hpo_space']
    
    # Use fixed algorithms for HPO benchmark or load from config if added later
    algorithm_map = {
        'random': (None, "Random Search"),
        'tpe': (TPEOptimizer, "TPE (Optuna)"),
        'soa': (SOA, "SOA"),
        'mosoa': (MoSOA, "MoSOA"),
    }
    
    # Get algorithms to run from config, or use all if not specified
    algorithms_to_run_names = exec_config.get('algorithms', ['mosoa', 'soa', 'tpe', 'random'])
    algorithms = [algorithm_map[name] for name in algorithms_to_run_names if name in algorithm_map]
    
    results = []
    print() # Spacing from command
    for opt_class, name in tqdm(algorithms, desc="HPO Comparison", leave=True, dynamic_ncols=True):
        extra_args = mosoa_params if name == "MoSOA" else {}
        res = run_tuning(opt_class, name, search_space, n_trials=n_trials, pop_size=pop_size, verbose=False, **extra_args)
        results.append(res)
    
    df = pd.DataFrame(results)
    print("\n\n================ HPO PIPELINE RESULTS ================")
    print(df.to_string(index=False))
    print("======================================================")
    
    out_dir = "reports/mosoa/hpo"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "hpo_results.csv"), index=False)
    
    try:
        # Add a small progress bar for the plotting phase as requested
        for _ in tqdm(range(1), desc="Generating Performance Plot", leave=True, dynamic_ncols=True):
            plot_hpo_performance(df, os.path.join(out_dir, "hpo_performance_comparison.png"))
    except Exception as e:
        print(f"Visualization error: {e}")

if __name__ == "__main__":
    main()
