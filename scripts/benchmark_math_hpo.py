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
from src.visualization.plot_hpo_tuning import run_hpo_plotting

def mock_training_pipeline(params: Dict[str, Any], **kwargs) -> float:
    """
    Standard Mathematical Research Surface:
    Combines a global quadratic bowl with a high-frequency multimodal 'ripple'
    to challenge the optimizer's ability to escape local minima.
    """
    # 1. Global Structure (Quadratic Bowl)
    lr_err = np.abs(params.get('learning_rate', 0.005) - 0.005) * 1000
    gcn_err = np.abs(params.get('gcn_hidden', 64) - 64) / 10
    lstm_err = np.abs(params.get('lstm_hidden', 32) - 32) / 10
    gru_err = np.abs(params.get('gru_hidden', 32) - 32) / 10
    drop_err = np.abs(params.get('dropout', 0.2) - 0.2) * 50
    layers_err = np.abs(params.get('num_layers', 3) - 3) * 5
    
    base_loss = lr_err + gcn_err + lstm_err + gru_err + drop_err + layers_err
    
    # 2. Multimodal Ripples (Non-convexity)
    # We add a cosine modulation to create local minima every ~scale units
    ripple = 0.0
    for val in params.values():
        ripple += 2.0 * (1.0 - np.cos(5.0 * np.pi * val))
        
    # 3. Stochastic Noise (Training variance)
    # We add noise for the tuner, but provide a way to get noise-free value for reporting
    noise = kwargs.get('noise', np.random.normal(0, 0.5))
    
    return max(0.1, base_loss + ripple + noise)

def run_tuning(optimizer_class: type, name: str, search_space: Dict[str, Any], 
               n_trials: int, pop_size: int = 10, verbose: bool = True, seed: int = 42, **kwargs):
    start_time = time.time()
    all_trials = []
    
    if optimizer_class is None:  # Random Search baseline
        best_val = np.inf
        start_exec = time.time()
        np.random.seed(seed)
        for i in range(n_trials):
            params = {}
            for k, bounds in search_space.items():
                if isinstance(bounds[0], int) and isinstance(bounds[1], int):
                    params[k] = np.random.randint(bounds[0], bounds[1] + 1)
                else:
                    params[k] = np.random.uniform(bounds[0], bounds[1])
            val = mock_training_pipeline(params)
            
            trial_data = params.copy()
            trial_data['Val_Loss'] = val
            trial_data['Algorithm'] = name
            trial_data['Model'] = "Mathematical_Benchmark"
            trial_data['Trial'] = i + 1
            trial_data['Time_Elapsed_s'] = time.time() - start_exec
            all_trials.append(trial_data)
            
            if val < best_val:
                best_val = val
    else:
        # Swarm / Optuna
        opt = optimizer_class(search_space=search_space, seed=seed, **kwargs)
        
        # We need to capture history for parallel coordinates
        captured_trials = []
        start_exec = time.time()
        def wrapped_fn(p):
            res = mock_training_pipeline(p)
            td = p.copy()
            td['Val_Loss'] = res
            td['Algorithm'] = name
            td['Model'] = "Mathematical_Benchmark"
            td['Trial'] = len(captured_trials) + 1
            td['Time_Elapsed_s'] = time.time() - start_exec
            captured_trials.append(td)
            return res
            
        best_res = opt.optimize(wrapped_fn, n_trials=n_trials, verbose=verbose)
        if isinstance(best_res, dict) and 'best_params' in best_res:
            best_params = best_res['best_params']
        else:
            best_params = best_res
            
        # For the final results table, we use the BEST VAL observed during search 
        # to ensure consistency with the plots, OR a noise-free evaluation.
        # Let's use the actual best recorded during trials.
        best_val = min([t['Val_Loss'] for t in captured_trials])
        all_trials = captured_trials
        
    execution_time = time.time() - start_time
    
    summary = {
        'Algorithm': name,
        'Best Val Loss': round(best_val, 4),
        'Time (s)': round(execution_time, 3),
    }
    return summary, all_trials

def main():
    parser = argparse.ArgumentParser(description='Benchmark HPO for PISTGNN (Proxy Mathematical Pipeline)')
    parser.add_argument('--case', type=str, default='case33', help='Power system case')
    args = parser.parse_args()

    # 1. Load config
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "mosoa.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
        
    with open(config_path, 'r') as f:
        full_conf = yaml.safe_load(f)
        exec_config = full_conf.get('mathematical_hpo', {})
        mosoa_params = full_conf.get('mosoa_params', {})
        search_space = full_conf.get('mathematical_hpo_space', {})

    iterations = exec_config.get('iterations', 200)
    pop_size = exec_config.get('pop_size', 10)
    n_trials = iterations * pop_size
    
    # 2. Setup reports
    base_dir = "reports/mosoa/mathematical_hpo"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    csv_dir = os.path.join(base_dir, "csv")
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # 3. Algorithms
    algorithm_map = {
        'random': (None, "Random Search"),
        'tpe': (TPEOptimizer, "TPE (Optuna)"),
        'soa': (SOA, "SOA"),
        'mosoa': (MoSOA, "MoSOA"),
    }
    
    algorithms_to_run_names = ['mosoa', 'soa', 'tpe', 'random']
    algorithms = [algorithm_map[name] for name in algorithms_to_run_names]
    
    num_runs = exec_config.get('num_runs', 1)
    all_summaries = []
    all_trial_data = []
    
    print(f"\nRunning Mathematical HPO Comparison (High-Complexity Landscape, {num_runs} Runs)...")
    print("-" * 75)
    
    for opt_class, name in tqdm(algorithms, desc=f"{'Tuning Stage':<15}", ncols=100):
        extra_args = mosoa_params if name == "MoSOA" else {}
        run_stats = []
        for run_id in range(num_runs):
            seed = 42 + run_id
            res_summary, trials = run_tuning(opt_class, name, search_space, n_trials=n_trials, pop_size=pop_size, 
                                           verbose=False, seed=seed, **extra_args)
            for t in trials:
                t['Run'] = run_id + 1
            run_stats.append(res_summary)
            all_trial_data.extend(trials)
        
        avg_loss = np.mean([s['Best Val Loss'] for s in run_stats])
        avg_time = np.mean([s['Time (s)'] for s in run_stats])
        std_loss = np.std([s['Best Val Loss'] for s in run_stats])
        
        all_summaries.append({
            'Algorithm': name,
            'Best Val Loss (Mean)': round(avg_loss, 4),
            'Best Val Loss (Std)': round(std_loss, 4),
            'Time (s) Avg': round(avg_time, 3),
        })
    
    df = pd.DataFrame(all_summaries)
    trials_df = pd.DataFrame(all_trial_data)
    
    print(f"\n================ MATHEMATICAL HPO RESULTS ({num_runs} Runs Avg) ================")
    print(df.to_string(index=False))
    print("=" * 75)
    
    df.to_csv(os.path.join(csv_dir, "math_hpo_results.csv"), index=False)
    trials_df.to_csv(os.path.join(csv_dir, "math_hpo_trials.csv"), index=False)
    
    # Generate high-tier research plots
    try:
        run_hpo_plotting(trials_df, out_dir=plot_dir)
    except Exception as e:
        print(f"Visualization error: {e}")
    
    print(f"\nBenchmark complete. Results and Research Plots saved to {base_dir}\n")

if __name__ == "__main__":
    main()
