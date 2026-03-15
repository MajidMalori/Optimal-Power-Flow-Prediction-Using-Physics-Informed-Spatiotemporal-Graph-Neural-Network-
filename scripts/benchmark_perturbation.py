"""
Stage 1.5 Benchmark Runner: Perturbation Strategy Comparison (Section 3.2).
Tests MoSOA using the 4 different perturbation decay strategies:
1. Exponential (default/best)
2. Linear
3. Cosine
4. Quadratic
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


class MoSOAPerturbationVariant(MoSOA):
    """
    A variant of MoSOA that allows explicitly overriding the perturbation
    decay strategy (beta) for ablation studies.
    """
    def __init__(self, search_space: Dict[str, Any], seed: int = 42, 
                 pop_size: int = 30, strategy: str = 'exponential'):
        super().__init__(search_space, seed, pop_size=pop_size)
        self.strategy = strategy

    def optimize(self, objective_fn, n_trials: int) -> Dict[str, Any]:
        t_max = n_trials // self.pop_size
        it = 0
        
        while it < t_max:
            # 1. Evaluate fitness
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

            # Adaptive Parameters
            sigma = 1.0 + (np.std(self.fitness) / (np.mean(self.fitness) + 1e-6))
            a = self.f_c * (1 - (it / t_max))**sigma
            w = 0.95 - (it / t_max) * (0.95 - 0.35)
            
            # --- SECTION 3.2: MULTIPLE PERTURBATION STRATEGIES ---
            if self.strategy == 'exponential':
                beta = np.exp(-5 * it / t_max)
            elif self.strategy == 'linear':
                beta = 1.0 - (it / t_max)
            elif self.strategy == 'cosine':
                beta = 0.5 * (1 + np.cos(np.pi * it / t_max))
            elif self.strategy == 'quadratic':
                beta = (1 - it / t_max)**2
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            # Position Update
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
                
                # Apply the perturbation
                noise = np.random.uniform(-1, 1, self.dim) * (beta * (self.g_best_position - self.positions[i]))
                new_positions[i] = self._normalize_position(p_learned + noise + (p_attack - p_learned) * (it/t_max))

            self.positions = new_positions
            it += 1

        return {name: self.g_best_position[j] for j, name in enumerate(self.param_names)}


def run_strategy_benchmark(strategy: str, func_name: str, num_runs: int = 15, 
                           n_trials: int = 300, dim: int = 10):
    
    benchmark = BENCHMARKS[func_name]
    _obj_fn = benchmark['fn']
    bounds = benchmark['bounds']
    search_space = {f'x_{i}': bounds for i in range(dim)}
    
    def obj_fn(params: Dict[str, Any]) -> float:
        x_array = np.array([params[f'x_{i}'] for i in range(dim)])
        return _obj_fn(x_array)
    
    results = []
    print(f"Running '{strategy}' on {func_name} ({num_runs} runs)...")
    
    for run in range(num_runs):
        opt = MoSOAPerturbationVariant(search_space=search_space, seed=run+42, strategy=strategy)
        opt.optimize(obj_fn, n_trials=n_trials)
        results.append(opt.g_best_fitness)
        
    return {
        'Strategy': strategy,
        'Function': func_name,
        'Mean': np.mean(results),
        'Std': np.std(results),
        'Best': np.min(results)
    }

def main():
    test_funcs = ['F1', 'F5', 'F9'] 
    strategies = ['linear', 'cosine', 'quadratic', 'exponential']
    
    all_results = []
    
    for fn_name in test_funcs:
        for strat in strategies:
            res = run_strategy_benchmark(strat, fn_name)
            all_results.append(res)
            
    df = pd.DataFrame(all_results)
    print("\\n\\n======= PERTURBATION STRATEGY COMPARISON =======")
    print(df.to_string(index=False))
    print("================================================")
    
    df.to_csv("src/benchmarks/perturbation_results.csv", index=False)
    print("Results saved to src/benchmarks/perturbation_results.csv")

if __name__ == "__main__":
    main()
