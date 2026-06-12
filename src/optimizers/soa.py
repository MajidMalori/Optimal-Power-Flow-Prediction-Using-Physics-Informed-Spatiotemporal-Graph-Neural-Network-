"""
Standard Seagull Optimization Algorithm (SOA) implementation.
"""
import numpy as np
from typing import Dict, Any, Callable
from .base import BaseOptimizer

class SOA(BaseOptimizer):
    """
    Standard Seagull Optimization Algorithm (Baseline).
    """

    def __init__(self, search_space: Dict[str, Any], seed: int = 42, 
                 pop_size: int = 30, f_c: float = 2.0):
        """
        Initialize standard SOA.
        """
        super().__init__(search_space, seed)
        self.pop_size = pop_size
        self.f_c = f_c
        self.dim = len(search_space)
        self.param_names = list(search_space.keys())
        
        # Initialize population
        self.positions = self._init_population()
        self.fitness = np.full(pop_size, np.inf)
        self.g_best_position = None
        self.g_best_fitness = np.inf

    def _init_population(self) -> np.ndarray:
        pop = np.zeros((self.pop_size, self.dim))
        for i, name in enumerate(self.param_names):
            limits = self.search_space[name]
            pop[:, i] = np.random.uniform(limits[0], limits[1], self.pop_size)
        return pop

    def _normalize_position(self, pos: np.ndarray) -> np.ndarray:
        """Handle boundary conditions using reflective strategy."""
        for i, name in enumerate(self.param_names):
            low, high = self.search_space[name]
            if pos[i] < low:
                pos[i] = 2 * low - pos[i]
            if pos[i] > high:
                pos[i] = 2 * high - pos[i]
            pos[i] = np.clip(pos[i], low, high)
        return pos

    def optimize(self, objective_fn: Callable[[Dict[str, Any]], float], n_trials: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute standard SOA optimization loop.
        """
        from tqdm import tqdm
        t_max = n_trials // self.pop_size
        it = 0
        
        pbar = tqdm(total=t_max, disable=not verbose, desc="SOA Optimization", leave=False)
        while it < t_max:
            # 1. Evaluate fitness
            for i in range(self.pop_size):
                params = {name: self.positions[i, j] for j, name in enumerate(self.param_names)}
                current_fitness = objective_fn(params)
                self.fitness[i] = current_fitness
                
                if current_fitness < self.g_best_fitness:
                    self.g_best_fitness = current_fitness
                    self.g_best_position = np.copy(self.positions[i])

            # 2. Linear Convergence Factor A (Eq. 24)
            a = self.f_c - (it * self.f_c / t_max)
            
            # 3. Update Position
            new_positions = np.zeros_like(self.positions)
            for i in range(self.pop_size):
                rd = np.random.random()
                b = 2 * (a**2) * rd
                
                # Standard fixed spiral (Eq. 28-32)
                k = np.random.uniform(0, 2 * np.pi)
                # Fixed u=1, v=1 for standard SOA
                radius = 1.0 * np.exp(k * 1.0)
                x = radius * np.cos(k)
                y = radius * np.sin(k)
                z = radius * k
                
                dist = np.abs(a * self.positions[i] + b * (self.g_best_position - self.positions[i]))
                
                # Standard position update
                p_attack = dist * x * y * z + self.g_best_position
                new_positions[i] = self._normalize_position(p_attack)

            self.positions = new_positions
            it += 1
            pbar.update(1)
            pbar.set_postfix(best=f"{self.g_best_fitness:.6f}")

        pbar.close()
        return {name: self.g_best_position[j] for j, name in enumerate(self.param_names)}
