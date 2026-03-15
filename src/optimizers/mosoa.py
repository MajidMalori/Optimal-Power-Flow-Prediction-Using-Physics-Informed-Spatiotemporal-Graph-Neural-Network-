"""
Modified Seagull Optimization Algorithm (MoSOA) implementation.
"""
import numpy as np
from typing import Dict, Any, Callable, List
from .base import BaseOptimizer

class MoSOA(BaseOptimizer):
    """
    Modified Seagull Optimization Algorithm for hyperparameter tuning.
    """

    def __init__(self, search_space: Dict[str, Any], seed: int = 42, 
                 pop_size: int = 30, f_c: float = 2.0, 
                 c1: float = 1.5, c2: float = 1.5):
        """
        Initialize MoSOA.

        Args:
            search_space: Dict of hyperparams with min/max or list of values.
            seed: Random seed.
            pop_size: Number of seagulls in the population.
            f_c: Frequency of manipulation for convergence factor A.
            c1: Global learning factor.
            c2: Personal learning factor.
        """
        super().__init__(search_space, seed)
        self.pop_size = pop_size
        self.f_c = f_c
        self.c1 = c1
        self.c2 = c2
        self.dim = len(search_space)
        self.param_names = list(search_space.keys())
        
        # Initialize population
        self.positions = self._init_population()
        self.fitness = np.full(pop_size, np.inf)
        
        # Memory
        self.p_best_positions = np.copy(self.positions)
        self.p_best_fitness = np.full(pop_size, np.inf)
        self.g_best_position = None
        self.g_best_fitness = np.inf

    def _init_population(self) -> np.ndarray:
        """Create initial random population within search space bounds."""
        pop = np.zeros((self.pop_size, self.dim))
        for i, name in enumerate(self.param_names):
            limits = self.search_space[name]
            pop[:, i] = np.random.uniform(limits[0], limits[1], self.pop_size)
        return pop

    def _normalize_position(self, pos: np.ndarray) -> np.ndarray:
        """Clip position to search space bounds."""
        for i, name in enumerate(self.param_names):
            limits = self.search_space[name]
            pos[i] = np.clip(pos[i], limits[0], limits[1])
        return pos

    def optimize(self, objective_fn: Callable[[Dict[str, Any]], float], n_trials: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute MoSOA optimization loop.
        """
        from tqdm import tqdm
        t_max = n_trials // self.pop_size
        it = 0
        
        pbar = tqdm(total=t_max, disable=not verbose, desc="MoSOA Optimization", leave=False)
        history = []
        while it < t_max:
            # 1. Evaluate fitness
            for i in range(self.pop_size):
                params = {name: self.positions[i, j] for j, name in enumerate(self.param_names)}
                current_fitness = objective_fn(params)
                self.fitness[i] = current_fitness
                
                # Update personal best
                if current_fitness < self.p_best_fitness[i]:
                    self.p_best_fitness[i] = current_fitness
                    self.p_best_positions[i] = np.copy(self.positions[i])
                
                # Update global best
                if current_fitness < self.g_best_fitness:
                    self.g_best_fitness = current_fitness
                    self.g_best_position = np.copy(self.positions[i])
            
            history.append(self.g_best_fitness)

            # 2. Adaptive Parameters
            # Eq. 33-34: Nonlinear convergence factor A
            # Simple adaptive decay for demonstration
            sigma = 1.0 + (np.std(self.fitness) / (np.mean(self.fitness) + 1e-6))
            a = self.f_c * (1 - (it / t_max))**sigma
            
            # Eq. 36: Inertia weight w
            w = 0.95 - (it / t_max) * (0.95 - 0.35)
            
            # Eq. 39: Exponential perturbation decay beta
            beta = np.exp(-5 * it / t_max)

            # 3. Position Update Loop
            new_positions = np.zeros_like(self.positions)
            for i in range(self.pop_size):
                # Exploration (Migration)
                # Avoid collision and move toward best
                rd = np.random.random()
                b = 2 * (a**2) * rd
                
                # Spiral/Helical Attack (Exploitation)
                # Tanh-based radius contraction (Eq. 35)
                k = np.random.uniform(0, 2 * np.pi)
                radius = np.tanh(1 - it / t_max) * np.exp(k * 0.1) # v=0.1
                
                x = radius * np.cos(k)
                y = radius * np.sin(k)
                z = radius * k
                
                # Distance to best
                dist = np.abs(a * self.positions[i] + b * (self.g_best_position - self.positions[i]))
                
                # Proposed update combining spiral and migration
                p_attack = dist * x * y * z + self.g_best_position
                
                # Learning Strategy (Eq. 40)
                r1, r2 = np.random.random(), np.random.random()
                p_learned = (w * self.positions[i] + 
                             self.c1 * r1 * (self.g_best_position - self.positions[i]) + 
                             self.c2 * r2 * (self.p_best_positions[i] - self.positions[i]))
                
                # Final position with dynamic perturbation
                noise = np.random.uniform(-1, 1, self.dim) * (beta * (self.g_best_position - self.positions[i]))
                new_positions[i] = self._normalize_position(p_learned + noise + (p_attack - p_learned) * (it/t_max))

            self.positions = new_positions
            it += 1
            pbar.update(1)
            pbar.set_postfix(best=f"{self.g_best_fitness:.6f}")

        pbar.close()
        return {
            'best_params': {name: self.g_best_position[j] for j, name in enumerate(self.param_names)},
            'history': history,
            'best_fitness': self.g_best_fitness
        }
