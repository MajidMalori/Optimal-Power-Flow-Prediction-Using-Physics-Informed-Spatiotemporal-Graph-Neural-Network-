"""
Base Optimizer module defining the abstract interface for all hyperparameter tuners.
"""
import abc
from typing import Dict, Any, Callable

import numpy as np

class BaseOptimizer(abc.ABC):
    """Abstract Base Class for all tuners (MoSOA, PSO, TPE, etc.)"""

    def __init__(self, search_space: Dict[str, Any], seed: int = 42):
        """
        Initialize the optimizer.

        Args:
            search_space: Dictionary defining hyperparameter ranges.
            seed: Random seed for reproducibility.
        """
        self.search_space = search_space
        self.seed = seed
        np.random.seed(seed)

    @abc.abstractmethod
    def optimize(self, objective_fn: Callable[[Dict[str, Any]], float], n_trials: int) -> Dict[str, Any]:
        """
        Runs the optimization process.
        
        Args:
            objective_fn: A function that takes hyperparameters (dict) and returns a scalar score.
            n_trials: Number of evaluations (trials) allowed.
            
        Returns:
            Best hyperparameters found during search.
        """
        pass
