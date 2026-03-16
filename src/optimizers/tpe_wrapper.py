"""
TPE Wrapper for Optuna-based hyperparameter tuning.
"""
try:
    import optuna
except ImportError:
    optuna = None

from typing import Dict, Any, Callable
from .base import BaseOptimizer

class TPEOptimizer(BaseOptimizer):
    """
    Wrapper for Optuna's Tree-structured Parzen Estimator (TPE).
    """

    def __init__(self, search_space: Dict[str, Any], seed: int = 42):
        """
        Initialize TPE Optimizer.
        """
        super().__init__(search_space, seed)
        if optuna is None:
            print("Warning: optuna is not installed. Please run 'pip install optuna'.")

    def optimize(self, objective_fn: Callable[[Dict[str, Any]], float], n_trials: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute Optuna TPE optimization.
        """
        if optuna is None:
            raise ImportError("Optuna is not installed.")

        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.ERROR)

        # Create study with faster settings for HPO benchmarking
        sampler = optuna.samplers.TPESampler(seed=self.seed, multivariate=False)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def optuna_objective(trial):
            params = {}
            for name, limits in self.search_space.items():
                if isinstance(limits[0], int) and isinstance(limits[1], int):
                    params[name] = trial.suggest_int(name, limits[0], limits[1])
                else:
                    params[name] = trial.suggest_float(name, limits[0], limits[1])
            return objective_fn(params)

        study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=verbose)
        return study.best_params
