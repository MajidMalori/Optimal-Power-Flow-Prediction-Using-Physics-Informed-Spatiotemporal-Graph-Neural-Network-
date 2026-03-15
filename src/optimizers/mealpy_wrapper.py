"""
Wrapper for mealpy metaheuristic algorithms.
Provides a unified interface for GA, PSO, GWO, HGSO, TSA, SOA, ESOA.
All implementations from the mealpy library (Van Thieu, 2023).
"""
import numpy as np
from typing import Callable
from mealpy import GA, PSO, GWO, TSA, SOA, HGSO, ESOA
from mealpy import FloatVar


# Map of algorithm names to their mealpy constructors
MEALPY_ALGORITHMS = {
    'GA':   GA.BaseGA,
    'PSO':  PSO.OriginalPSO,
    'GWO':  GWO.OriginalGWO,
    'HGSO': HGSO.OriginalHGSO,    # Physics-based (Henry Gas Solubility)
    'TSA':  TSA.OriginalTSA,
    'SOA*': SOA.OriginalSOA,       # mealpy's standard SOA (library reference)
    'ESOA': ESOA.OriginalESOA,     # Enhanced SOA (ISOA equivalent)
}


def run_mealpy_optimizer(
    algo_name: str,
    obj_fn: Callable[[np.ndarray], float],
    bounds: list,
    dim: int,
    n_trials: int = 500,
    pop_size: int = 30,
    seed: int = 42,
) -> dict:
    """
    Run a mealpy optimizer on a given objective function.

    Args:
        algo_name: Key from MEALPY_ALGORITHMS.
        obj_fn: Function that takes np.ndarray and returns float.
        bounds: [lower, upper] for all dimensions, or list of [lo, hi] per dim.
        dim: Number of dimensions.
        n_trials: Total function evaluations budget.
        pop_size: Population size.
        seed: Random seed.

    Returns:
        Dict with 'best_fitness' and 'best_position'.
    """
    # Build bounds list for mealpy
    if isinstance(bounds[0], list):
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]
    else:
        lb = [bounds[0]] * dim
        ub = [bounds[1]] * dim

    bound_obj = FloatVar(lb=lb, ub=ub)

    n_epochs = max(1, n_trials // pop_size)

    problem = {
        "obj_func": obj_fn,
        "bounds": bound_obj,
        "minmax": "min",
        "log_to": None,
    }

    algo_cls = MEALPY_ALGORITHMS[algo_name]
    optimizer = algo_cls(epoch=n_epochs, pop_size=pop_size)

    best = optimizer.solve(problem, seed=seed)

    return {
        'best_fitness': best.target.fitness,
        'best_position': best.solution,
        'history': optimizer.history.list_global_best_fit,
    }
