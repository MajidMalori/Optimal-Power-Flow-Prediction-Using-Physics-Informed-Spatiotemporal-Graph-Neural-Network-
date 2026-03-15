import sys
import copy
from src.optimizers.mosoa import MoSOA
from src.optimizers.soa import SOA
from src.optimizers.tpe_wrapper import TPEOptimizer

def objective_function(params):
    # simple convex function for testing
    x = params['x']
    y = params['y']
    return (x - 3)**2 + (y + 2)**2

search_space = {
    'x': [-10.0, 10.0],
    'y': [-10.0, 10.0]
}

def test_optimizer(optimizer_class, name, n_trials=100, **kwargs):
    print(f"\\nTesting {name}...")
    opt = optimizer_class(search_space=copy.deepcopy(search_space), **kwargs)
    best_params = opt.optimize(objective_function, n_trials)
    best_val = objective_function(best_params)
    print(f"Best Params: {best_params}\\nBest Value: {best_val}")

if __name__ == "__main__":
    test_optimizer(SOA, "SOA (Baseline)", n_trials=100, pop_size=10, f_c=2.0)
    test_optimizer(MoSOA, "MoSOA", n_trials=100, pop_size=10, f_c=2.0, c1=1.5, c2=1.5)
    test_optimizer(TPEOptimizer, "TPE (Optuna)", n_trials=100)
    print("\\nAll tests completed.")
