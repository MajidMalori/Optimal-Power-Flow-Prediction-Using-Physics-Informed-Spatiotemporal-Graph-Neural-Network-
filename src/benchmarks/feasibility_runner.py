from typing import Callable, Dict, List, Tuple

from src.benchmarks.benchmark_state import BenchmarkState
from src.benchmarks.metrics import aggregate_feasibility
from src.benchmarks.warmstart_protocol import SolverRunResult


RunAllMethodsFn = Callable[[BenchmarkState], Dict[str, SolverRunResult]]


from tqdm import tqdm

def run_feasibility_benchmark(
    states: List[BenchmarkState],
    run_all_methods_fn: RunAllMethodsFn,
) -> Tuple[List[Dict], Dict[str, float]]:
    records: List[Dict] = []
    case_name = states[0].case_name.upper() if states else "FEASIBILITY"
    desc = f"Solving {case_name}"
    desc = f"{desc:<25}"
    for state in tqdm(states, desc=desc,
                      bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} states",
                      unit="state", leave=False):
        results = run_all_methods_fn(state)
        for method in ("flat", "dc", "warmstart"):
            r = results[method]
            records.append(
                {
                    "sample_id": state.sample_id,
                    "case_name": state.case_name,
                    "timestep": state.timestep,
                    "renewable_fraction": state.renewable_fraction,
                    "method": method,
                    "converged": r.converged,
                    "is_feasible": r.is_feasible,
                    "constraint_satisfaction_rate": r.constraint_satisfaction_rate,
                }
            )
    return records, aggregate_feasibility(records)
