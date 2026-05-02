from typing import Callable, Dict, List, Tuple

from src.benchmarks.benchmark_state import BenchmarkState
from src.benchmarks.metrics import aggregate_feasibility
from src.benchmarks.warmstart_protocol import SolverRunResult


RunAllMethodsFn = Callable[[BenchmarkState], Dict[str, SolverRunResult]]


def run_feasibility_benchmark(
    states: List[BenchmarkState],
    run_all_methods_fn: RunAllMethodsFn,
) -> Tuple[List[Dict], Dict[str, float]]:
    records: List[Dict] = []
    for state in states:
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
