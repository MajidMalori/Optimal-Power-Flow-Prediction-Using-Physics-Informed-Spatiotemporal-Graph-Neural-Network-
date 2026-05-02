from typing import Callable, Dict, List, Tuple

from src.benchmarks.benchmark_state import BenchmarkState
from src.benchmarks.metrics import aggregate_speed
from src.benchmarks.warmstart_protocol import SolverRunResult


RunAllMethodsFn = Callable[[BenchmarkState], Dict[str, SolverRunResult]]


def run_speed_benchmark(
    states: List[BenchmarkState],
    run_all_methods_fn: RunAllMethodsFn,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Run flat/dc/warmstart for each state and aggregate speed metrics.
    """
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
                    "time_ms": r.time_ms,
                    "iterations": r.iterations,
                }
            )
    return records, aggregate_speed(records)
