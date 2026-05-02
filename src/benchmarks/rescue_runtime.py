from typing import Dict

from src.benchmarks.benchmark_state import BenchmarkState
from src.benchmarks.speed_runtime import run_all_methods_for_state


def run_rescue_flow_for_state(
    state: BenchmarkState,
    max_iter: int = 100,
    tolerance: float = 1e-5,
) -> Dict:
    """
    Rescue flow:
    1) flat start first
    2) if flat converges -> not a rescue candidate
    3) else run dc + warmstart on same state
    """
    all_res = run_all_methods_for_state(state, max_iter=max_iter, tolerance=tolerance)
    flat = {
        "converged": all_res["flat"].converged,
        "time_ms": all_res["flat"].time_ms,
        "iterations": all_res["flat"].iterations,
    }
    if flat["converged"]:
        return {"flat": flat}
    return {
        "flat": flat,
        "dc": {
            "converged": all_res["dc"].converged,
            "time_ms": all_res["dc"].time_ms,
            "iterations": all_res["dc"].iterations,
        },
        "warmstart": {
            "converged": all_res["warmstart"].converged,
            "time_ms": all_res["warmstart"].time_ms,
            "iterations": all_res["warmstart"].iterations,
        },
    }
