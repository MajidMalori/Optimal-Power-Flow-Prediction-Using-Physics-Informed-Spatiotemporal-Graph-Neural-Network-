from typing import Dict

from src.benchmarks.benchmark_state import BenchmarkState
from src.benchmarks.speed_runtime import run_all_methods_for_state


def run_rescue_flow_for_state(
    state: BenchmarkState,
    max_iter: int = 100,
    tolerance: float = 1e-5,
    pred_vm=None,
    pred_va=None,
) -> Dict:
    """
    Rescue flow:
    1) flat start first
    2) if flat converges -> not a rescue candidate
    3) else run dc + warmstart on same state
    """
    import time
    print(f"[DEBUG] Starting rescue flow for state {state.sample_id}")
    
    start_time = time.time()
    print(f"[DEBUG] Calling run_all_methods_for_state...")
    all_res = run_all_methods_for_state(state, max_iter=max_iter, tolerance=tolerance, pred_vm=pred_vm, pred_va=pred_va)
    elapsed = time.time() - start_time
    print(f"[DEBUG] run_all_methods_for_state completed in {elapsed:.2f}s")
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
