from typing import Dict, Callable

from src.benchmarks.warmstart_protocol import SolverRunResult


RunMethodFn = Callable[[str], SolverRunResult]


def evaluate_all_methods(run_method: RunMethodFn) -> Dict[str, SolverRunResult]:
    """
    Evaluate flat/dc/warmstart on the same reconstructed network state.
    This is used for speed + feasibility pillars.
    """
    return {
        "flat": run_method("flat"),
        "dc": run_method("dc"),
        "warmstart": run_method("warmstart"),
    }


def evaluate_rescue_flow(run_method: RunMethodFn) -> Dict[str, SolverRunResult]:
    """
    Rescue flow:
      1) run flat first
      2) only if flat fails, run dc + warmstart as recovery attempts
    """
    flat = run_method("flat")
    out: Dict[str, SolverRunResult] = {"flat": flat}
    if flat.converged:
        return out
    out["dc"] = run_method("dc")
    out["warmstart"] = run_method("warmstart")
    return out
