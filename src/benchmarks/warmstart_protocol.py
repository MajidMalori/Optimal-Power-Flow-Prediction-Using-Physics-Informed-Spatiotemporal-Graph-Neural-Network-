from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class SolverRunResult:
    method: str
    converged: bool
    time_ms: float
    iterations: int
    is_feasible: bool = False
    constraint_satisfaction_rate: float = 0.0


def normalize_method_name(method: str) -> str:
    """Normalize method aliases to benchmark keys."""
    key = method.strip().lower()
    mapping = {
        "flat": "flat",
        "dc": "dc",
        "warmstart": "warmstart",
        "results": "warmstart",
        "neural": "warmstart",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported method: {method}")
    return mapping[key]


def is_rescue_candidate(flat_result: SolverRunResult) -> bool:
    """Rescue pool is defined by flat-start non-convergence."""
    return normalize_method_name(flat_result.method) == "flat" and not flat_result.converged


def compute_speed_summary(results: Dict[str, SolverRunResult]) -> Dict[str, float]:
    """
    Return compact speed stats from one-sample method results.
    Uses flat as denominator for speedup.
    """
    flat = results["flat"]
    warm = results["warmstart"]
    dc = results["dc"]
    speedup_vs_flat = (flat.time_ms / warm.time_ms) if warm.time_ms > 0 else 0.0
    return {
        "flat_time_ms": flat.time_ms,
        "dc_time_ms": dc.time_ms,
        "warmstart_time_ms": warm.time_ms,
        "flat_iterations": float(flat.iterations),
        "dc_iterations": float(dc.iterations),
        "warmstart_iterations": float(warm.iterations),
        "warmstart_speedup_vs_flat": speedup_vs_flat,
    }


def compute_feasibility_summary(results: Dict[str, SolverRunResult]) -> Dict[str, float]:
    """Return one-sample feasibility indicators for each method."""
    out = {}
    for key in ("flat", "dc", "warmstart"):
        r = results[key]
        out[f"{key}_converged"] = float(r.converged)
        out[f"{key}_feasible"] = float(r.is_feasible)
        out[f"{key}_constraint_rate"] = r.constraint_satisfaction_rate
    return out
