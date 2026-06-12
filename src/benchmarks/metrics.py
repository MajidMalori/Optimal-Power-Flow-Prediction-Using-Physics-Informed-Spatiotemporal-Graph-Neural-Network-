from typing import Dict, List


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def aggregate_speed(records: List[Dict]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for method in ("flat", "dc", "warmstart"):
        rows = [r for r in records if r["method"] == method]
        out[f"{method}_time_ms_mean"] = _safe_mean([r["time_ms"] for r in rows if r["converged"]])
        out[f"{method}_iter_mean"] = _safe_mean([float(r["iterations"]) for r in rows if r["converged"]])
        out[f"{method}_success_rate"] = _safe_mean([1.0 if r["converged"] else 0.0 for r in rows])
    denom = out["warmstart_time_ms_mean"]
    out["warmstart_speedup_vs_flat"] = (out["flat_time_ms_mean"] / denom) if denom > 0 else 0.0
    return out


def aggregate_feasibility(records: List[Dict]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for method in ("flat", "dc", "warmstart"):
        rows = [r for r in records if r["method"] == method]
        out[f"{method}_feasibility_rate"] = _safe_mean([1.0 if r["is_feasible"] else 0.0 for r in rows])
        out[f"{method}_constraint_rate_mean"] = _safe_mean([r["constraint_satisfaction_rate"] for r in rows])
    return out


def aggregate_rescue(records: List[Dict]) -> Dict[str, float]:
    candidates = len(records)
    if candidates == 0:
        return {"rescue_candidates": 0.0, "dc_recovery_rate": 0.0, "warmstart_recovery_rate": 0.0}
    dc = sum(1 for r in records if r.get("dc_recovered"))
    ws = sum(1 for r in records if r.get("warmstart_recovered"))
    return {
        "rescue_candidates": float(candidates),
        "dc_recovery_rate": dc / candidates,
        "warmstart_recovery_rate": ws / candidates,
    }
