from src.benchmarks.metrics import aggregate_speed, aggregate_feasibility, aggregate_rescue


def test_aggregate_speed():
    rows = [
        {"method": "flat", "time_ms": 20.0, "iterations": 8, "converged": True},
        {"method": "warmstart", "time_ms": 10.0, "iterations": 5, "converged": True},
        {"method": "dc", "time_ms": 15.0, "iterations": 6, "converged": True},
    ]
    out = aggregate_speed(rows)
    assert out["warmstart_speedup_vs_flat"] == 2.0
    assert out["dc_success_rate"] == 1.0


def test_aggregate_feasibility():
    rows = [
        {"method": "flat", "is_feasible": True, "constraint_satisfaction_rate": 1.0},
        {"method": "flat", "is_feasible": False, "constraint_satisfaction_rate": 0.8},
        {"method": "warmstart", "is_feasible": True, "constraint_satisfaction_rate": 0.95},
        {"method": "dc", "is_feasible": False, "constraint_satisfaction_rate": 0.7},
    ]
    out = aggregate_feasibility(rows)
    assert out["flat_feasibility_rate"] == 0.5
    assert out["warmstart_constraint_rate_mean"] == 0.95


def test_aggregate_rescue():
    rows = [
        {"dc_recovered": False, "warmstart_recovered": True},
        {"dc_recovered": True, "warmstart_recovered": True},
    ]
    out = aggregate_rescue(rows)
    assert out["rescue_candidates"] == 2.0
    assert out["dc_recovery_rate"] == 0.5
    assert out["warmstart_recovery_rate"] == 1.0
