import pytest

from src.benchmarks.warmstart_protocol import (
    SolverRunResult,
    normalize_method_name,
    is_rescue_candidate,
    compute_speed_summary,
    compute_feasibility_summary,
)


def test_normalize_method_aliases():
    assert normalize_method_name("flat") == "flat"
    assert normalize_method_name("dc") == "dc"
    assert normalize_method_name("results") == "warmstart"
    assert normalize_method_name("warmstart") == "warmstart"


def test_normalize_method_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_method_name("foo")


def test_rescue_candidate_rule():
    failed_flat = SolverRunResult("flat", False, 10.0, 100)
    passed_flat = SolverRunResult("flat", True, 5.0, 7)
    failed_dc = SolverRunResult("dc", False, 6.0, 10)
    assert is_rescue_candidate(failed_flat) is True
    assert is_rescue_candidate(passed_flat) is False
    assert is_rescue_candidate(failed_dc) is False


def test_speed_summary():
    res = {
        "flat": SolverRunResult("flat", True, 20.0, 9),
        "dc": SolverRunResult("dc", True, 15.0, 7),
        "warmstart": SolverRunResult("warmstart", True, 10.0, 5),
    }
    out = compute_speed_summary(res)
    assert out["warmstart_speedup_vs_flat"] == 2.0
    assert out["warmstart_iterations"] == 5.0


def test_feasibility_summary():
    res = {
        "flat": SolverRunResult("flat", True, 20.0, 9, True, 1.0),
        "dc": SolverRunResult("dc", True, 15.0, 7, False, 0.7),
        "warmstart": SolverRunResult("warmstart", False, 10.0, 100, False, 0.0),
    }
    out = compute_feasibility_summary(res)
    assert out["flat_feasible"] == 1.0
    assert out["dc_constraint_rate"] == 0.7
    assert out["warmstart_converged"] == 0.0
