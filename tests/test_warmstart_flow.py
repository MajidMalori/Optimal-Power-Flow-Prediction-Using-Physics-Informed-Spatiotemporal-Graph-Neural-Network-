from src.benchmarks.warmstart_flow import evaluate_all_methods, evaluate_rescue_flow
from src.benchmarks.warmstart_protocol import SolverRunResult


def test_evaluate_all_methods_runs_all_three():
    called = []

    def _run(method: str):
        called.append(method)
        return SolverRunResult(method=method, converged=True, time_ms=1.0, iterations=1)

    out = evaluate_all_methods(_run)
    assert set(out.keys()) == {"flat", "dc", "warmstart"}
    assert called == ["flat", "dc", "warmstart"]


def test_rescue_flow_skips_recovery_if_flat_converges():
    called = []

    def _run(method: str):
        called.append(method)
        return SolverRunResult(method=method, converged=True, time_ms=1.0, iterations=1)

    out = evaluate_rescue_flow(_run)
    assert set(out.keys()) == {"flat"}
    assert called == ["flat"]


def test_rescue_flow_runs_dc_and_warmstart_when_flat_fails():
    called = []

    def _run(method: str):
        called.append(method)
        converged = method != "flat"
        return SolverRunResult(method=method, converged=converged, time_ms=1.0, iterations=1)

    out = evaluate_rescue_flow(_run)
    assert set(out.keys()) == {"flat", "dc", "warmstart"}
    assert called == ["flat", "dc", "warmstart"]
