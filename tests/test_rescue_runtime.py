from src.benchmarks.benchmark_state import BenchmarkState
from src.benchmarks import rescue_runtime
from src.benchmarks.warmstart_protocol import SolverRunResult


def test_rescue_runtime_returns_only_flat_when_flat_converges(monkeypatch):
    def _run_all(_state, max_iter=100, tolerance=1e-5):
        return {
            "flat": SolverRunResult("flat", True, 20.0, 8),
            "dc": SolverRunResult("dc", True, 16.0, 7),
            "warmstart": SolverRunResult("warmstart", True, 10.0, 5),
        }

    monkeypatch.setattr(rescue_runtime, "run_all_methods_for_state", _run_all)
    state = BenchmarkState(
        sample_id="case33_rf020_t000001",
        case_name="case33",
        timestep=1,
        renewable_fraction=0.2,
        topology_id=0,
        features=[[0.1] * 11 for _ in range(3)],
        active_edges=[[0, 1], [1, 2]],
        metadata={},
    )
    out = rescue_runtime.run_rescue_flow_for_state(state)
    assert set(out.keys()) == {"flat"}


def test_rescue_runtime_returns_recovery_methods_when_flat_fails(monkeypatch):
    def _run_all(_state, max_iter=100, tolerance=1e-5):
        return {
            "flat": SolverRunResult("flat", False, 20.0, 100),
            "dc": SolverRunResult("dc", False, 16.0, 100),
            "warmstart": SolverRunResult("warmstart", True, 10.0, 5),
        }

    monkeypatch.setattr(rescue_runtime, "run_all_methods_for_state", _run_all)
    state = BenchmarkState(
        sample_id="case33_rf020_t000002",
        case_name="case33",
        timestep=2,
        renewable_fraction=0.2,
        topology_id=0,
        features=[[0.1] * 11 for _ in range(3)],
        active_edges=[[0, 1], [1, 2]],
        metadata={},
    )
    out = rescue_runtime.run_rescue_flow_for_state(state)
    assert set(out.keys()) == {"flat", "dc", "warmstart"}
    assert out["warmstart"]["converged"] is True
