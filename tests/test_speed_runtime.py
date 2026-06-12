from src.benchmarks.benchmark_state import BenchmarkState
from src.benchmarks import speed_runtime


class _FakeEvaluator:
    def __init__(self, net, case_name, max_iter=100, tolerance=1e-5):
        self.net = net
        self.case_name = case_name

    def evaluate_sample(self, **_kwargs):
        return {
            "flat": {"success": True, "time_ms": 20.0, "iterations": 8},
            "dc": {"success": True, "time_ms": 16.0, "iterations": 7},
            "results": {"success": True, "time_ms": 10.0, "iterations": 5},
        }


def test_run_all_methods_for_state_with_mocked_evaluator(monkeypatch):
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

    out = speed_runtime.run_all_methods_for_state(
        state,
        load_network_fn=lambda _case: object(),
        evaluator_cls=_FakeEvaluator,
    )
    assert out["flat"].iterations == 8
    assert out["dc"].time_ms == 16.0
    assert out["warmstart"].converged is True
