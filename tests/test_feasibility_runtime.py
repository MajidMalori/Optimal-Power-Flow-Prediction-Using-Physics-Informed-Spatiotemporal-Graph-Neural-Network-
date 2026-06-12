from src.benchmarks.benchmark_state import BenchmarkState
from src.benchmarks.feasibility_runtime import run_all_methods_for_state_feasibility


class _FakeEvaluator:
    def __init__(self, net, case_name, max_iter=100, tolerance=1e-5):
        self.net = net

    def evaluate_sample(self, **_kwargs):
        include_nets = _kwargs.get("include_nets", False)
        flat_net = object()
        dc_net = object()
        ws_net = object()
        return {
            "flat": {"success": True, "time_ms": 20.0, "iterations": 8, "net": flat_net if include_nets else None},
            "dc": {"success": True, "time_ms": 16.0, "iterations": 7, "net": dc_net if include_nets else None},
            "results": {"success": False, "time_ms": 10.0, "iterations": 100, "net": ws_net if include_nets else None},
        }


def test_feasibility_runtime_with_mocks():
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

    def _validate(net, _stats, case_name=None):
        assert case_name == "case33"
        # pretend flat is fully valid, dc has one soft-violation
        if net is not None and net.__class__ is object:
            pass
        return True, "Valid", {
            "voltage_violation": False,
            "angle_violation": False,
            "line_loading_violation": False,
            "slack_power_violation": False,
            "generator_capacity_violation": False,
            "inverter_capability_violation": False,
        }

    out = run_all_methods_for_state_feasibility(
        state,
        load_network_fn=lambda _case: object(),
        evaluator_cls=_FakeEvaluator,
        validate_outputs_fn=_validate,
    )
    assert out["flat"].is_feasible is True
    assert out["warmstart"].converged is False
    assert out["warmstart"].constraint_satisfaction_rate == 0.0
