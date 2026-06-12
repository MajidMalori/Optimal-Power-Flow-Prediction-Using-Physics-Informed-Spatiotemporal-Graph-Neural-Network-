from src.benchmarks.benchmark_state import BenchmarkState
from src.benchmarks.rescue_runner import run_rescue_benchmark


def test_rescue_runner_only_counts_flat_failed():
    states = [
        BenchmarkState(
            sample_id="case33_rf020_t000001",
            case_name="case33",
            timestep=1,
            renewable_fraction=0.2,
            topology_id=0,
            features=[[0.1] * 11 for _ in range(3)],
            active_edges=[[0, 1], [1, 2]],
            metadata={},
        ),
        BenchmarkState(
            sample_id="case33_rf020_t000002",
            case_name="case33",
            timestep=2,
            renewable_fraction=0.2,
            topology_id=0,
            features=[[0.1] * 11 for _ in range(3)],
            active_edges=[[0, 1], [1, 2]],
            metadata={},
        ),
    ]

    def _flow(state):
        if state.timestep == 1:
            return {"flat": {"converged": True}}
        return {
            "flat": {"converged": False},
            "dc": {"converged": False},
            "warmstart": {"converged": True},
        }

    records, summary = run_rescue_benchmark(states, _flow)
    assert len(records) == 1
    assert records[0]["sample_id"].endswith("000002")
    assert summary["warmstart_recovery_rate"] == 1.0
