from src.benchmarks.benchmark_state import BenchmarkState
from src.benchmarks.speed_runner import run_speed_benchmark
from src.benchmarks.warmstart_protocol import SolverRunResult


def test_speed_runner_records_and_summary():
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
        )
    ]

    def _run_all(_state):
        return {
            "flat": SolverRunResult("flat", True, 20.0, 8),
            "dc": SolverRunResult("dc", True, 16.0, 7),
            "warmstart": SolverRunResult("warmstart", True, 10.0, 5),
        }

    records, summary = run_speed_benchmark(states, _run_all)
    assert len(records) == 3
    assert all(r["renewable_fraction"] == 0.2 for r in records)
    assert summary["warmstart_speedup_vs_flat"] == 2.0
