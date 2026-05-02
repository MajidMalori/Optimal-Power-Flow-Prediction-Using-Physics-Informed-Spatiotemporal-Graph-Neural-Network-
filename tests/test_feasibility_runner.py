from src.benchmarks.benchmark_state import BenchmarkState
from src.benchmarks.feasibility_runner import run_feasibility_benchmark
from src.benchmarks.warmstart_protocol import SolverRunResult


def test_feasibility_runner_records_and_summary():
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
            "flat": SolverRunResult("flat", True, 20.0, 8, True, 1.0),
            "dc": SolverRunResult("dc", True, 16.0, 7, False, 0.5),
            "warmstart": SolverRunResult("warmstart", True, 10.0, 5, True, 1.0),
        }

    records, summary = run_feasibility_benchmark(states, _run_all)
    assert len(records) == 3
    assert summary["flat_feasibility_rate"] == 1.0
    assert summary["dc_constraint_rate_mean"] == 0.5
