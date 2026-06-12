from src.benchmarks.benchmark_dataset import save_states_jsonl, load_states_jsonl
from src.benchmarks.benchmark_state import BenchmarkState


def test_jsonl_roundtrip(tmp_path):
    p = tmp_path / "bench" / "states.jsonl"
    states = [
        BenchmarkState(
            sample_id="case33_rf020_t000001",
            case_name="case33",
            timestep=1,
            renewable_fraction=0.2,
            topology_id=1,
            features=[[0.1] * 11 for _ in range(3)],
            active_edges=[[0, 1], [1, 2]],
            metadata={"noise_seed": 42},
        ),
        BenchmarkState(
            sample_id="case33_rf020_t000002",
            case_name="case33",
            timestep=2,
            renewable_fraction=0.2,
            topology_id=2,
            features=[[0.2] * 11 for _ in range(3)],
            active_edges=[[0, 2]],
            metadata={"noise_seed": 43},
        ),
    ]

    save_states_jsonl(states, str(p))
    loaded = load_states_jsonl(str(p))
    assert len(loaded) == 2
    assert loaded[0].sample_id == states[0].sample_id
    assert loaded[1].topology_id == 2
