import pytest

from src.benchmarks.benchmark_state import (
    BenchmarkState,
    build_sample_id,
    validate_state,
)


def test_build_sample_id():
    assert build_sample_id("case33", 0.2, 17) == "case33_rf020_t000017"


def test_validate_state_happy_path():
    state = BenchmarkState(
        sample_id="case33_rf020_t000017",
        case_name="case33",
        timestep=17,
        renewable_fraction=0.2,
        topology_id=3,
        features=[[0.1] * 11 for _ in range(33)],
        active_edges=[[0, 1], [1, 2]],
        metadata={"noise_seed": 42},
    )
    validate_state(state)  # no raise


@pytest.mark.parametrize(
    "kwargs",
    [
        {"case_name": "33"},
        {"timestep": -1},
        {"renewable_fraction": 1.5},
        {"features": []},
        {"active_edges": []},
    ],
)
def test_validate_state_failures(kwargs):
    base = dict(
        sample_id="case33_rf020_t000017",
        case_name="case33",
        timestep=17,
        renewable_fraction=0.2,
        topology_id=3,
        features=[[0.1] * 11 for _ in range(33)],
        active_edges=[[0, 1], [1, 2]],
        metadata={},
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        validate_state(BenchmarkState(**base))
