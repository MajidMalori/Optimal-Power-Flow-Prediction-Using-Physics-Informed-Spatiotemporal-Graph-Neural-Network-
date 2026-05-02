import torch

from src.benchmarks.state_builder import (
    build_state_from_batch_item,
    build_states_from_dataloader_batch,
)


def test_build_state_from_batch_item():
    features = torch.tensor([[0.1] * 11, [0.2] * 11], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    state = build_state_from_batch_item(
        case_name="case33",
        timestep=7,
        renewable_fraction=0.4,
        topology_id=2,
        features=features,
        edge_index=edge_index,
        metadata={"noise_seed": 1},
    )
    assert state.sample_id == "case33_rf040_t000007"
    assert len(state.features) == 2
    assert state.active_edges == [[0, 1], [1, 0]]


def test_build_states_from_dataloader_batch():
    batch = {
        "features": torch.rand(2, 3, 11),
        "topology_ids": torch.tensor([0, 3], dtype=torch.int64),
        "edge_index": [
            torch.tensor([[0, 1], [1, 2]], dtype=torch.int64),
            torch.tensor([[0, 2], [2, 1]], dtype=torch.int64),
        ],
    }
    states = build_states_from_dataloader_batch(
        case_name="case57",
        batch=batch,
        start_timestep=10,
        renewable_fraction=0.0,
    )
    assert len(states) == 2
    assert states[0].timestep == 10
    assert states[1].timestep == 11
    assert states[1].topology_id == 3
