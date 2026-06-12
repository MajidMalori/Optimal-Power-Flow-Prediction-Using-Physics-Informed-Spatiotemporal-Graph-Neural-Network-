from typing import Any, Dict, List

import torch

from src.benchmarks.benchmark_state import BenchmarkState, build_sample_id, validate_state


def _to_edge_pairs(edge_index: torch.Tensor) -> List[List[int]]:
    if edge_index.dim() != 2:
        raise ValueError("edge_index must be rank-2")
    if edge_index.shape[0] == 2:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        return [[int(u), int(v)] for u, v in zip(src, dst)]
    if edge_index.shape[1] == 2:
        return [[int(row[0]), int(row[1])] for row in edge_index.tolist()]
    raise ValueError("edge_index must have shape [2,E] or [E,2]")


def build_state_from_batch_item(
    case_name: str,
    timestep: int,
    renewable_fraction: float,
    topology_id: int,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    metadata: Dict[str, Any] | None = None,
) -> BenchmarkState:
    """
    Build one canonical benchmark state from one sample item.
    """
    state = BenchmarkState(
        sample_id=build_sample_id(case_name, renewable_fraction, timestep),
        case_name=case_name,
        timestep=int(timestep),
        renewable_fraction=float(renewable_fraction),
        topology_id=int(topology_id),
        features=features.detach().cpu().tolist(),
        active_edges=_to_edge_pairs(edge_index.detach().cpu()),
        metadata=metadata or {},
    )
    validate_state(state)
    return state


def build_states_from_dataloader_batch(
    case_name: str,
    batch: Dict[str, Any],
    start_timestep: int,
    renewable_fraction: float = 0.0,
) -> List[BenchmarkState]:
    """
    Convert one dataloader batch into canonical states.
    Works with the current DataModule collate format (batch_size >= 1, seq_len==1).
    """
    features_b = batch["features"]  # [B, N, F]
    topo_b = batch["topology_ids"]  # [B]
    edges_b = batch["edge_index"]   # list length B of [2,E]
    out: List[BenchmarkState] = []
    for i in range(features_b.shape[0]):
        out.append(
            build_state_from_batch_item(
                case_name=case_name,
                timestep=start_timestep + i,
                renewable_fraction=renewable_fraction,
                topology_id=int(topo_b[i].item()),
                features=features_b[i],
                edge_index=edges_b[i],
                metadata={
                    "source": "prep_test_loader",
                    "renewable_fraction_known": renewable_fraction != 0.0,
                },
            )
        )
    return out
