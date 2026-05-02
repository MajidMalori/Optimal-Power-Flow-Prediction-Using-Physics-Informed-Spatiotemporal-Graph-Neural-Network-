from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass(frozen=True)
class BenchmarkState:
    """
    Canonical one-timestep state shared by model-input and NR-input paths.
    """
    sample_id: str
    case_name: str
    timestep: int
    renewable_fraction: float
    topology_id: int
    features: List[List[float]]
    active_edges: List[List[int]]
    metadata: Dict[str, Any]


def build_sample_id(case_name: str, renewable_fraction: float, timestep: int) -> str:
    rf = int(round(renewable_fraction * 100))
    return f"{case_name}_rf{rf:03d}_t{timestep:06d}"


def validate_state(state: BenchmarkState) -> None:
    if not state.case_name.startswith("case"):
        raise ValueError("case_name must look like case33/case57/case118")
    if state.timestep < 0:
        raise ValueError("timestep must be non-negative")
    if not (0.0 <= state.renewable_fraction <= 1.0):
        raise ValueError("renewable_fraction must be between 0 and 1")
    if len(state.features) == 0:
        raise ValueError("features cannot be empty")
    if len(state.active_edges) == 0:
        raise ValueError("active_edges cannot be empty")
