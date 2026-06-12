from typing import Callable, Dict, List, Tuple

from src.benchmarks.benchmark_state import BenchmarkState
from src.benchmarks.metrics import aggregate_rescue


RescueFlowFn = Callable[[BenchmarkState], Dict]


def run_rescue_benchmark(
    states: List[BenchmarkState],
    rescue_flow_fn: RescueFlowFn,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Rescue pool is flat-failed samples only.
    Records contain candidate rows where flat failed.
    """
    records: List[Dict] = []
    for state in states:
        out = rescue_flow_fn(state)
        flat = out["flat"]
        if flat["converged"]:
            continue
        records.append(
            {
                "sample_id": state.sample_id,
                "case_name": state.case_name,
                "timestep": state.timestep,
                "renewable_fraction": state.renewable_fraction,
                "flat_failed": True,
                "dc_recovered": bool(out.get("dc", {}).get("converged", False)),
                "warmstart_recovered": bool(out.get("warmstart", {}).get("converged", False)),
            }
        )
    return records, aggregate_rescue(records)
