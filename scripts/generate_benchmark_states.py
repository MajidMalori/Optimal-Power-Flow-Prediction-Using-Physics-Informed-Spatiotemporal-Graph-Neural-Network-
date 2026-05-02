#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable, **_kwargs):
        return iterable

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def build_states_from_case(case_name: str, max_samples: int | None = None):
    from src.models.data_module import PowerFlowDataModule
    from src.benchmarks.state_builder import build_states_from_dataloader_batch
    dm = PowerFlowDataModule(
        data_dir=os.path.join(PROJECT_ROOT, "data", "prep"),
        case_name=case_name,
        batch_size=1,
        seq_len=1,
    )
    dm.setup(stage="test")
    loader = dm.test_dataloader()

    states = []
    timestep = 0
    for batch in tqdm(loader, desc=f"Building states {case_name}", leave=False):
        built = build_states_from_dataloader_batch(
            case_name=case_name,
            batch=batch,
            start_timestep=timestep,
            renewable_fraction=0.0,
        )
        states.extend(built)
        timestep += len(built)
        if max_samples is not None and len(states) >= max_samples:
            states = states[:max_samples]
            break
    return states


def main() -> None:
    from src.benchmarks.benchmark_dataset import save_states_jsonl
    parser = argparse.ArgumentParser(description="Generate canonical benchmark states from prepared data.")
    parser.add_argument("--case", type=str, required=True, help="case33, case57, case118, or all")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on exported states per case")
    args = parser.parse_args()

    if args.case.lower() == "all":
        cases = ["case33", "case57", "case118"]
    else:
        cases = [args.case if args.case.startswith("case") else f"case{args.case}"]

    for case in cases:
        states = build_states_from_case(case, max_samples=args.max_samples)
        out_path = os.path.join(PROJECT_ROOT, "data", "benchmark", case, "states.jsonl")
        save_states_jsonl(states, out_path)
        print(f"[{case}] wrote {len(states)} states -> {out_path}")


if __name__ == "__main__":
    main()
