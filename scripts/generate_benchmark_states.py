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

def _case_prep_dir(case_name: str) -> str:
    return os.path.join(PROJECT_ROOT, "data", "prep", case_name)


def _require_prep_artifacts(case_name: str) -> None:
    """
    Produce a helpful error if a user forgot to copy/regenerate prepared data
    (common when migrating machines).
    """
    prep_dir = _case_prep_dir(case_name)
    normalization_path = os.path.join(prep_dir, "normalization.json")
    if not os.path.isdir(prep_dir):
        raise FileNotFoundError(
            f"Missing prepared data directory: {prep_dir}\n"
            f"Generate/preprocess the dataset for {case_name} first (so it creates "
            f"`normalization.json`, `*_features.pt`, `*_targets.pt`, etc.)."
        )
    if not os.path.exists(normalization_path):
        raise FileNotFoundError(
            f"Missing prepared artifact: {normalization_path}\n"
            f"Generate/preprocess the dataset for {case_name} first."
        )


def build_states_from_case(case_name: str, max_samples: int | None = None):
    from src.models.data_module import PowerFlowDataModule
    from src.benchmarks.state_builder import build_states_from_dataloader_batch
    _require_prep_artifacts(case_name)
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
    total = None
    try:
        total = len(loader)
    except TypeError:
        total = None
    if max_samples is not None and total is not None:
        total = min(total, max_samples)

    progress_total = max_samples if max_samples is not None else total
    bar = tqdm(
        total=progress_total,
        desc=f"Exporting benchmark states ({case_name})",
        leave=True,
        dynamic_ncols=True,
    )
    for batch in loader:
        built = build_states_from_dataloader_batch(
            case_name=case_name,
            batch=batch,
            start_timestep=timestep,
            renewable_fraction=0.0,
        )
        before = len(states)
        states.extend(built)
        timestep += len(built)
        if progress_total is None:
            bar.update(len(built))
        else:
            remaining = max(0, progress_total - before)
            bar.update(min(len(built), remaining))
        if max_samples is not None and len(states) >= max_samples:
            states = states[:max_samples]
            break
    bar.close()
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

    cases_iter = tqdm(cases, desc="Cases", leave=True, dynamic_ncols=True) if len(cases) > 1 else cases
    for case in cases_iter:
        try:
            states = build_states_from_case(case, max_samples=args.max_samples)
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            raise SystemExit(2) from e
        out_path = os.path.join(PROJECT_ROOT, "data", "benchmark", case, "states.jsonl")
        save_states_jsonl(states, out_path)
        print(f"[{case}] wrote {len(states)} states -> {out_path}")


if __name__ == "__main__":
    main()
