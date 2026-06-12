import glob
import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np

from src.benchmarks.benchmark_state import build_sample_id
from src.constants import TRAIN_RATIO, VAL_RATIO


DEFAULT_RENEWABLE_FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def parse_fraction_from_path(path: str) -> float:
    match = re.search(r"frac([\d.]+)", os.path.basename(path))
    if not match:
        raise ValueError(f"Could not parse renewable fraction from path: {path}")
    val = match.group(1)
    if val.endswith("."):
        val = val[:-1]
    return float(val)


def infer_samples_per_fraction(test_per_fraction: int, train_ratio: float = TRAIN_RATIO, val_ratio: float = VAL_RATIO) -> int:
    """Recover original timesteps-per-fraction from the test split size."""
    for n in range(test_per_fraction, 200_000):
        val_end = int(n * (train_ratio + val_ratio))
        if n - val_end == test_per_fraction:
            return n
    raise ValueError(
        f"Could not infer samples-per-fraction for test_per_fraction={test_per_fraction}. "
        "Provide raw data or sample_index.json."
    )


def build_test_metadata_from_raw(
    case_name: str,
    raw_dir: str,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> List[Dict[str, Any]]:
    from scripts.preprocess_data import get_case_files, time_based_split

    feat_files, _, topo_files = get_case_files(raw_dir, case_name)
    if not feat_files:
        return []

    entries: List[Dict[str, Any]] = []
    for ff, topf in zip(feat_files, topo_files):
        frac = parse_fraction_from_path(ff)
        features = np.load(ff, mmap_mode="r")
        topology_ids = np.load(topf, mmap_mode="r")
        n_samples = features.shape[0]
        _, val_end = time_based_split(n_samples, train_ratio, val_ratio)
        for t in range(val_end, n_samples):
            entries.append(
                {
                    "sample_id": build_sample_id(case_name, frac, t),
                    "case_name": case_name,
                    "timestep": int(t),
                    "renewable_fraction": float(frac),
                    "topology_id": int(topology_ids[t]),
                }
            )
    return entries


def build_test_metadata_from_prep(
    case_name: str,
    prep_dir: str,
    renewable_fractions: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """
    Reconstruct test-set metadata from processed tensors when raw .npy files are absent.
    Assumes the same chronological split and fraction ordering used in preprocess_data.py.
    """
    case_dir = os.path.join(prep_dir, case_name)
    meta_path = os.path.join(case_dir, "normalization.json")
    topo_path = os.path.join(case_dir, "test_topology_ids.pt")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    import torch

    topology_ids = torch.load(topo_path, weights_only=True).numpy()
    test_size = int(meta["splits"]["test"])
    num_fractions = int(meta.get("num_fractions", len(DEFAULT_RENEWABLE_FRACTIONS)))
    if test_size % num_fractions != 0:
        raise ValueError(
            f"Test size {test_size} is not divisible by num_fractions={num_fractions} for {case_name}."
        )

    fractions = renewable_fractions or DEFAULT_RENEWABLE_FRACTIONS[:num_fractions]
    if len(fractions) < num_fractions:
        raise ValueError(f"Need at least {num_fractions} renewable fractions, got {len(fractions)}.")

    test_per_fraction = test_size // num_fractions
    samples_per_fraction = infer_samples_per_fraction(test_per_fraction)
    val_end = int(samples_per_fraction * (TRAIN_RATIO + VAL_RATIO))

    entries: List[Dict[str, Any]] = []
    for frac_idx in range(num_fractions):
        frac = float(fractions[frac_idx])
        for local_idx in range(test_per_fraction):
            global_idx = frac_idx * test_per_fraction + local_idx
            timestep = val_end + local_idx
            entries.append(
                {
                    "sample_id": build_sample_id(case_name, frac, timestep),
                    "case_name": case_name,
                    "timestep": int(timestep),
                    "renewable_fraction": frac,
                    "topology_id": int(topology_ids[global_idx]),
                }
            )
    return entries


def select_coverage_indices(
    metadata: List[Dict[str, Any]],
    include_topology: bool = True,
    max_topology: int = 3,
) -> List[int]:
    """
    Pick a small, representative test subset:
      - one sample per renewable fraction (0%, 20%, ..., 100%)
      - optionally extra samples where topology_id > 0 (contingency)
    """
    selected: List[int] = []
    seen_fracs: set = set()

    for i, row in enumerate(metadata):
        frac = round(float(row["renewable_fraction"]), 1)
        if frac not in seen_fracs:
            selected.append(i)
            seen_fracs.add(frac)

    if include_topology:
        topo_added = 0
        for i, row in enumerate(metadata):
            if int(row["topology_id"]) > 0 and i not in selected:
                selected.append(i)
                topo_added += 1
                if topo_added >= max_topology:
                    break

    return sorted(selected)


def load_test_sample_metadata(case_name: str, project_root: str) -> List[Dict[str, Any]]:
    raw_dir = os.path.join(project_root, "data", "raw")
    prep_dir = os.path.join(project_root, "data", "prep")
    index_path = os.path.join(prep_dir, case_name, "test_sample_index.json")

    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    entries = build_test_metadata_from_raw(case_name, raw_dir)
    if entries:
        return entries

    return build_test_metadata_from_prep(case_name, prep_dir)
