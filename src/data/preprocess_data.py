"""
Data Preprocessing Pipeline
Reads raw .npy from 01_raw, applies per-unit normalization,
time-based train/val/test splits, and saves .pt tensors to 03_processed.
"""

import os
import sys
import json
import glob
import yaml
import numpy as np
import torch
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
# script_dir is src/data, parent is src, root is one level up
root_dir = os.path.dirname(os.path.dirname(script_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.constants import (
    FeatureIndices, TargetIndices, TRAIN_RATIO, VAL_RATIO
)
NUM_FEATURES = FeatureIndices.NUM_FEATURES
NUM_TARGETS = FeatureIndices.NUM_TARGETS

RAW_DIR = os.path.join(script_dir, '01_raw')
PROCESSED_DIR = os.path.join(script_dir, '03_processed')

def load_config():
    yaml_path = os.path.join(root_dir, 'configs', 'data_generation.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

# Feature indices that represent power quantities (MW/MVar) needing per-unit scaling
POWER_FEATURE_INDICES = [
    FeatureIndices.P_LOAD, FeatureIndices.Q_LOAD,
    FeatureIndices.P_EXT_GRID, FeatureIndices.Q_EXT_GRID,
    FeatureIndices.P_CONV, FeatureIndices.Q_CONV,
    FeatureIndices.P_REN, FeatureIndices.Q_REN,
]

# Target indices 0-7 are the same power quantities
POWER_TARGET_INDICES = list(range(8))


def get_case_files(raw_dir: str, case_name: str):
    """Discover all fraction files for a case, sorted by fraction value."""
    feat_files = sorted(glob.glob(os.path.join(raw_dir, f"{case_name}_features_frac*.npy")))
    targ_files = sorted(glob.glob(os.path.join(raw_dir, f"{case_name}_targets_frac*.npy")))
    topo_files = sorted(glob.glob(os.path.join(raw_dir, f"{case_name}_topology_ids_frac*.npy")))
    return feat_files, targ_files, topo_files


def per_unit_normalize(features: np.ndarray, targets: np.ndarray, s_base: float, max_degree: float):
    """
    Apply per-unit normalization in-place.
    Power columns (MW/MVar) are divided by S_base.
    Degree is divided by max_degree.
    V_m is mean-centered by subtracting 1.0 (nominal voltage).
    V_a (radians) is left untouched (already small-scale).
    """
    features = features.copy()
    targets = targets.copy()

    for idx in POWER_FEATURE_INDICES:
        features[:, :, idx] /= s_base

    features[:, :, FeatureIndices.DEGREE] /= max_degree

    # Mean-center voltage magnitude around nominal (1.0 p.u.)
    features[:, :, FeatureIndices.VM] -= 1.0

    for idx in POWER_TARGET_INDICES:
        targets[:, :, idx] /= s_base

    # Also center target V_m
    targets[:, :, TargetIndices.VM] -= 1.0

    return features, targets


def denormalize_predictions(preds: np.ndarray, meta: dict) -> np.ndarray:
    """
    Reverse the normalization applied during preprocessing.
    Call this on model outputs before comparing to real-world values.

    Args:
        preds: Model predictions array [..., num_targets]
        meta: The normalization.json metadata dict

    Returns:
        Denormalized predictions in original physical units.
    """
    preds = preds.copy()
    s_base = meta['s_base']

    for idx in meta['power_target_indices']:
        preds[..., idx] *= s_base

    # Restore V_m from deviation back to absolute per-unit
    preds[..., TargetIndices.VM] += meta.get('vm_nominal', 1.0)

    return preds


def time_based_split(n_samples: int, train_ratio=None, val_ratio=None):
    """Returns (train_end, val_end) indices for chronological split."""
    if train_ratio is None: train_ratio = TRAIN_RATIO
    if val_ratio is None: val_ratio = VAL_RATIO
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    return train_end, val_end


def load_ybus_data(raw_dir: str, case_name: str):
    """Load base Ybus and contingency Ybus matrices."""
    ybus_base_files = sorted(glob.glob(os.path.join(raw_dir, f"{case_name}_ybus_base_frac*.npy")))
    ybus_cont_files = sorted(glob.glob(os.path.join(raw_dir, f"{case_name}_ybus_contingency_matrices_frac*.npy")))
    ybus_ts_files = sorted(glob.glob(os.path.join(raw_dir, f"{case_name}_ybus_contingency_timesteps_frac*.npy")))

    # Use first fraction's base Ybus (topology is the same across fractions)
    ybus_base = np.load(ybus_base_files[0]) if ybus_base_files else None

    # Collect all contingency matrices across fractions
    all_cont_matrices = []
    all_cont_timesteps = []
    for cf, tf in zip(ybus_cont_files, ybus_ts_files):
        cm = np.load(cf)
        ct = np.load(tf)
        if cm.size > 0:
            all_cont_matrices.append(cm)
            all_cont_timesteps.append(ct)

    contingency_matrices = np.concatenate(all_cont_matrices) if all_cont_matrices else np.array([])
    contingency_timesteps = np.concatenate(all_cont_timesteps) if all_cont_timesteps else np.array([])

    return ybus_base, contingency_matrices, contingency_timesteps


def load_adjacency(raw_dir: str, case_name: str):
    """Load base adjacency edge index."""
    adj_files = sorted(glob.glob(os.path.join(raw_dir, f"{case_name}_base_adjacency_frac*.npy")))
    if adj_files:
        return np.load(adj_files[0], allow_pickle=True)
    return None


def preprocess_case(case_name: str, raw_dir: str, processed_dir: str, config: dict = None):
    """Full preprocessing pipeline for one bus case."""
    # Source s_base from SYSTEM_PHYSICS constants
    from src.constants import SYSTEM_PHYSICS
    physics = SYSTEM_PHYSICS.get(case_name, SYSTEM_PHYSICS['default'])
    s_base = physics['base_mva']
    
    train_r = config.get('train_ratio', TRAIN_RATIO) if config else TRAIN_RATIO
    val_r = config.get('val_ratio', VAL_RATIO) if config else VAL_RATIO
    
    feat_files, targ_files, topo_files = get_case_files(raw_dir, case_name)
    if not feat_files:
        print(f"  No raw data for {case_name}")
        return False

    case_dir = os.path.join(processed_dir, case_name)
    if os.path.exists(case_dir):
        import shutil
        shutil.rmtree(case_dir)
    os.makedirs(case_dir, exist_ok=True)



    # Collect splits across all fractions
    train_feats, train_targs, train_topos = [], [], []
    val_feats, val_targs, val_topos = [], [], []
    test_feats, test_targs, test_topos = [], [], []

    max_degree = 1.0  # Will be computed from data

    # First pass: find global max degree
    for ff in feat_files:
        f = np.load(ff, mmap_mode='r')
        deg_max = f[:, :, FeatureIndices.DEGREE].max()
        max_degree = max(max_degree, float(deg_max))

    # Second pass: normalize, split, collect
    num_buses = case_name.replace('case', '')
    print(f"\n{num_buses}-bus | {len(feat_files)} fractions | s_base={s_base} MVA | max_degree={max_degree:.0f}")
    
    for ff, tf, topf in tqdm(zip(feat_files, targ_files, topo_files),
                              total=len(feat_files), desc=f"  Processing {case_name}",
                              bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} fractions"):
        features = np.load(ff)
        targets = np.load(tf)
        topology_ids = np.load(topf)

        n_samples, _, n_feats = features.shape
        if n_feats != NUM_FEATURES:
            print(f"  Warning: {ff} has {n_feats} features, but expected {NUM_FEATURES}. Skipping.")
            continue

        features, targets = per_unit_normalize(features, targets, s_base, max_degree)

        train_end, val_end = time_based_split(n_samples, train_r, val_r)

        train_feats.append(features[:train_end])
        train_targs.append(targets[:train_end])
        train_topos.append(topology_ids[:train_end])

        val_feats.append(features[train_end:val_end])
        val_targs.append(targets[train_end:val_end])
        val_topos.append(topology_ids[train_end:val_end])

        test_feats.append(features[val_end:])
        test_targs.append(targets[val_end:])
        test_topos.append(topology_ids[val_end:])

    # Concatenate all fractions per split
    splits = {
        'train': (np.concatenate(train_feats), np.concatenate(train_targs), np.concatenate(train_topos)),
        'val':   (np.concatenate(val_feats),   np.concatenate(val_targs),   np.concatenate(val_topos)),
        'test':  (np.concatenate(test_feats),  np.concatenate(test_targs),  np.concatenate(test_topos)),
    }

    # Save as .pt tensors
    for split_name, (feats, targs, topos) in splits.items():
        torch.save(torch.from_numpy(feats), os.path.join(case_dir, f'{split_name}_features.pt'))
        torch.save(torch.from_numpy(targs), os.path.join(case_dir, f'{split_name}_targets.pt'))
        torch.save(torch.from_numpy(topos), os.path.join(case_dir, f'{split_name}_topology_ids.pt'))

    # Save Ybus data
    ybus_base, cont_matrices, cont_timesteps = load_ybus_data(raw_dir, case_name)
    if ybus_base is not None:
        # Normalize Ybus by S_base (convert to per-unit admittance)
        torch.save(torch.from_numpy(ybus_base / s_base), os.path.join(case_dir, 'ybus_base.pt'))
        if cont_matrices.size > 0:
            torch.save(torch.from_numpy(cont_matrices / s_base), os.path.join(case_dir, 'ybus_contingencies.pt'))
            torch.save(torch.from_numpy(cont_timesteps), os.path.join(case_dir, 'ybus_contingency_timesteps.pt'))

    # Save adjacency edge index
    adj = load_adjacency(raw_dir, case_name)
    if adj is not None:
        # Adjacency might be nested or object array; normalize to [2, E] long tensor
        if adj.ndim == 3 and adj.shape[0] == 1:
            adj = adj[0]
        torch.save(torch.from_numpy(adj.astype(np.int64)), os.path.join(case_dir, 'adjacency.pt'))

    # Save normalization metadata for denormalization at inference
    meta = {
        's_base': s_base,
        'max_degree': max_degree,
        'vm_nominal': 1.0,
        'num_features': NUM_FEATURES,
        'num_targets': NUM_TARGETS,
        'power_feature_indices': [int(i) for i in POWER_FEATURE_INDICES],
        'power_target_indices': POWER_TARGET_INDICES,
        'splits': {
            'train': int(splits['train'][0].shape[0]),
            'val': int(splits['val'][0].shape[0]),
            'test': int(splits['test'][0].shape[0]),
        },
        'num_fractions': len(feat_files),
    }
    with open(os.path.join(case_dir, 'normalization.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess Spatio-Temporal Data")
    parser.add_argument('--case', '--cases', '--buses', type=str, dest='case', default=None, 
                        help="Comma separated list of cases (e.g., '33,57' or 'all')")
    args = parser.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    config = load_config()

    # Determine cases to process
    if args.case and args.case.lower() != 'all':
        # Handle '33, 57' or 'case33, case57'
        cases = [c.strip() if c.strip().startswith('case') else f"case{c.strip()}" 
                 for c in args.case.split(',')]
    else:
        cases = config.get('test_cases', ["case33", "case57", "case118"])


    
    success = 0
    processed_cases = []
    for case in cases:
        if preprocess_case(case, RAW_DIR, PROCESSED_DIR, config):
            success += 1
            processed_cases.append(case)

    # Generate diagnostic plots for all successfully processed cases
    if processed_cases:
        print()
        try:
            from src.visualization.plot_preprocessing import generate_all_preprocessing_plots
            reports_dir = os.path.join(root_dir, 'reports', 'figures', '03_processed')
            for case in processed_cases:
                case_dir = os.path.join(PROCESSED_DIR, case)
                generate_all_preprocessing_plots(case_dir, case, reports_dir)
        except Exception as e:
            print(f"Plot generation error: {e}")

    print(f"\nPipeline complete. Processed {success}/{len(cases)} cases.")
