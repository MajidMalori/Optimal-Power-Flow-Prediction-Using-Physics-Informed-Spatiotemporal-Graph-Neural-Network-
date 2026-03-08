# pylint: disable=duplicate-code
import json
import os
import sys

import pytest
import torch


from constants import (  # pylint: disable=wrong-import-position
    FeatureIndices, TargetIndices, 
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)

NUM_FEATURES = FeatureIndices.NUM_FEATURES
NUM_TARGETS = TargetIndices.NUM_TARGETS

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(parent_dir, 'data', '03_processed')


@pytest.fixture
def case_path(case_name):
    """Fixture to provide the directory for the current case."""
    return os.path.join(PROCESSED_DIR, case_name)

def test_split_files_exist(case_path, case_name):
    """Verify all expected .pt files and normalization.json exist."""
    path = case_path
    name = case_name
    for split in ['train', 'val', 'test']:
        assert os.path.exists(os.path.join(path, f'{split}_features.pt')), f"Missing {split}_features.pt for {name}"
        assert os.path.exists(os.path.join(path, f'{split}_targets.pt')), f"Missing {split}_targets.pt for {name}"
        assert os.path.exists(os.path.join(path, f'{split}_topology_ids.pt')), f"Missing {split}_topology_ids.pt for {name}"
    assert os.path.exists(os.path.join(path, 'normalization.json')), f"Missing normalization.json for {name}"


def test_feature_target_shapes(case_path, case_name):
    """Verify feature shape is [T, N, 11] and target shape is [T, N, 10]."""
    path = case_path
    name = case_name
    for split in ['train', 'val', 'test']:
        feats = torch.load(os.path.join(path, f'{split}_features.pt'), weights_only=True)
        targs = torch.load(os.path.join(path, f'{split}_targets.pt'), weights_only=True)

        assert feats.dim() == 3, f"{name}/{split} features should be 3D"
        assert targs.dim() == 3, f"{name}/{split} targets should be 3D"
        assert feats.shape[0] == targs.shape[0], f"{name}/{split} sample count mismatch"
        assert feats.shape[1] == targs.shape[1], f"{name}/{split} bus count mismatch"
        assert feats.shape[2] == NUM_FEATURES, f"{name}/{split} expected {NUM_FEATURES} features, got {feats.shape[2]}"
        assert targs.shape[2] == NUM_TARGETS, f"{name}/{split} expected {NUM_TARGETS} targets, got {targs.shape[2]}"


def test_no_nans(case_path, case_name):
    """Verify no NaN values leaked into processed data."""
    path = case_path
    name = case_name
    for split in ['train', 'val', 'test']:
        feats = torch.load(os.path.join(path, f'{split}_features.pt'), weights_only=True)
        targs = torch.load(os.path.join(path, f'{split}_targets.pt'), weights_only=True)
        assert not torch.isnan(feats).any(), f"NaN in {name}/{split} features"
        assert not torch.isnan(targs).any(), f"NaN in {name}/{split} targets"


def test_per_unit_voltage_range(case_path, case_name):
    """Verify voltage magnitudes (mean-centered around 1.0 p.u.) are in reasonable range."""
    path = case_path
    name = case_name
    feats = torch.load(os.path.join(path, 'train_features.pt'), weights_only=True)
    # VM is centered: actual_vm = stored_vm + 1.0
    # So stored values should be roughly in [-0.3, +0.3] (i.e. 0.7–1.3 p.u.)
    # Exclude unobserved buses where original vm was 0 (now -1.0 after centering)
    vm_centered = feats[:, :, FeatureIndices.VM]
    observed = vm_centered[vm_centered > -0.9]
    if observed.numel() > 0:
        assert observed.min() > -0.5, f"{name} voltage deviation too low: {observed.min():.3f}"
        assert observed.max() < 0.5, f"{name} voltage deviation too high: {observed.max():.3f}"


def test_normalization_metadata(case_path, case_name):
    """Verify normalization.json contains all required fields."""
    path = case_path
    name = case_name
    with open(os.path.join(path, 'normalization.json')) as f:
        meta = json.load(f)

    assert 's_base' in meta, f"Missing s_base in {name}"
    assert 'max_degree' in meta, f"Missing max_degree in {name}"
    assert meta['s_base'] > 0, f"Invalid s_base in {name}"
    assert meta['max_degree'] > 0, f"Invalid max_degree in {name}"
    assert meta['num_features'] == NUM_FEATURES
    assert meta['num_targets'] == NUM_TARGETS
    assert 'splits' in meta


def test_split_proportions(case_path, case_name):
    """Verify train/val/test ratios are approximately 70/15/15."""
    path = case_path
    name = case_name
    with open(os.path.join(path, 'normalization.json')) as f:
        meta = json.load(f)

    total = sum(meta['splits'].values())
    train_pct = meta['splits']['train'] / total
    val_pct = meta['splits']['val'] / total
    test_pct = meta['splits']['test'] / total

    # Check against constants with 5% tolerance
    assert abs(train_pct - TRAIN_RATIO) < 0.05, f"{name} train ratio {train_pct:.2f} mismatch"
    assert abs(val_pct - VAL_RATIO) < 0.05, f"{name} val ratio {val_pct:.2f} mismatch"
    assert abs(test_pct - TEST_RATIO) < 0.05, f"{name} test ratio {test_pct:.2f} mismatch"
