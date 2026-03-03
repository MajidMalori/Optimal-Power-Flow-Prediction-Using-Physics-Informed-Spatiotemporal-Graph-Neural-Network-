import os
import sys
import glob

import numpy as np
import pytest

# Add parent directory to path to import constants
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import FeatureIndices, V_GARBAGE_LOW, V_GARBAGE_HIGH


DATA_DIR = os.path.join("data", "01_raw")

def get_feature_files():
    pattern = os.path.join(DATA_DIR, "*_features_frac*.npy")
    return sorted(glob.glob(pattern))

@pytest.mark.parametrize("filepath", get_feature_files())
def test_data_shapes_and_nans(filepath):
    features = np.load(filepath)
    assert features.ndim == 3, f"Expected 3D array [samples, buses, features], got {features.shape}"
    assert features.shape[2] == FeatureIndices.NUM_FEATURES, f"Expected {FeatureIndices.NUM_FEATURES} features, got {features.shape[2]}"
    
    # Check for NaNs and Infs
    assert not np.isnan(features).any(), f"NaNs found in {filepath}"
    assert not np.isinf(features).any(), f"Infinities found in {filepath}"

@pytest.mark.parametrize("filepath", get_feature_files())
def test_sign_conventions(filepath):
    features = np.load(filepath)
    
    p_load = features[:, :, FeatureIndices.P_LOAD]
    p_ren = features[:, :, FeatureIndices.P_REN]
    
    # Loads should be non-negative (consuming power)
    assert np.all(p_load >= -1e-6), f"Negative loads found in {filepath}"
    
    # Renewable generation should be non-negative (injecting power)
    assert np.all(p_ren >= -1e-6), f"Negative renewable generation found in {filepath}"

@pytest.mark.parametrize("filepath", get_feature_files())
def test_physical_bounds(filepath):
    
    features = np.load(filepath)
    vm = features[:, :, FeatureIndices.VM]
    
    # Ignore missing data points (0.0) which are handled correctly in modeling
    valid_vm = vm[vm > 0.1]
    
    if len(valid_vm) > 0:
        assert np.all(valid_vm >= V_GARBAGE_LOW), f"Voltage below garbage limit {V_GARBAGE_LOW} in {filepath}"
        assert np.all(valid_vm <= V_GARBAGE_HIGH), f"Voltage above garbage limit {V_GARBAGE_HIGH} in {filepath}"

@pytest.mark.parametrize("filepath", get_feature_files())
def test_power_balance(filepath):
    features = np.load(filepath)
    
    p_load = features[:, :, FeatureIndices.P_LOAD]
    p_ext = features[:, :, FeatureIndices.P_EXT_GRID]
    p_conv = features[:, :, FeatureIndices.P_CONV]
    p_ren = features[:, :, FeatureIndices.P_REN]
    
    total_load = np.sum(p_load, axis=1)
    total_gen = np.sum(p_ext + p_conv + p_ren, axis=1)
    
    # Generation must be strictly greater than or equal to load (to account for losses)
    assert np.all(total_gen >= total_load - 1e-4), f"Total generation is less than total load (negative losses) in {filepath}"
    
    # Losses shouldn't be ridiculously high (e.g., > 100% of load) in a stable grid
    losses = total_gen - total_load
    assert np.all(losses <= total_load * 1.5), f"Line losses are unphysically high in {filepath}"
