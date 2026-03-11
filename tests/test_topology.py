"""
Topology Degree Verification Test
Verifies that bus degrees change correctly during recorded switching events.
"""

import os
import numpy as np
import glob
import json
import sys
import pytest

# Add project root to path for constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.constants import FeatureIndices

@pytest.fixture
def data_dir():
    return "src/data/raw"

def test_degree_transitions(data_dir, case_name):
    print(f"--- Verifying Degree Transitions for {case_name} ---")
    
    # 1. Load feature files (containing the DEGREE information)
    feature_pattern = os.path.join(data_dir, f'{case_name}_features_frac*.npy')
    feature_files = sorted(glob.glob(feature_pattern))
    
    if not feature_files:
        print(f"No feature files found in {data_dir}")
        return

    # 2. Load switching metadata
    audit_pattern = os.path.join(data_dir, f'{case_name}_data_quality_audit_frac*.json')
    audit_files = sorted(glob.glob(audit_pattern))
    
    # 3. Analyze each audit file and its corresponding features
    total_events = 0
    success_count = 0
    
    for af in audit_files:
        # Load audit
        with open(af, 'r') as f:
            audit = json.load(f)
        
        events = audit.get('switching_events', [])
        if not events:
            continue
            
        # Load corresponding feature file
        frac_str = af.split('frac')[-1].split('.json')[0]
        ff = os.path.join(data_dir, f'{case_name}_features_frac{frac_str}.npy')
        
        if not os.path.exists(ff):
            continue
            
        features = np.load(ff)
        degrees = features[:, :, FeatureIndices.DEGREE] # [T_chunk, N]
        
        print(f"\nProcessing Fraction {frac_str} ({len(events)} events):")
        
        for event in events:
            total_events += 1
            # In main.py, e['t'] for chunked saves is relative to the start of THAT chunk
            # BUT we need to check how it was saved.
            # Looking at main.py: t starts at 0 for each frac loop
            t = event['t']
            
            if t == 0 or t >= len(degrees):
                continue
                
            deg_before = degrees[t-1]
            deg_after = degrees[t]
            
            diff = deg_after - deg_before
            changed_nodes = np.where(diff != 0)[0]
            
            if len(changed_nodes) > 0:
                success_count += 1
                # print(f"  [T={t:03d}] Shift at nodes {list(changed_nodes)}")
            else:
                print(f"  [T={t:03d}] ERROR: No degree shift detected!")

    print(f"\nFinal Result: {success_count}/{total_events} events verified in bus features.")

if __name__ == "__main__":
    DATA_PATH = "src/data/raw"
    test_degree_transitions(DATA_PATH, "case33")
