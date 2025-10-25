#!/usr/bin/env python3
"""
Strict Data Processing Script
This script validates and processes data to ensure:
1. No NaN values
2. Physically reasonable values
3. Consistent data shapes (truncation, no padding)
4. Clean data for training
"""

import os
import numpy as np
import json
import glob
from pathlib import Path

class StrictDataProcessor:
    """Strict data processor for power system data"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Physical constraints
        self.vm_min, self.vm_max = 0.8, 1.2  # Voltage magnitude limits
        self.va_min, self.va_max = -np.pi, np.pi  # Voltage angle limits
        self.power_max = 10000  # Maximum power flow (MW/MVAR)
        
    def validate_timestep(self, features, targets, timestep):
        """Validate a single timestep for physical reasonableness"""
        issues = []
        
        # Check for NaN values
        if np.any(np.isnan(features)) or np.any(np.isnan(targets)):
            issues.append("NaN values detected")
            return False, issues
        
        # Check voltage magnitudes
        vm = features[:, 0]  # vm_pu
        if np.any(vm < self.vm_min) or np.any(vm > self.vm_max):
            issues.append(f"Voltage magnitudes out of range: {np.min(vm):.3f}-{np.max(vm):.3f}")
        
        # Check voltage angles
        va = features[:, 1]  # va_rad
        if np.any(va < self.va_min) or np.any(va > self.va_max):
            issues.append(f"Voltage angles out of range: {np.min(va):.3f}-{np.max(va):.3f}")
        
        # Check power flows (should be reasonable)
        p_load = features[:, 2]  # p_load
        q_load = features[:, 3]  # q_load
        if np.any(np.abs(p_load) > self.power_max) or np.any(np.abs(q_load) > self.power_max):
            issues.append(f"Power flows too large: {np.max(np.abs(p_load)):.1f} MW")
        
        # Check generation (should be reasonable)
        gen_data = features[:, 4:]  # All generation
        if np.any(np.abs(gen_data) > self.power_max):
            issues.append(f"Generation too large: {np.max(np.abs(gen_data)):.1f} MW")
        
        return len(issues) == 0, issues
    
    def process_scenario(self, case_name, renewable_fraction):
        """Process a single scenario"""
        print(f"\nPROCESSING {case_name} - {renewable_fraction*100:.0f}% renewable")
        
        # Find latest files
        pattern = f"{self.data_dir}/{case_name}_*_frac{renewable_fraction:.1f}_*.npy"
        files = glob.glob(pattern)
        
        if not files:
            print(f"ERROR: No files found for {case_name} {renewable_fraction*100:.0f}%")
            return None
        
        # Load data
        data = {}
        for file in files:
            if 'features' in file:
                data['features'] = np.load(file)
            elif 'targets' in file:
                data['targets'] = np.load(file)
            elif 'adjacency' in file:
                data['adjacency'] = np.load(file)
            elif 'ybus_base' in file:
                data['ybus_base'] = np.load(file)
        
        if 'features' not in data:
            print(f"ERROR: Features not found for {case_name}")
            return None
        
        print(f"  Original shape: {data['features'].shape}")
        
        # Validate each timestep
        valid_timesteps = []
        invalid_timesteps = []
        
        for t in range(len(data['features'])):
            is_valid, issues = self.validate_timestep(
                data['features'][t], 
                data['targets'][t] if 'targets' in data else data['features'][t], 
                t
            )
            
            if is_valid:
                valid_timesteps.append(t)
            else:
                invalid_timesteps.append((t, issues))
        
        print(f"  Valid timesteps: {len(valid_timesteps)}/{len(data['features'])}")
        print(f"  Invalid timesteps: {len(invalid_timesteps)}")
        
        if invalid_timesteps:
            print(f"  Invalid timestep details:")
            for t, issues in invalid_timesteps[:5]:  # Show first 5
                print(f"    Timestep {t}: {', '.join(issues)}")
            if len(invalid_timesteps) > 5:
                print(f"    ... and {len(invalid_timesteps) - 5} more")
        
        # Truncate data to valid timesteps only
        if len(valid_timesteps) == 0:
            print(f"ERROR: No valid timesteps found for {case_name}")
            return None
        
        processed_data = {}
        for key, value in data.items():
            if key == 'adjacency':
                # Adjacency is per-timestep
                processed_data[key] = value[valid_timesteps]
            else:
                processed_data[key] = value[valid_timesteps]
        
        print(f"  Processed shape: {processed_data['features'].shape}")
        
        # Save processed data
        timestamp = "processed"
        for key, value in processed_data.items():
            filename = f"{self.processed_dir}/{case_name}_{key}_frac{renewable_fraction:.1f}_{timestamp}.npy"
            np.save(filename, value)
            print(f"  Saved: {filename}")
        
        # Save processing report
        report = {
            'case_name': case_name,
            'renewable_fraction': renewable_fraction,
            'original_timesteps': len(data['features']),
            'valid_timesteps': len(valid_timesteps),
            'invalid_timesteps': len(invalid_timesteps),
            'success_rate': len(valid_timesteps) / len(data['features']) * 100,
            'valid_timestep_indices': valid_timesteps,
            'invalid_timestep_details': [(t, issues) for t, issues in invalid_timesteps]
        }
        
        report_file = f"{self.processed_dir}/{case_name}_processing_report_frac{renewable_fraction:.1f}_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  Processing report: {report_file}")
        
        return processed_data
    
    def process_all_scenarios(self):
        """Process all scenarios"""
        print("STARTING STRICT DATA PROCESSING")
        print("=" * 60)
        
        bus_systems = [33, 57, 118]
        renewable_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        results = {}
        
        for num_buses in bus_systems:
            case_name = f"case{num_buses}"
            results[case_name] = {}
            
            for frac in renewable_fractions:
                processed_data = self.process_scenario(case_name, frac)
                if processed_data is not None:
                    results[case_name][frac] = {
                        'success': True,
                        'valid_timesteps': len(processed_data['features']),
                        'shape': processed_data['features'].shape
                    }
                else:
                    results[case_name][frac] = {
                        'success': False,
                        'valid_timesteps': 0,
                        'shape': None
                    }
        
        # Summary report
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        
        for case_name, case_results in results.items():
            print(f"\n{case_name.upper()}:")
            for frac, result in case_results.items():
                status = "OK" if result['success'] else "FAILED"
                if result['success']:
                    print(f"  {frac*100:3.0f}%: {status} {result['valid_timesteps']} timesteps {result['shape']}")
                else:
                    print(f"  {frac*100:3.0f}%: {status} FAILED")
        
        return results

if __name__ == "__main__":
    processor = StrictDataProcessor()
    results = processor.process_all_scenarios()
