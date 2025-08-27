#!/usr/bin/env python3
"""
Data validation utilities for ensuring data files exist before training.
"""

import os
import subprocess
import sys
from typing import List, Tuple

def check_data_files_exist(config) -> Tuple[bool, List[str]]:
    """
    Check if all required data files exist for the specified bus systems.
    
    Args:
        config: Configuration object containing NUM_BUSES and renewable fractions
        
    Returns:
        Tuple of (all_files_exist: bool, missing_files: List[str])
    """
    bus_systems = config.NUM_BUSES if isinstance(config.NUM_BUSES, list) else [config.NUM_BUSES]
    renewable_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Standard fractions from gen_meas_best.py
    
    required_files = []
    missing_files = []
    
    data_dir = "./data"
    
    for num_buses in bus_systems:
        case_name = f"case{num_buses}"
        for frac in renewable_fractions:
            # Check for all required file types
            base_name = f"{case_name}_{{file_type}}_frac{frac:.1f}"
            
            required_file_types = [
                "features.npy",
                "targets.npy", 
                "adjacency.npy",
                "ybus_matrices.npy",
                "time_energy_coeffs.txt",
                "time_carbon_coeffs.txt"
            ]
            
            for file_type in required_file_types:
                filename = base_name.format(file_type=file_type.split('.')[0]) + '.' + file_type.split('.')[1]
                filepath = os.path.join(data_dir, filename)
                required_files.append(filepath)
                
                if not os.path.exists(filepath):
                    missing_files.append(filepath)
    
    all_exist = len(missing_files) == 0
    return all_exist, missing_files

def generate_data_if_missing(config) -> bool:
    """
    Generate data if any files are missing by running gen_meas_best.py
    
    Args:
        config: Configuration object
        
    Returns:
        bool: True if data generation was successful, False otherwise
    """
    data_exist, missing_files = check_data_files_exist(config)
    
    if data_exist:
        print("✅ All required data files found. Skipping data generation.")
        return True
    
    print(f"❌ Missing {len(missing_files)} data files. Examples:")
    for i, file in enumerate(missing_files[:5]):  # Show first 5 missing files
        print(f"   - {file}")
    if len(missing_files) > 5:
        print(f"   ... and {len(missing_files) - 5} more files")
    
    print("\n🔄 Running data generation script...")
    
    try:
        # Run gen_meas_best.py from the data directory
        data_gen_script = os.path.join("data", "gen_meas_best.py")
        
        if not os.path.exists(data_gen_script):
            print(f"❌ Data generation script not found: {data_gen_script}")
            return False
        
        # Run the script and capture output
        result = subprocess.run(
            [sys.executable, data_gen_script],
            capture_output=True,
            text=True,
            cwd="."  # Run from project root
        )
        
        if result.returncode == 0:
            print("✅ Data generation completed successfully!")
            
            # Verify data was actually generated
            data_exist_after, remaining_missing = check_data_files_exist(config)
            if data_exist_after:
                print("✅ All required data files now exist.")
                return True
            else:
                print(f"⚠️ Data generation completed but {len(remaining_missing)} files still missing.")
                return False
        else:
            print(f"❌ Data generation failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running data generation: {e}")
        return False

def validate_data_before_training(config) -> bool:
    """
    Main function to validate data exists and generate if needed.
    
    Args:
        config: Configuration object
        
    Returns:
        bool: True if data is ready for training, False otherwise
    """
    print("\n" + "="*60)
    print("🔍 VALIDATING DATA FILES")
    print("="*60)
    
    success = generate_data_if_missing(config)
    
    if success:
        print("\n✅ Data validation completed successfully. Ready for training!")
    else:
        print("\n❌ Data validation failed. Cannot proceed with training.")
    
    print("="*60 + "\n")
    return success
