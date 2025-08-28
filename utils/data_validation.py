#!/usr/bin/env python3
"""
Data validation utilities for ensuring data files exist before training.
"""

import os
import subprocess
import sys
import threading
import time
from typing import List, Tuple
from tqdm import tqdm

def find_latest_timestamp(data_dir: str) -> str:
    """
    Find the latest timestamp from existing data files.
    
    Returns:
        Latest timestamp string found in filenames, or None if no timestamped files exist
    """
    import glob
    import re
    
    # Look for any timestamped files with pattern YYYYMMDD_HHMMSS
    timestamp_pattern = r"_(\d{8}_\d{6})"
    timestamps = set()
    
    for file in os.listdir(data_dir):
        if file.endswith(('.npy', '.txt')):
            match = re.search(timestamp_pattern, file)
            if match:
                timestamps.add(match.group(1))
    
    return max(timestamps) if timestamps else None

def check_data_files_exist(config) -> Tuple[bool, List[str]]:
    """
    Check if all required data files exist for the specified bus systems.
    Now supports both legacy (no timestamp) and new timestamped file formats.
    
    Args:
        config: Configuration object containing NUM_BUSES and renewable fractions
        
    Returns:
        Tuple of (all_files_exist: bool, missing_files: List[str])
    """
    bus_systems = config.NUM_BUSES if isinstance(config.NUM_BUSES, list) else [config.NUM_BUSES]
    renewable_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Standard fractions from gen_meas_best.py
    
    missing_files = []
    data_dir = "./data"
    
    # First, try to find the latest timestamp from existing files
    latest_timestamp = find_latest_timestamp(data_dir)
    
    for num_buses in bus_systems:
        case_name = f"case{num_buses}"
        for frac in renewable_fractions:
            required_file_types = [
                "features.npy",
                "targets.npy", 
                "adjacency.npy",
                "ybus_matrices.npy",
                "time_energy_coeffs.txt",
                "time_carbon_coeffs.txt"
            ]
            
            for file_type in required_file_types:
                # Try timestamped format first, then legacy format
                found = False
                
                if latest_timestamp:
                    # Check for timestamped file
                    base_name = f"{case_name}_{file_type.split('.')[0]}_frac{frac:.1f}_{latest_timestamp}"
                    filename = base_name + '.' + file_type.split('.')[1]
                    filepath = os.path.join(data_dir, filename)
                    if os.path.exists(filepath):
                        found = True
                
                if not found:
                    # Check for legacy format (no timestamp)
                    base_name = f"{case_name}_{file_type.split('.')[0]}_frac{frac:.1f}"
                    filename = base_name + '.' + file_type.split('.')[1]
                    filepath = os.path.join(data_dir, filename)
                    if os.path.exists(filepath):
                        found = True
                
                if not found:
                    missing_files.append(f"{case_name}_{file_type.split('.')[0]}_frac{frac:.1f}.{file_type.split('.')[1]}")
    
    all_exist = len(missing_files) == 0
    return all_exist, missing_files

def check_data_consistency(config) -> Tuple[bool, str]:
    """
    Check if existing data files have consistent timestamps to detect mixed data.
    Now uses the filename timestamps for perfect detection.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (is_consistent: bool, reason: str)
    """
    import re
    
    data_dir = "./data"
    filename_timestamps = set()
    
    # Look for timestamp patterns in filenames
    timestamp_pattern = r"_(\d{8}_\d{6})"
    
    for file in os.listdir(data_dir):
        if file.endswith(('.npy', '.txt')):
            match = re.search(timestamp_pattern, file)
            if match:
                filename_timestamps.add(match.group(1))
    
    if len(filename_timestamps) == 0:
        # No timestamped files found - check if legacy files exist
        bus_systems = config.NUM_BUSES if isinstance(config.NUM_BUSES, list) else [config.NUM_BUSES]
        renewable_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        legacy_files_exist = False
        for num_buses in bus_systems:
            case_name = f"case{num_buses}"
            for frac in renewable_fractions:
                legacy_filename = f"{case_name}_features_frac{frac:.1f}.npy"
                if os.path.exists(os.path.join(data_dir, legacy_filename)):
                    legacy_files_exist = True
                    break
            if legacy_files_exist:
                break
        
        if legacy_files_exist:
            return False, "Found legacy data files without timestamps - mixed data possible"
        else:
            return True, "No data files found"
    
    # Check if all files have the same timestamp
    if len(filename_timestamps) == 1:
        timestamp = list(filename_timestamps)[0]
        return True, f"All data files have consistent timestamp: {timestamp}"
    else:
        timestamps_list = sorted(list(filename_timestamps))
        return False, f"Found mixed timestamps in data files: {', '.join(timestamps_list)}"

def monitor_data_generation_progress(config, pbar, stop_event):
    """
    Monitor data generation progress by checking file creation.
    Updates progress bar based on how many files have been created.
    """
    bus_systems = config.NUM_BUSES if isinstance(config.NUM_BUSES, list) else [config.NUM_BUSES]
    renewable_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Calculate total expected files
    total_files = 0
    for num_buses in bus_systems:
        total_files += len(renewable_fractions) * 6  # 6 file types per fraction
    
    pbar.total = total_files
    last_count = 0
    
    while not stop_event.is_set():
        # Check how many files exist now
        data_exist, missing_files = check_data_files_exist(config)
        current_count = total_files - len(missing_files)
        
        # Update progress bar if new files were created
        if current_count > last_count:
            pbar.update(current_count - last_count)
            last_count = current_count
            
            # Update description based on which bus system we're likely processing
            if current_count <= len(renewable_fractions) * 6:
                pbar.set_description("Generating case33 data")
            elif current_count <= len(renewable_fractions) * 6 * 2:
                pbar.set_description("Generating case57 data")
            else:
                pbar.set_description("Generating case118 data")
        
        # Check if we're done
        if data_exist:
            pbar.update(total_files - pbar.n)  # Complete the bar
            pbar.set_description("Data generation complete")
            break
            
        time.sleep(0.5)  # Check every 0.5 seconds

def clean_existing_data(config):
    """
    Remove all existing data files to ensure data integrity.
    This prevents mixing data from different generation runs.
    Now handles both timestamped and legacy file formats.
    """
    import glob
    
    data_dir = "./data"
    files_removed = 0
    
    print("🧹 Cleaning existing data files to ensure data integrity...")
    
    # Remove all data files (both timestamped and legacy formats)
    patterns = [
        "case*_*_frac*.npy",      # Legacy and timestamped .npy files
        "case*_*_frac*.txt",      # Legacy and timestamped .txt files
    ]
    
    for pattern in patterns:
        file_pattern = os.path.join(data_dir, pattern)
        files_to_remove = glob.glob(file_pattern)
        
        for filepath in files_to_remove:
            # Skip the generation script and check script
            filename = os.path.basename(filepath)
            if filename in ["gen_meas_best.py", "check_data.py"]:
                continue
                
            try:
                os.remove(filepath)
                files_removed += 1
            except OSError as e:
                print(f"⚠️ Warning: Could not remove {filepath}: {e}")
    
    if files_removed > 0:
        print(f"✅ Removed {files_removed} existing data files to ensure clean generation.")
    else:
        print("✅ No existing data files to clean.")

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
        # Even if all files exist, check for data consistency (mixed timestamps)
        is_consistent, reason = check_data_consistency(config)
        
        if is_consistent:
            print("✅ All required data files found with consistent timestamps. Skipping data generation.")
            return True
        else:
            print(f"⚠️ All data files exist but detected inconsistency: {reason}")
            print("🔄 Will regenerate all data to ensure consistency.")
            # Continue to cleaning and regeneration
    else:
        print(f"❌ Missing {len(missing_files)} data files. Examples:")
        for i, file in enumerate(missing_files[:5]):  # Show first 5 missing files
            print(f"   - {file}")
        if len(missing_files) > 5:
            print(f"   ... and {len(missing_files) - 5} more files")
    
    # CRITICAL: Clean all existing data to prevent mixing different time periods
    clean_existing_data(config)
    
    print("\n🔄 Running data generation script...")
    
    try:
        # Run gen_meas_best.py from the data directory
        data_gen_script = os.path.join("data", "gen_meas_best.py")
        
        if not os.path.exists(data_gen_script):
            print(f"❌ Data generation script not found: {data_gen_script}")
            return False
        
        # Run the script with unified progress bar
        print("📊 Starting data generation...")
        
        # Create progress bar
        with tqdm(total=100, desc="Initializing data generation", 
                 bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} files") as pbar:
            
            # Start monitoring progress in background thread
            stop_event = threading.Event()
            monitor_thread = threading.Thread(
                target=monitor_data_generation_progress, 
                args=(config, pbar, stop_event)
            )
            monitor_thread.start()
            
            # Run data generation script (capture output to avoid interference)
            result = subprocess.run(
                [sys.executable, data_gen_script],
                cwd=".",
                capture_output=True,  # Capture to avoid interfering with progress bar
                text=True
            )
            
            # Stop monitoring thread
            stop_event.set()
            monitor_thread.join(timeout=2)  # Wait max 2 seconds for thread to finish
        
        if result.returncode == 0:
            print("✅ Data generation completed successfully!")
            
            # Verify data was actually generated
            data_exist_after, remaining_missing = check_data_files_exist(config)
            if data_exist_after:
                print("✅ All required data files now exist.")
                return True
            else:
                print(f"⚠️ Data generation completed but {len(remaining_missing)} files still missing.")
                for missing in remaining_missing[:3]:  # Show a few examples
                    print(f"   - Still missing: {missing}")
                return False
        else:
            print(f"❌ Data generation failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error details: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running data generation: {e}")
        return False

def force_clean_all_data(config) -> bool:
    """
    Force clean all data files and regenerate from scratch.
    Useful when you want to ensure completely fresh data.
    
    Args:
        config: Configuration object
        
    Returns:
        bool: True if data generation was successful, False otherwise
    """
    print("\n" + "="*60)
    print("🔄 FORCE REGENERATING ALL DATA")
    print("="*60)
    
    # Clean all existing data
    clean_existing_data(config)
    
    print("\n🔄 Running data generation script...")
    
    try:
        # Run gen_meas_best.py from the data directory
        data_gen_script = os.path.join("data", "gen_meas_best.py")
        
        if not os.path.exists(data_gen_script):
            print(f"❌ Data generation script not found: {data_gen_script}")
            return False
        
        # Run the script with unified progress bar
        print("📊 Starting fresh data generation...")
        
        # Create progress bar
        with tqdm(total=100, desc="Initializing data generation", 
                 bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} files") as pbar:
            
            # Start monitoring progress in background thread
            stop_event = threading.Event()
            monitor_thread = threading.Thread(
                target=monitor_data_generation_progress, 
                args=(config, pbar, stop_event)
            )
            monitor_thread.start()
            
            # Run data generation script
            result = subprocess.run(
                [sys.executable, data_gen_script],
                cwd=".",
                capture_output=True,
                text=True
            )
            
            # Stop monitoring thread
            stop_event.set()
            monitor_thread.join(timeout=2)
        
        if result.returncode == 0:
            print("✅ Fresh data generation completed successfully!")
            return True
        else:
            print(f"❌ Data generation failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error details: {result.stderr}")
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
