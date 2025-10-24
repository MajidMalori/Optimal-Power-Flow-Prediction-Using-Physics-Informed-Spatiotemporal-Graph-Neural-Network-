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
            # Support both sparse (new) and dense (old) Ybus formats
            # Try sparse format first, then fall back to dense format
            required_file_types = [
                "features.npy",
                "targets.npy", 
                "adjacency.npy",
                "time_energy_coeffs.txt",
                "time_carbon_coeffs.txt",
                "ext_grid_generation.npy",
                "conventional_generation.npy",
                "renewable_generation.npy"
            ]
            
            # Add Ybus files - check for sparse format
            sparse_ybus_types = [
                "ybus_base.npy",
                "ybus_contingency_timesteps.npy",
                "ybus_contingency_matrices.npy",
                "convergence_report.json"
            ]
            
            # Try sparse format first
            dense_ybus_type = "ybus_matrices.npy"
            
            # Check common files (features, targets, etc.)
            for file_type in required_file_types:
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
            
            # Check Ybus files - REQUIRE sparse format (new implementation)
            # Old dense format is only for backward compatibility during loading, not validation
            sparse_ybus_found = True
            for ybus_type in sparse_ybus_types:
                found = False
                
                if latest_timestamp:
                    base_name = f"{case_name}_{ybus_type.split('.')[0]}_frac{frac:.1f}_{latest_timestamp}"
                    filename = base_name + '.' + ybus_type.split('.')[1]
                    filepath = os.path.join(data_dir, filename)
                    if os.path.exists(filepath):
                        found = True
                
                if not found:
                    base_name = f"{case_name}_{ybus_type.split('.')[0]}_frac{frac:.1f}"
                    filename = base_name + '.' + ybus_type.split('.')[1]
                    filepath = os.path.join(data_dir, filename)
                    if os.path.exists(filepath):
                        found = True
                
                if not found:
                    # Sparse file missing - add to missing list
                    missing_files.append(f"{case_name}_{ybus_type.split('.')[0]}_frac{frac:.1f}.{ybus_type.split('.')[1]}")
                    sparse_ybus_found = False
    
    all_exist = len(missing_files) == 0
    return all_exist, missing_files

def check_data_consistency(config) -> Tuple[bool, str]:
    """
    Check if existing data files have consistent timestamps and correct timesteps.
    Validates both file consistency and configuration match.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (is_consistent: bool, reason: str)
    """
    import re
    import glob
    import numpy as np
    from data.gen_meas_best import CONFIG as GEN_CONFIG
    
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
    if len(filename_timestamps) > 1:
        timestamps_list = sorted(list(filename_timestamps))
        return False, f"Found mixed timestamps in data files: {', '.join(timestamps_list)}"
    
    timestamp = list(filename_timestamps)[0]
    
    # Check if data was generated with correct number of timesteps
    bus_systems = config.NUM_BUSES if isinstance(config.NUM_BUSES, list) else [config.NUM_BUSES]
    expected_timesteps = GEN_CONFIG['time_steps']
    
    # Check first available features file to validate timesteps
    for num_buses in bus_systems:
        case_name = f"case{num_buses}"
        pattern = os.path.join(data_dir, f"{case_name}_features_frac*.npy")
        feature_files = glob.glob(pattern)
        
        if feature_files:
            try:
                # Load first file to check shape
                features = np.load(feature_files[0])
                actual_timesteps = features.shape[0]
                
                # Check timestep consistency (exact match required)
                if actual_timesteps != expected_timesteps:
                    return False, f"Data generated with {actual_timesteps} timesteps, but config requires {expected_timesteps}. Regeneration needed."
                
                break  # Only need to check one file
            except Exception as e:
                return False, f"Error reading data file for timestep validation: {e}"
    
    return True, f"All data files consistent (timestamp: {timestamp}, timesteps: {expected_timesteps})"

def monitor_data_generation_progress_per_system(config, stop_event):
    """
    Monitor data generation progress by checking file creation.
    Shows one unified progress bar for all bus systems.
    """
    import glob
    import re
    bus_systems = config.NUM_BUSES if isinstance(config.NUM_BUSES, list) else [config.NUM_BUSES]
    renewable_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    data_dir = "./data"
    
    # Expected files per scenario: 9 files (5 base + 3 ybus + 1 convergence report)
    files_per_scenario = 9
    scenarios_per_bus = len(renewable_fractions)
    files_per_bus = scenarios_per_bus * files_per_scenario  # 54 files per bus system
    total_expected_files = files_per_bus * len(bus_systems)  # 162 total files
    
    # Find the current timestamp being generated
    def get_current_timestamp():
        """Get the most recent timestamp from existing files"""
        timestamp_pattern = r"_(\d{8}_\d{6})"
        timestamps = set()
        for file in os.listdir(data_dir):
            match = re.search(timestamp_pattern, file)
            if match:
                timestamps.add(match.group(1))
        return max(timestamps) if timestamps else None
    
    current_timestamp = None
    last_count = 0
    
    # Create progress bar immediately at 0%
    pbar = tqdm(
        initial=0,
        total=total_expected_files,
        desc="📊 Generating data",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} files",
        leave=True
    )
    
    wait_start_time = time.time()
    
    try:
        while not stop_event.is_set():
            # Get current timestamp on first iteration
            if current_timestamp is None:
                current_timestamp = get_current_timestamp()
                if current_timestamp is None:
                    # Still waiting for first file to determine timestamp
                    time.sleep(0.5)
                    continue  # Wait for first file to be created
            
            # Count all files from current timestamp
            pattern = os.path.join(data_dir, f"case*_*_{current_timestamp}.*")
            existing_files = glob.glob(pattern)
            current_count = len(existing_files)
            
            # Update progress bar if new files were created
            if current_count > last_count:
                delta = current_count - last_count
                pbar.update(delta)
                last_count = current_count
            
            # Check if we're done
            if current_count >= total_expected_files:
                pbar.update(total_expected_files - pbar.n)  # Complete the bar
                pbar.close()
                return
            
            time.sleep(0.2)  # Check every 0.2 seconds (faster for quick generations)
    finally:
        # Ensure progress bar is closed
        if pbar is not None:
            pbar.close()

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
        "case*_*_frac*.json",     # Convergence reports
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
        print(f"✅ Removed {files_removed} existing data files.")
        print("   Cleanup complete. Ready for fresh data generation.\n")
    else:
        print("✅ No existing data files to clean.\n")

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
        # Even if all files exist, check for data consistency (mixed timestamps + timesteps)
        is_consistent, reason = check_data_consistency(config)
        
        if is_consistent:
            print(f"✅ Data validation passed: {reason}")
            print("   Skipping data generation.")
            return True
        else:
            print(f"⚠️  Data inconsistency detected: {reason}")
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
    
    # Wait for file system to sync and confirm deletion (especially important on slower systems)
    import glob
    data_dir = "./data"
    max_wait = 5  # Maximum 5 seconds
    wait_interval = 0.2
    elapsed = 0
    
    while elapsed < max_wait:
        # Check if any data files still exist
        remaining = []
        for pattern in ["case*_*_frac*.npy", "case*_*_frac*.txt", "case*_*_frac*.json"]:
            remaining.extend(glob.glob(os.path.join(data_dir, pattern)))
        
        if not remaining:
            break  # All files deleted
        
        time.sleep(wait_interval)
        elapsed += wait_interval
    
    print("🔄 Running data generation script...")
    
    try:
        # Run gen_meas_best.py from the data directory
        data_gen_script = os.path.join("data", "gen_meas_best.py")
        
        if not os.path.exists(data_gen_script):
            print(f"❌ Data generation script not found: {data_gen_script}")
            return False
        
        # Run the script with per-system progress bars
        print("📊 Starting data generation...\n")
        
        # Start monitoring progress in background thread
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_data_generation_progress_per_system, 
            args=(config, stop_event)
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
        monitor_thread.join(timeout=3)  # Wait for thread to finish
        
        if result.returncode == 0:
            print("\n✅ Data generation completed successfully!")
            
            # Verify data was actually generated
            data_exist_after, remaining_missing = check_data_files_exist(config)
            if data_exist_after:
                print("✅ All required data files now exist.\n")
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

def display_convergence_analysis(config):
    """
    Display detailed convergence analysis from all generated convergence reports.
    Shows power flow success rates and any failures (especially with contingencies).
    """
    import json
    import glob
    
    data_dir = "./data"
    bus_systems = config.NUM_BUSES if isinstance(config.NUM_BUSES, list) else [config.NUM_BUSES]
    renewable_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    print("\n" + "="*80)
    print("📊 CONVERGENCE ANALYSIS")
    print("="*80)
    
    total_scenarios = 0
    total_successful = 0
    total_failed = 0
    scenarios_with_failures = []
    
    # Collect all data first
    all_data = {}  # {num_buses: {frac: stats}}
    for num_buses in bus_systems:
        case_name = f"case{num_buses}"
        all_data[num_buses] = {}
        
        for frac in renewable_fractions:
            pattern = f"{case_name}_convergence_report_frac{frac:.1f}_*.json"
            report_files = glob.glob(os.path.join(data_dir, pattern))
            
            if report_files:
                report_file = sorted(report_files)[-1]
                with open(report_file, 'r') as f:
                    stats = json.load(f)
                all_data[num_buses][frac] = stats
                
                total_scenarios += 1
                total_successful += stats['successful']
                total_failed += stats['failed']
                
                if len(stats['failed_with_contingency']) > 0:
                    scenarios_with_failures.append({'case': case_name, 'fraction': frac, 'stats': stats})
    
    # Print horizontal table header with fixed column width
    col_width = 17  # Fixed width for each case column
    header = "Renewable%  "
    for num_buses in bus_systems:
        header += f"| CASE{num_buses:<{col_width-6}} "
    print(f"\n{header}")
    print("-" * len(header))
    
    # Print each renewable fraction row
    for frac in renewable_fractions:
        row = f"  {frac*100:>5.1f}%    "
        for num_buses in bus_systems:
            if frac in all_data[num_buses]:
                stats = all_data[num_buses][frac]
                has_cont_fail = len(stats['failed_with_contingency']) > 0
                icon = "⚠️" if has_cont_fail else "✅"
                rate = stats['success_rate']
                
                # Compact display with fixed width
                if stats['failed'] > 0:
                    nf = len(stats['failed_no_contingency'])
                    cf = len(stats['failed_with_contingency'])
                    if cf > 0:
                        cell = f"{icon}{rate:5.1f}% {nf}💡{cf}🔴"
                    else:
                        cell = f"{icon}{rate:5.1f}% {nf}💡"
                else:
                    cell = f"{icon}{rate:5.1f}%"
                
                # Pad to fixed width (accounting for emoji width issues, just right-pad with spaces)
                row += f"| {cell:<{col_width-3}} "
            else:
                row += f"| {'---':<{col_width-3}} "
        print(row)
    
    print("\n📈 Summary:")
    
    # Count contingency-specific failures across all scenarios
    total_contingency_failures = 0
    total_normal_failures = 0
    total_contingency_timesteps = 0
    
    # Re-scan all convergence reports for detailed counts
    for num_buses in bus_systems:
        case_name = f"case{num_buses}"
        for frac in renewable_fractions:
            pattern = f"{case_name}_convergence_report_frac{frac:.1f}_*.json"
            report_files = glob.glob(os.path.join(data_dir, pattern))
            if report_files:
                with open(sorted(report_files)[-1], 'r') as f:
                    stats = json.load(f)
                total_contingency_failures += len(stats['failed_with_contingency'])
                total_normal_failures += len(stats['failed_no_contingency'])
                total_contingency_timesteps += len(stats['failed_with_contingency']) + len([t for t in range(stats['total_timesteps']) if t not in stats['failed_no_contingency'] and t not in stats['failed_with_contingency']])
    
    total_timesteps = total_successful + total_failed
    overall_rate = (total_successful / total_timesteps * 100) if total_timesteps > 0 else 0
    contingency_success_rate = ((total_contingency_timesteps - total_contingency_failures) / total_contingency_timesteps * 100) if total_contingency_timesteps > 0 else 100
    
    print(f"  • Overall: {overall_rate:.1f}% ({total_successful}/{total_timesteps}) | Normal fail: {total_normal_failures}💡 Contingency fail: {total_contingency_failures}🔴")
    print(f"  • 🎯 Critical: {contingency_success_rate:.1f}% contingency success")
    if scenarios_with_failures:
        print(f"  • ⚠️  {len(scenarios_with_failures)} scenario(s) with contingency failures")
    print("="*80)

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
        
        # Run the script with per-system progress bars
        print("📊 Starting fresh data generation...\n")
        
        # Start monitoring progress in background thread
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_data_generation_progress_per_system, 
            args=(config, stop_event)
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
    Always displays convergence analysis after validation.
    
    Args:
        config: Configuration object
        
    Returns:
        bool: True if data is ready for training, False otherwise
    """
    print("\n" + "="*80)
    print("🔍 DATA VALIDATION")
    print("="*80)
    
    success = generate_data_if_missing(config)
    
    if success:
        # Always show convergence analysis after successful data validation
        display_convergence_analysis(config)
        print("\n✅ Ready for training!")
    else:
        print("\n❌ Data validation failed!")
    
    print("="*80)
    return success
