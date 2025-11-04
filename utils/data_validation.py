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
    
    if not os.path.exists(data_dir):
        return None
    
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
    data_dir = config.DATA_DIR  # Use mode-specific directory from config
    
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
                "time_carbon_coeffs.txt"
                # Note: Generation components are now included in features/targets matrices
            ]
            
            # Add Ybus files - check for sparse format
            sparse_ybus_types = [
                "ybus_base.npy",
                "ybus_contingency_timesteps.npy",
                "ybus_contingency_matrices.npy",
                "convergence_report.json"
            ]
            
            # OPF: Check for bus_types file (new structure)
            opf_files = [
                "bus_types.npy"  # OPF: bus type codes [0=PQ, 1=PV, 2=Slack]
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
            
            # Check OPF files (bus_types) - required for new OPF structure
            for file_type in opf_files:
                found = False
                
                if latest_timestamp:
                    base_name = f"{case_name}_{file_type.split('.')[0]}_frac{frac:.1f}_{latest_timestamp}"
                    filename = base_name + '.' + file_type.split('.')[1]
                    filepath = os.path.join(data_dir, filename)
                    if os.path.exists(filepath):
                        found = True
                
                if not found:
                    base_name = f"{case_name}_{file_type.split('.')[0]}_frac{frac:.1f}"
                    filename = base_name + '.' + file_type.split('.')[1]
                    filepath = os.path.join(data_dir, filename)
                    if os.path.exists(filepath):
                        found = True
                
                if not found:
                    missing_files.append(f"{case_name}_{file_type.split('.')[0]}_frac{frac:.1f}.{file_type.split('.')[1]} (OPF structure)")
            
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
    Validates both file consistency and configuration match (time-series mode only).
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (is_consistent: bool, reason: str)
    """
    import re
    import glob
    import numpy as np
    import json
    
    data_dir = config.DATA_DIR  # Use mode-specific directory
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        return True, "Data directory does not exist yet"
    
    # Check for metadata file (new system)
    metadata_file = os.path.join(data_dir, "data_generation_metadata.json")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check generation mode (should always be time-series)
            stored_mode = metadata.get('generation_mode', 'time_series')
            
            if stored_mode != 'time_series':
                return False, f"Data generation mode mismatch: existing={stored_mode}, expected=time_series. Regeneration needed."
            
            # Check data mode (train vs test)
            stored_data_mode = metadata.get('data_mode', 'unknown')
            if stored_data_mode != config.DATA_MODE:
                return False, f"Data mode mismatch: existing={stored_data_mode}, config={config.DATA_MODE}. Regeneration needed."
            
            # Check timesteps
            expected_timesteps = config.DATA_MODE_TIMESTEPS[config.DATA_MODE]
            stored_timesteps = metadata.get('timesteps', 0)
            if stored_timesteps != expected_timesteps:
                return False, f"Data generated with {stored_timesteps} timesteps, but config requires {expected_timesteps}. Regeneration needed."
            
            # Check timestamp
            timestamp = metadata.get('timestamp', 'unknown')
            
            return True, f"Data consistent ({stored_mode} mode, {stored_timesteps} timesteps, {stored_data_mode})"
            
        except Exception as e:
            return False, f"Error reading metadata file: {e}. Regeneration recommended."
    
    # Fallback to old timestamp-based validation if no metadata file
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
            return False, "Found legacy data files without metadata. Regeneration recommended."
        else:
            return True, "No data files found"
    
    # Check if all files have the same timestamp
    if len(filename_timestamps) > 1:
        timestamps_list = sorted(list(filename_timestamps))
        return False, f"Found mixed timestamps in data files: {', '.join(timestamps_list)}"
    
    timestamp = list(filename_timestamps)[0]
    
    # Check if data was generated with correct number of timesteps
    bus_systems = config.NUM_BUSES if isinstance(config.NUM_BUSES, list) else [config.NUM_BUSES]
    expected_timesteps = config.DATA_MODE_TIMESTEPS[config.DATA_MODE]  # Use mode-specific timesteps
    
    # Check first available features file to validate timesteps
    for num_buses in bus_systems:
        case_name = f"case{num_buses}"
        pattern = os.path.join(data_dir, f"{case_name}_features_frac*.npy")
        feature_files = glob.glob(pattern)
        
        if feature_files:
            try:
                # Load first file to check shape and structure
                features = np.load(feature_files[0])
                actual_timesteps = features.shape[0]
                
                # Check timestep consistency (exact match required)
                if actual_timesteps != expected_timesteps:
                    return False, f"Data generated with {actual_timesteps} timesteps, but config requires {expected_timesteps}. Regeneration needed."
                
                # OPF Structure Validation: Check target shape (should be 2 features, not 10)
                targets_pattern = feature_files[0].replace('features', 'targets')
                if os.path.exists(targets_pattern):
                    targets = np.load(targets_pattern)
                    if targets.shape[-1] != 2:
                        return False, f"OPF structure mismatch: targets have {targets.shape[-1]} features (expected 2). Old state estimation data detected. Regeneration needed."
                    
                    # Check if bus_types file exists (required for OPF)
                    bus_types_pattern = feature_files[0].replace('features', 'bus_types')
                    if not os.path.exists(bus_types_pattern):
                        return False, f"OPF structure incomplete: bus_types file missing. Old state estimation data detected. Regeneration needed."
                    
                    # Validate bus_types shape matches targets
                    bus_types = np.load(bus_types_pattern)
                    if bus_types.shape != targets.shape[:2]:  # Should be [timesteps, buses]
                        return False, f"OPF structure mismatch: bus_types shape {bus_types.shape} doesn't match targets shape {targets.shape[:2]}. Regeneration needed."
                
                break  # Only need to check one file
            except Exception as e:
                return False, f"Error reading data file for structure validation: {e}"
    
    return False, f"Data files exist but no metadata found (timestamp: {timestamp}). Regeneration recommended to add metadata."

def monitor_data_generation_progress_per_system(config, stop_event):
    """
    Monitor data generation progress by checking file creation.
    Shows one unified tqdm progress bar for all bus systems.
    
    Waits for filesystem synchronization before starting to ensure accurate tracking.
    """
    import glob
    import re
    bus_systems = config.NUM_BUSES if isinstance(config.NUM_BUSES, list) else [config.NUM_BUSES]
    renewable_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    data_dir = config.DATA_DIR  # Use mode-specific directory
    
    # Expected files per scenario: 9 files (5 base + 3 ybus + 1 convergence report)
    files_per_scenario = 9
    scenarios_per_bus = len(renewable_fractions)
    files_per_bus = scenarios_per_bus * files_per_scenario  # 54 files per bus system
    total_expected_files = files_per_bus * len(bus_systems)  # 162 total files
    
    # Find the current timestamp being generated
    def get_current_timestamp():
        """Get the most recent timestamp from existing files"""
        if not os.path.exists(data_dir):
            return None
        timestamp_pattern = r"_(\d{8}_\d{6})"
        timestamps = set()
        try:
            for file in os.listdir(data_dir):
                if file.endswith(('.npy', '.txt', '.json')):
                    match = re.search(timestamp_pattern, file)
                    if match:
                        timestamps.add(match.group(1))
            return max(timestamps) if timestamps else None
        except (OSError, PermissionError):
            return None
    
    current_timestamp = None
    last_count = 0
    max_wait_for_first_file = 30  # Maximum 30 seconds to wait for first file
    wait_elapsed = 0
    
    # Create tqdm progress bar
    pbar = tqdm(
        initial=0,
        total=total_expected_files,
        desc="Generating data",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} files",
        leave=True,
        unit="file",
        unit_scale=False
    )
    
    try:
        while not stop_event.is_set():
            # Wait for first file to determine timestamp
            if current_timestamp is None:
                current_timestamp = get_current_timestamp()
                if current_timestamp is None:
                    # Still waiting for first file
                    wait_elapsed += 0.5
                    if wait_elapsed > max_wait_for_first_file:
                        pbar.set_description("Waiting for data generation to start...")
                        time.sleep(0.5)
                        continue
                    time.sleep(0.5)
                    continue
                else:
                    # Found timestamp - start tracking
                    pbar.set_description(f"Generating data (timestamp: {current_timestamp[:8]})")
            
            # Count all files from current timestamp
            try:
                pattern = os.path.join(data_dir, f"case*_*_{current_timestamp}.*")
                existing_files = glob.glob(pattern)
                current_count = len(existing_files)
            except (OSError, PermissionError):
                # Filesystem still syncing - wait a bit
                time.sleep(0.2)
                continue
            
            # Update progress bar if new files were created
            if current_count > last_count:
                delta = current_count - last_count
                pbar.update(delta)
                last_count = current_count
                pbar.refresh()  # Force refresh display
            
            # Check if we're done
            if current_count >= total_expected_files:
                pbar.update(total_expected_files - pbar.n)  # Complete the bar
                pbar.set_description("✓ Data generation complete")
                pbar.close()
                return
            
            time.sleep(0.2)  # Check every 0.2 seconds
            
    except KeyboardInterrupt:
        pbar.set_description("⚠ Interrupted")
        pbar.close()
        raise
    finally:
        # Ensure progress bar is closed
        if pbar is not None and not pbar.disable:
            pbar.close()

def clean_existing_data(config, aggressive=True):
    """
    Remove ALL existing data files to ensure complete data integrity.
    This is a ROBUST cleanup that prevents any data mixing.
    
    Args:
        config: Configuration object
        aggressive: If True, removes ALL files including metadata (recommended)
    """
    import glob
    import shutil
    
    data_dir = config.DATA_DIR  # Use mode-specific directory (train or test)
    files_removed = 0
    
    print(f"\n{'='*80}")
    print(f"CLEANING ALL {config.DATA_MODE.upper()} DATA")
    print(f"{'='*80}")
    print(f"Target directory: {data_dir}")
    
    if not os.path.exists(data_dir):
        print("Directory doesn't exist yet. Nothing to clean.")
        return
    
    if aggressive:
        # AGGRESSIVE CLEANUP: Remove the entire mode-specific directory and recreate it
        # This ensures absolutely NO leftover files from previous runs
        try:
            print(f"Performing aggressive cleanup: removing entire directory...")
            shutil.rmtree(data_dir)
            os.makedirs(data_dir, exist_ok=True)
            print(f"✓ Successfully cleaned and recreated: {data_dir}")
            print("   All previous data has been completely removed.\n")
            return
        except Exception as e:
            print(f"WARNING: Could not remove directory: {e}")
            print("Falling back to file-by-file cleanup...\n")
    
    # FALLBACK: File-by-file cleanup (if aggressive fails)
    # Remove ALL data files (both timestamped and legacy formats)
    patterns = [
        "case*_*_frac*.npy",          # All .npy data files
        "case*_*_frac*.txt",          # All .txt coefficient files
        "case*_*_frac*.json",         # All convergence reports
        "data_generation_metadata.json",  # Metadata file
        "*.npy",                      # Any other numpy files
        "*.json",                     # Any other json files
        "*.txt",                      # Any other text files
    ]
    
    for pattern in patterns:
        file_pattern = os.path.join(data_dir, pattern)
        files_to_remove = glob.glob(file_pattern)
        
        for filepath in files_to_remove:
            # Skip the generation script and check script (should not be in data dir anyway)
            filename = os.path.basename(filepath)
            if filename in ["gen_meas_best.py", "check_data.py"]:
                continue
                
            try:
                os.remove(filepath)
                files_removed += 1
            except OSError as e:
                print(f"WARNING: Could not remove {filepath}: {e}")
    
    if files_removed > 0:
        print(f"✓ Removed {files_removed} data files from {config.DATA_MODE} folder.")
        print("   Ready for fresh data generation.\n")
    else:
        print("No existing data files found to clean.\n")
    
    print("="*80)

def generate_data_if_missing(config) -> bool:
    """
    ROBUST data validation and generation system.
    
    This function performs comprehensive validation and ensures data integrity by:
    1. Checking if all required files exist
    2. Validating data consistency (timestamps, timesteps, mode)
    3. Detecting ANY inconsistency and triggering full cleanup
    4. Regenerating everything from scratch if needed
    
    Args:
        config: Configuration object
        
    Returns:
        bool: True if data generation was successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"ROBUST DATA VALIDATION - {config.DATA_MODE.upper()} MODE")
    print(f"{'='*80}")
    print(f"Expected timesteps: {config.DATA_MODE_TIMESTEPS[config.DATA_MODE]}")
    print(f"Data directory: {config.DATA_DIR}")
    
    # STEP 1: Check if files exist
    data_exist, missing_files = check_data_files_exist(config)
    
    # STEP 2: Check data consistency (even if files exist)
    needs_regeneration = False
    regeneration_reason = []
    
    if not data_exist:
        needs_regeneration = True
        regeneration_reason.append(f"Missing {len(missing_files)} data files")
        print(f"\n❌ VALIDATION FAILED: {len(missing_files)} files missing")
        print("   Examples:")
        for i, file in enumerate(missing_files[:5]):
            print(f"      - {file}")
        if len(missing_files) > 5:
            print(f"      ... and {len(missing_files) - 5} more files")
    else:
        print("\n✓ All files present")
        
        # Even if files exist, validate consistency
        is_consistent, consistency_reason = check_data_consistency(config)
        
        if not is_consistent:
            needs_regeneration = True
            regeneration_reason.append(consistency_reason)
            print(f"❌ VALIDATION FAILED: {consistency_reason}")
        else:
            print(f"✓ Data consistent: {consistency_reason}")
    
    # STEP 3: If data is valid, skip generation
    if not needs_regeneration:
        print("\n" + "="*80)
        print("✓ DATA VALIDATION PASSED - Using existing data")
        print("="*80)
        return True
    
    # STEP 4: Data needs regeneration - perform AGGRESSIVE cleanup
    print(f"\n{'='*80}")
    print("DATA REGENERATION REQUIRED")
    print(f"{'='*80}")
    print("Reason(s):")
    for reason in regeneration_reason:
        print(f"  • {reason}")
    
    # AGGRESSIVE CLEANUP: Remove ALL data in the mode-specific folder
    print("\nPerforming aggressive cleanup to ensure data integrity...")
    clean_existing_data(config, aggressive=True)
    
    # Verify cleanup was successful and synchronize filesystem
    import glob
    data_dir = config.DATA_DIR
    remaining_patterns = ["*.npy", "*.txt", "*.json"]
    remaining_files = []
    for pattern in remaining_patterns:
        remaining_files.extend(glob.glob(os.path.join(data_dir, pattern)))
    
    if remaining_files:
        print(f"\n⚠ WARNING: {len(remaining_files)} files still remain after cleanup:")
        for f in remaining_files[:5]:
            print(f"   - {os.path.basename(f)}")
        print("   Waiting for filesystem sync...")
        time.sleep(2)
    else:
        print("✓ Cleanup verified: All old data removed")
    
    # CRITICAL: Synchronization delay before starting progress monitoring
    # This ensures file deletion is complete and filesystem is ready
    print("   Synchronizing filesystem (2 seconds)...")
    time.sleep(2)
    print("✓ Filesystem synchronized\n")
    
    # STEP 5: Generate fresh data with tqdm progress bar
    print(f"{'='*80}")
    print("GENERATING FRESH DATA")
    print(f"{'='*80}")
    
    try:
        data_gen_script = os.path.join("data", "gen_meas_best.py")
        
        if not os.path.exists(data_gen_script):
            print(f"❌ ERROR: Data generation script not found: {data_gen_script}")
            return False
        
        # Run data generation with tqdm progress monitoring
        timesteps = config.DATA_MODE_TIMESTEPS[config.DATA_MODE]
        print(f"Mode: {config.DATA_MODE} | Timesteps: {timesteps}")
        print("Starting data generation with progress tracking...\n")
        
        # Start monitoring progress in background thread (with tqdm)
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_data_generation_progress_per_system,
            args=(config, stop_event),
            daemon=True
        )
        monitor_thread.start()
        
        # Small delay to ensure progress bar is initialized
        time.sleep(0.5)
        
        # Run data generation script (capture output to avoid interfering with tqdm)
        result = subprocess.run(
            [sys.executable, data_gen_script, config.DATA_MODE, str(timesteps)],
            cwd=".",
            capture_output=True,  # Capture to avoid interfering with tqdm progress bar
            text=True
        )
        
        # Stop monitoring thread
        stop_event.set()
        monitor_thread.join(timeout=5)  # Wait for thread to finish
        
        # Print captured output after progress bar is done
        if result.stdout:
            # Only print non-progress lines to avoid clutter
            output_lines = result.stdout.split('\n')
            important_lines = [line for line in output_lines if line.strip() and 
                             'Progress:' not in line and 'timesteps' not in line.lower()]
            if important_lines:
                print("\nData generation output:")
                for line in important_lines[:10]:  # Show first 10 important lines
                    print(f"  {line}")
        
        if result.stderr:
            print("\nWarnings/Errors:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"\n❌ ERROR: Data generation failed (exit code {result.returncode})")
            return False
        
        # STEP 6: Verify generation was successful
        print(f"\n{'='*80}")
        print("VERIFYING GENERATED DATA")
        print(f"{'='*80}")
        
        data_exist_after, remaining_missing = check_data_files_exist(config)
        
        if data_exist_after:
            print("✓ All required files successfully generated")
            
            # Double-check consistency
            is_consistent_after, reason_after = check_data_consistency(config)
            if is_consistent_after:
                print(f"✓ Generated data is consistent: {reason_after}")
                print(f"\n{'='*80}")
                print("✓✓✓ DATA GENERATION SUCCESSFUL ✓✓✓")
                print(f"{'='*80}\n")
                return True
            else:
                print(f"❌ Generated data has consistency issues: {reason_after}")
                return False
        else:
            print(f"❌ Generation incomplete: {len(remaining_missing)} files still missing")
            print("   Examples:")
            for missing in remaining_missing[:5]:
                print(f"      - {missing}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: Exception during data generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def display_convergence_analysis(config, bus_systems_to_show=None):
    """
    Display detailed convergence analysis from generated convergence reports.
    Shows power flow success rates and any failures (especially with contingencies).
    
    Args:
        config: Configuration object
        bus_systems_to_show: List of bus systems to show (if None, shows all available)
    """
    import json
    import glob
    
    data_dir = config.DATA_DIR  # Use mode-specific directory
    # Use provided bus systems or default to config
    if bus_systems_to_show is not None:
        bus_systems = bus_systems_to_show
    else:
        bus_systems = config.NUM_BUSES if isinstance(config.NUM_BUSES, list) else [config.NUM_BUSES]
    
    renewable_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS")
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
                icon = "⚠" if has_cont_fail else "✓"
                rate = stats['success_rate']
                
                # Simple display showing just icon and success rate
                cell = f"{icon} {rate:5.1f}%"
                
                # Pad to fixed width
                row += f"| {cell:<{col_width-3}} "
            else:
                row += f"| {'---':<{col_width-3}} "
        print(row)
    
    # Calculate summary statistics
    total_timesteps = total_successful + total_failed
    
    # Check if ANY data was found
    if total_timesteps == 0:
        print("\n⚠ WARNING: No convergence data found!")
        print("  Data may not have been generated yet.")
        print("  Run with validate_data=True or manually run data/gen_meas_best.py")
        print("="*80)
        return  # Exit early
    
    overall_rate = (total_successful / total_timesteps * 100) if total_timesteps > 0 else 0
    
    print(f"\nOverall convergence: {overall_rate:.1f}% ({total_successful}/{total_timesteps} timesteps)")
    if scenarios_with_failures:
        print(f"Note: {len(scenarios_with_failures)} scenario(s) with contingency failures")
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
    print("FORCE REGENERATING ALL DATA")
    print("="*60)
    
    # Clean all existing data
    clean_existing_data(config)
    
    print("\nRunning data generation script...")
    
    try:
        # Run gen_meas_best.py from the data directory
        data_gen_script = os.path.join("data", "gen_meas_best.py")
        
        if not os.path.exists(data_gen_script):
            print(f"ERROR: Data generation script not found: {data_gen_script}")
            return False
        
        # Run the script with per-system progress bars
        print("Starting fresh data generation...\n")
        
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
            print("Fresh data generation completed successfully!")
            return True
        else:
            print(f"ERROR: Data generation failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error details: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"ERROR: Error running data generation: {e}")
        return False

def validate_data_before_training(config, bus_systems_to_show=None, run_integrity_analysis=True) -> bool:
    """
    Main function to validate data exists and generate if needed.
    All analysis (including convergence) is done by data integrity module.
    
    Args:
        config: Configuration object
        bus_systems_to_show: List of bus systems for data integrity analysis
        run_integrity_analysis: If True, run comprehensive data integrity analysis (default: True)
        
    Returns:
        bool: True if data is ready for training, False otherwise
    """
    print("\n" + "="*80)
    print("DATA VALIDATION")
    print("="*80)
    
    success = generate_data_if_missing(config)
    
    if success:
        # Run comprehensive data integrity analysis (includes convergence display)
        if run_integrity_analysis:
            print("\n" + "="*80)
            print("RUNNING COMPREHENSIVE DATA INTEGRITY ANALYSIS")
            print("="*80)
            
            try:
                from utils.data_integrity import analyze_data_integrity
                
                # Determine output directory (experimental_results/data_integrity)
                output_dir = os.path.join(config.EXPERIMENTAL_RESULTS_DIR, "data_integrity")
                
                # Run analysis for bus systems being tested
                if bus_systems_to_show:
                    cases = [f"case{bus}" for bus in bus_systems_to_show]
                else:
                    cases = None  # Auto-detect all
                
                analyze_data_integrity(config.DATA_DIR, output_dir, cases)
                
                print(f"\n✓ Data integrity analysis complete!")
                print(f"  Reports saved to: {output_dir}")
                
            except Exception as e:
                print(f"\n⚠ Warning: Data integrity analysis failed: {e}")
                import traceback
                traceback.print_exc()
                print("  Continuing with training...")
        
        print("\nReady for training!")
    else:
        print("\nData validation failed!")
    
    print("="*80)
    return success
