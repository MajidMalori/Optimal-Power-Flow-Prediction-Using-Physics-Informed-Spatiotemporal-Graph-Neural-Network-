#!/usr/bin/env python3
"""
Data validation utilities for ensuring data files exist before training.
"""

import os
import subprocess
import sys
import threading
import time
import glob
import re
import json
import shutil
import numpy as np
import hashlib
from typing import List, Tuple, Dict, Set
from tqdm import tqdm

def find_latest_timestamp(data_dir: str) -> str:
    """
    Find the latest timestamp from existing data files.
    
    Returns:
        Latest timestamp string found in filenames, or None if no timestamped files exist
    """
    
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

def check_data_files_exist(config, bus_systems=None) -> Tuple[bool, List[str]]:
    """
    Check if all required data files exist for the specified bus systems.
    Now supports both legacy (no timestamp) and new timestamped file formats.
    
    Args:
        config: Configuration object containing NUM_BUSES and renewable fractions
        bus_systems: Optional list of bus systems to check (if None, uses config.NUM_BUSES)
        
    Returns:
        Tuple of (all_files_exist: bool, missing_files: List[str])
    """
    if bus_systems is None:
        bus_systems = config.NUM_BUSES if isinstance(config.NUM_BUSES, list) else [config.NUM_BUSES]
    elif not isinstance(bus_systems, list):
        bus_systems = [bus_systems]
    renewable_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Standard fractions from gen_meas_best.py
    
    missing_files = []
    data_dir = config.DATA_DIR  # Use mode-specific directory from config
    
    # First, try to find the latest timestamp from existing files
    latest_timestamp = find_latest_timestamp(data_dir)
    
    for num_buses in bus_systems:
        case_name = f"case{num_buses}"
        for frac in renewable_fractions:
            # NEW FORMAT: Topology caching system requires base_adjacency and topology_ids
            # OLD FORMAT: adjacency.npy (no longer generated, but checked for backward compatibility)
            required_file_types = [
                "features.npy",
                "targets.npy", 
                "base_adjacency.npy",  # NEW: Single base adjacency matrix
                "topology_ids.npy",    # NEW: Topology ID for each timestep
                "time_carbon_coeffs.txt"
                # Note: Generation components are now included in features/targets matrices
            ]
            
            # Legacy format check (for backward compatibility detection)
            legacy_file_types = [
                "adjacency.npy"  # Old format - if this exists but new format doesn't, trigger regeneration
            ]
            
            # Add Ybus files - check for sparse format
            sparse_ybus_types = [
                "ybus_base.npy",
                "ybus_contingency_timesteps.npy",
                "ybus_contingency_matrices.npy",
                "data_quality_audit.json"  # New professional format (legacy convergence_report.json also accepted)
            ]
            
            # OPF: Check for bus_types file (new structure)
            opf_files = [
                "bus_types.npy"  # OPF: bus type codes [0=PQ, 1=PV, 2=Slack]
            ]
            
            # Try sparse format first
            dense_ybus_type = "ybus_matrices.npy"
            
            # Check common files (features, targets, etc.) - NEW FORMAT REQUIRED
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
            
            # Check for legacy adjacency format - if it exists but new format doesn't, trigger regeneration
            legacy_adjacency_found = False
            for file_type in legacy_file_types:
                if latest_timestamp:
                    base_name = f"{case_name}_{file_type.split('.')[0]}_frac{frac:.1f}_{latest_timestamp}"
                    filename = base_name + '.' + file_type.split('.')[1]
                    filepath = os.path.join(data_dir, filename)
                    if os.path.exists(filepath):
                        legacy_adjacency_found = True
                        break
                
                if not legacy_adjacency_found:
                    base_name = f"{case_name}_{file_type.split('.')[0]}_frac{frac:.1f}"
                    filename = base_name + '.' + file_type.split('.')[1]
                    filepath = os.path.join(data_dir, filename)
                    if os.path.exists(filepath):
                        legacy_adjacency_found = True
                        break
            
            # If legacy format exists but new format is missing, mark as missing (triggers regeneration)
            if legacy_adjacency_found:
                new_format_exists = False
                # Check if base_adjacency exists
                if latest_timestamp:
                    base_adj_path = os.path.join(data_dir, f"{case_name}_base_adjacency_frac{frac:.1f}_{latest_timestamp}.npy")
                    if os.path.exists(base_adj_path):
                        new_format_exists = True
                if not new_format_exists:
                    base_adj_path = os.path.join(data_dir, f"{case_name}_base_adjacency_frac{frac:.1f}.npy")
                    if os.path.exists(base_adj_path):
                        new_format_exists = True
                
                if not new_format_exists:
                    missing_files.append(f"{case_name}_base_adjacency_frac{frac:.1f}.npy (LEGACY FORMAT DETECTED - regeneration required)")
                    missing_files.append(f"{case_name}_topology_ids_frac{frac:.1f}.npy (LEGACY FORMAT DETECTED - regeneration required)")
            
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
                
                # Special handling for data_quality_audit.json - also accept legacy convergence_report.json
                if ybus_type == "data_quality_audit.json":
                    # Try new format first
                    if latest_timestamp:
                        base_name = f"{case_name}_data_quality_audit_frac{frac:.1f}_{latest_timestamp}"
                        filename = base_name + '.json'
                        filepath = os.path.join(data_dir, filename)
                        if os.path.exists(filepath):
                            found = True
                    
                    if not found:
                        base_name = f"{case_name}_data_quality_audit_frac{frac:.1f}"
                        filename = base_name + '.json'
                        filepath = os.path.join(data_dir, filename)
                        if os.path.exists(filepath):
                            found = True
                    
                    # Fallback to legacy convergence_report.json
                    if not found:
                        if latest_timestamp:
                            base_name = f"{case_name}_convergence_report_frac{frac:.1f}_{latest_timestamp}"
                            filename = base_name + '.json'
                            filepath = os.path.join(data_dir, filename)
                            if os.path.exists(filepath):
                                found = True
                        
                        if not found:
                            base_name = f"{case_name}_convergence_report_frac{frac:.1f}"
                            filename = base_name + '.json'
                            filepath = os.path.join(data_dir, filename)
                            if os.path.exists(filepath):
                                found = True
                else:
                    # Normal file checking
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
                pbar.set_description("Data generation complete")
                pbar.close()
                return
            
            time.sleep(0.2)  # Check every 0.2 seconds
            
    except KeyboardInterrupt:
        pbar.set_description("Interrupted")
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
    
    data_dir = config.DATA_DIR  # Use mode-specific directory (train or test)
    files_removed = 0
    
    print(f"\n{'='*80}\nCLEANING ALL {config.DATA_MODE.upper()} DATA | Dir: {data_dir}\n{'='*80}")
    
    if not os.path.exists(data_dir):
        print("Directory doesn't exist yet. Nothing to clean.")
        return
    
    if aggressive:
        # AGGRESSIVE CLEANUP: Remove the entire mode-specific directory and recreate it
        # This ensures absolutely NO leftover files from previous runs
        try:
            shutil.rmtree(data_dir)
            os.makedirs(data_dir, exist_ok=True)
            print(f"Cleaned and recreated directory: {data_dir}\n")
            return
        except Exception as e:
            print(f"WARNING: Could not remove directory: {e}. Falling back to file-by-file cleanup...\n")
    
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
            if filename in ["main.py", "check_data.py"]:
                continue
                
            try:
                os.remove(filepath)
                files_removed += 1
            except OSError as e:
                print(f"WARNING: Could not remove {filepath}: {e}")
    
    if files_removed > 0:
        print(f"Removed {files_removed} data files from {config.DATA_MODE} folder. Ready for fresh data generation.\n")
    else:
        print("No existing data files found to clean.\n")

def compute_config_hash(config) -> str:
    """
    Compute a hash of key configuration parameters that affect data generation.
    This allows detecting when data needs regeneration due to config changes.
    
    Args:
        config: Configuration object
        
    Returns:
        Hex string hash of configuration
    """
    key_params = {
        'data_mode': config.DATA_MODE,
        'timesteps': config.DATA_MODE_TIMESTEPS[config.DATA_MODE],
        'hours_per_day': getattr(config, 'HOURS_PER_DAY', 24),
        'contingency_rate': getattr(config, 'CONTINGENCY_RATE', 0.05),
        'pmu_coverage': getattr(config, 'PMU_COVERAGE', 0.3),
    }
    
    # Create a deterministic string representation
    param_str = json.dumps(key_params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()

def validate_bus_system_data(config, bus_system: int) -> Tuple[bool, str, Dict]:
    """
    Validate data for a specific bus system individually.
    Returns detailed validation results.
    
    Args:
        config: Configuration object
        bus_system: Bus system number (e.g., 33, 57, 118)
        
    Returns:
        Tuple of (is_valid: bool, reason: str, details: dict)
    """
    case_name = f"case{bus_system}"
    data_dir = config.DATA_DIR
    expected_timesteps = config.DATA_MODE_TIMESTEPS[config.DATA_MODE]
    
    details = {
        'bus_system': bus_system,
        'case_name': case_name,
        'files_exist': False,
        'timesteps_match': False,
        'mode_match': False,
        'config_hash_match': False,
        'actual_timesteps': None,
        'stored_mode': None,
        'stored_config_hash': None,
    }
    
    # Check if files exist
    bus_systems_list = [bus_system]
    files_exist, missing_files = check_data_files_exist(config, bus_systems_list)
    details['files_exist'] = files_exist
    details['missing_files'] = missing_files
    
    if not files_exist:
        return False, f"Missing {len(missing_files)} files for {case_name}", details
    
    # Check metadata for configuration consistency
    metadata_file = os.path.join(data_dir, "data_generation_metadata.json")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if metadata has 'runs' array (from parallel execution)
            if 'runs' in metadata and isinstance(metadata['runs'], list):
                # Find the run that includes this bus system
                relevant_run = None
                for run in metadata['runs']:
                    if case_name in run.get('test_cases', []):
                        relevant_run = run
                        break
                
                if relevant_run:
                    stored_timesteps = relevant_run.get('timesteps', 0)
                    stored_mode = relevant_run.get('data_mode', 'unknown')
                    stored_config_hash = relevant_run.get('config_hash', None)
                else:
                    # Fallback to first run if this bus system not found
                    relevant_run = metadata['runs'][0] if metadata['runs'] else {}
                    stored_timesteps = relevant_run.get('timesteps', 0)
                    stored_mode = relevant_run.get('data_mode', 'unknown')
                    stored_config_hash = relevant_run.get('config_hash', None)
            else:
                # Single run format
                stored_timesteps = metadata.get('timesteps', 0)
                stored_mode = metadata.get('data_mode', 'unknown')
                stored_config_hash = metadata.get('config_hash', None)
            
            details['actual_timesteps'] = stored_timesteps
            details['stored_mode'] = stored_mode
            details['stored_config_hash'] = stored_config_hash
            
            # Check mode match
            if stored_mode != config.DATA_MODE:
                details['mode_match'] = False
                return False, f"{case_name}: Mode mismatch (stored={stored_mode}, required={config.DATA_MODE})", details
            details['mode_match'] = True
            
            # Check timesteps match
            if stored_timesteps != expected_timesteps:
                details['timesteps_match'] = False
                return False, f"{case_name}: Timesteps mismatch (stored={stored_timesteps}, required={expected_timesteps})", details
            details['timesteps_match'] = True
            
            # Check config hash if available
            current_config_hash = compute_config_hash(config)
            if stored_config_hash:
                if stored_config_hash != current_config_hash:
                    details['config_hash_match'] = False
                    return False, f"{case_name}: Configuration changed (config hash mismatch)", details
                details['config_hash_match'] = True
            else:
                # If no hash stored, we can't verify - but timesteps match is good enough
                details['config_hash_match'] = None
            
        except Exception as e:
            return False, f"{case_name}: Error reading metadata: {e}", details
    else:
        # No metadata - check file directly
        pattern = os.path.join(data_dir, f"{case_name}_features_frac*.npy")
        feature_files = glob.glob(pattern)
        if feature_files:
            try:
                features = np.load(feature_files[0])
                actual_timesteps = features.shape[0]
                details['actual_timesteps'] = actual_timesteps
                
                if actual_timesteps != expected_timesteps:
                    details['timesteps_match'] = False
                    return False, f"{case_name}: Timesteps mismatch (file={actual_timesteps}, required={expected_timesteps})", details
                details['timesteps_match'] = True
            except Exception as e:
                return False, f"{case_name}: Error reading data file: {e}", details
        else:
            return False, f"{case_name}: No data files found", details
    
    return True, f"{case_name}: Valid", details

def clean_bus_system_data(config, bus_system: int) -> int:
    """
    Clean data files for a specific bus system only.
    
    Args:
        config: Configuration object
        bus_system: Bus system number to clean
        
    Returns:
        Number of files removed
    """
    case_name = f"case{bus_system}"
    data_dir = config.DATA_DIR
    files_removed = 0
    
    if not os.path.exists(data_dir):
        return 0
    
    # Find all files matching this bus system
    patterns = [
        f"{case_name}_*_frac*.npy",
        f"{case_name}_*_frac*.txt",
        f"{case_name}_*_frac*.json",
    ]
    
    for pattern in patterns:
        file_pattern = os.path.join(data_dir, pattern)
        files_to_remove = glob.glob(file_pattern)
        
        for filepath in files_to_remove:
            try:
                os.remove(filepath)
                files_removed += 1
            except OSError as e:
                print(f"Warning: Could not remove {filepath}: {e}")
    
    return files_removed

def generate_data_if_missing(config, bus_systems=None) -> bool:
    """
    ROBUST data validation and generation system.
    
    This function performs comprehensive validation and ensures data integrity by:
    1. Checking if all required files exist
    2. Validating data consistency (timestamps, timesteps, mode)
    3. Detecting ANY inconsistency and triggering full cleanup
    4. Regenerating everything from scratch if needed
    
    Args:
        config: Configuration object
        bus_systems: Optional list of bus systems to validate/generate (if None, uses config.NUM_BUSES)
        
    Returns:
        bool: True if data generation was successful, False otherwise
    """
    print(f"\n{'='*80}\nINTELLIGENT DATA VALIDATION - {config.DATA_MODE.upper()} MODE | Timesteps: {config.DATA_MODE_TIMESTEPS[config.DATA_MODE]} | Dir: {config.DATA_DIR}\n{'='*80}")
    
    # Determine which bus systems to validate
    if bus_systems is None:
        bus_systems_to_validate = config.NUM_BUSES if isinstance(config.NUM_BUSES, list) else [config.NUM_BUSES]
    elif isinstance(bus_systems, list):
        bus_systems_to_validate = bus_systems
    else:
        bus_systems_to_validate = [bus_systems]
    
    bus_list_str = ", ".join([f"case{b}" for b in bus_systems_to_validate])
    print(f"Validating bus systems: {bus_list_str}")
    
    # STEP 1: Per-bus-system validation (PROFESSIONAL APPROACH)
    validation_results = {}
    systems_need_regeneration = []
    systems_valid = []
    
    print(f"\n{'─'*80}")
    print("Per-Bus-System Validation:")
    print(f"{'─'*80}")
    
    for bus_system in bus_systems_to_validate:
        is_valid, reason, details = validate_bus_system_data(config, bus_system)
        validation_results[bus_system] = {'valid': is_valid, 'reason': reason, 'details': details}
        
        if is_valid:
            systems_valid.append(bus_system)
            print(f"  ✓ {reason}")
        else:
            systems_need_regeneration.append(bus_system)
            print(f"  ✗ {reason}")
    
    # STEP 2: Summary and decision
    print(f"\n{'─'*80}")
    print("Validation Summary:")
    print(f"{'─'*80}")
    print(f"  Valid systems: {len(systems_valid)}/{len(bus_systems_to_validate)}")
    if systems_valid:
        print(f"    → {', '.join([f'case{b}' for b in systems_valid])}")
    print(f"  Systems needing regeneration: {len(systems_need_regeneration)}/{len(bus_systems_to_validate)}")
    if systems_need_regeneration:
        print(f"    → {', '.join([f'case{b}' for b in systems_need_regeneration])}")
        for bus_system in systems_need_regeneration:
            reason = validation_results[bus_system]['reason']
            print(f"      • case{bus_system}: {reason}")
    
    # STEP 3: If all systems are valid, skip generation
    if not systems_need_regeneration:
        print(f"\n{'='*80}\n✓ All data validation passed - Using existing data\n{'='*80}")
        return True
    
    # STEP 4: Selective regeneration (only for systems that need it)
    print(f"\n{'='*80}\nSELECTIVE DATA REGENERATION REQUIRED\n{'='*80}")
    print(f"\nRegenerating only invalid systems: {', '.join([f'case{b}' for b in systems_need_regeneration])}")
    print(f"Preserving valid systems: {', '.join([f'case{b}' for b in systems_valid]) if systems_valid else 'None'}")
    
    # Selective cleanup: Only remove files for systems that need regeneration
    print(f"\n{'─'*80}")
    print("Selective Cleanup (preserving valid data):")
    print(f"{'─'*80}")
    
    total_files_removed = 0
    for bus_system in systems_need_regeneration:
        files_removed = clean_bus_system_data(config, bus_system)
        total_files_removed += files_removed
        print(f"  Cleaned case{bus_system}: {files_removed} files removed")
    
    if total_files_removed > 0:
        print(f"\nTotal files removed: {total_files_removed}")
    else:
        print("\nNo files to remove (already clean)")
    
    # Small delay for filesystem sync
    time.sleep(0.5)
    
    # STEP 5: Generate data only for systems that need it
    print(f"\n{'='*80}")
    print("GENERATING DATA FOR INVALID SYSTEMS")
    print(f"{'='*80}")
    
    try:
        data_gen_script = os.path.join("data", "main.py")
        
        if not os.path.exists(data_gen_script):
            print(f"Error: Data generation script not found: {data_gen_script}")
            return False
        
        timesteps = config.DATA_MODE_TIMESTEPS[config.DATA_MODE]
        
        # Generate data for each system that needs regeneration
        for bus_system in systems_need_regeneration:
            print(f"\n{'─'*80}")
            print(f"Generating data for case{bus_system}...")
            print(f"{'─'*80}")
            
            # Start monitoring progress in background thread
            stop_event = threading.Event()
            monitor_thread = threading.Thread(
                target=monitor_data_generation_progress_per_system,
                args=(config, stop_event),
                daemon=True
            )
            monitor_thread.start()
            
            time.sleep(0.5)
            
            # Run data generation for this specific bus system
            buses_arg = str(bus_system)
            result = subprocess.run(
                [sys.executable, data_gen_script, 
                 "--mode", config.DATA_MODE, 
                 "--time_steps", str(timesteps),
                 "--buses", buses_arg,
                 "--no_progress_bar"],
                cwd=".",
                capture_output=True,
                text=True
            )
            
            # Stop monitoring thread
            stop_event.set()
            monitor_thread.join(timeout=5)
            
            if result.returncode != 0:
                print(f"\n✗ Error: Data generation failed for case{bus_system} (exit code {result.returncode})")
                if result.stderr:
                    print(f"Error details: {result.stderr[:500]}")
                return False
            else:
                print(f"✓ Successfully generated data for case{bus_system}")
        
        # STEP 6: Verify all systems are now valid
        print(f"\n{'='*80}")
        print("VERIFYING GENERATED DATA")
        print(f"{'='*80}")
        
        all_valid = True
        for bus_system in bus_systems_to_validate:
            is_valid, reason, details = validate_bus_system_data(config, bus_system)
            if is_valid:
                print(f"  ✓ {reason}")
            else:
                print(f"  ✗ {reason}")
                all_valid = False
        
        if all_valid:
            print(f"\n{'='*80}")
            print("✓ All data validation passed - Ready for training")
            print(f"{'='*80}\n")
            return True
        else:
            print(f"\n{'='*80}")
            print("✗ Some systems still have validation issues")
            print(f"{'='*80}\n")
            return False
            
    except Exception as e:
        print(f"\nError: Exception during data generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def display_data_generation_summary(config, bus_systems_to_show=None):
    """
    Display concise data generation summary from audit files.
    Shows success rates, curtailment, and trip/fail rates for each case and renewable fraction.
    
    Args:
        config: Configuration object
        bus_systems_to_show: List of bus systems to show (if None, shows all available)
    """
    
    data_dir = config.DATA_DIR
    if bus_systems_to_show is not None:
        bus_systems = bus_systems_to_show
    else:
        bus_systems = config.NUM_BUSES if isinstance(config.NUM_BUSES, list) else [config.NUM_BUSES]
    
    # Find latest timestamp from existing files
    latest_timestamp = find_latest_timestamp(data_dir)
    if not latest_timestamp:
        print("\nNo timestamped data files found. Cannot display summary.")
        return
    
    print("\n" + "="*80)
    print("DATA GENERATION SUMMARY")
    print("="*80)
    print(f"{'Case':<10} {'Frac':<6} {'Success%':<10} {'Curtail%':<10} {'Trip/Fail%':<12} {'Status'}")
    print("-" * 80)
    
    for num_buses in bus_systems:
        case_name = f"case{num_buses}"
        renewable_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        for frac in renewable_fractions:
            # Find audit file for this case and fraction
            pattern = f"{case_name}_data_quality_audit_frac{frac:.1f}_{latest_timestamp}.json"
            audit_file = os.path.join(data_dir, pattern)
            
            if not os.path.exists(audit_file):
                # Try without timestamp
                pattern_no_ts = f"{case_name}_data_quality_audit_frac{frac:.1f}.json"
                audit_file = os.path.join(data_dir, pattern_no_ts)
                
                if not os.path.exists(audit_file):
                    continue
            
            try:
                with open(audit_file, 'r') as f:
                    audit = json.load(f)
                
                case_short = case_name.replace('case', '')
                
                # Extract actual values from audit structure
                total = audit.get('total_timesteps', 1)
                successful_count = audit.get('successful', 0)
                failed_count = audit.get('failed', 0)
                
                # Calculate percentages
                success = 100 * successful_count / total if total > 0 else 0
                fail = 100 * failed_count / total if total > 0 else 0
                
                # Check for curtailment data (might be in intervention_stats or separate)
                intervention = audit.get('intervention_stats', {})
                curtail = intervention.get('curtailed_rate', 0)
                
                # Status without emojis
                status = "OK"
                if fail > 0: status = "Issues"
                if fail > 5: status = "Critical"
                
                print(f"{case_short:<10} {frac:<6.1f} {success:<10.1f} {curtail:<10.1f} {fail:<12.1f} {status}")
                
            except Exception as e:
                print(f"{case_name.replace('case', ''):<10} {frac:<6.1f} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12} Error")
    
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
        # Run data/main.py from the data directory
        data_gen_script = os.path.join("data", "main.py")
        
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

def validate_data_before_training(config, bus_systems_to_test=None) -> bool:
    """
    Main function to validate data exists and generate if needed.
    
    Args:
        config: Configuration object
        bus_systems_to_test: List of bus systems to validate (if None, validates all in config.NUM_BUSES)
        
    Returns:
        bool: True if data is ready for training, False otherwise
    """
    print("\n" + "="*80)
    print("DATA VALIDATION")
    print("="*80)
    
    # Pass bus_systems to generation function
    success = generate_data_if_missing(config, bus_systems_to_test)
    
    if success:
        # Display data generation summary table (shows success/curtail/fail rates)
        try:
            display_data_generation_summary(config, bus_systems_to_test)
        except Exception as e:
            print(f"Warning: Could not display data generation summary: {e}")
        
        print("\nReady for training!")
    else:
        print("\nData validation failed!")
    
    print("="*80)
    return success
