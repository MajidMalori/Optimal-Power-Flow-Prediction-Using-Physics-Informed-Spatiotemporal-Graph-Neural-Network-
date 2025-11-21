"""
Bus Type Exploratory Data Analysis (EDA)

Validates and reports bus type distribution before training/evaluation.
Prevents plotting errors for non-existent bus types.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import warnings


def analyze_bus_types(bus_types: np.ndarray, case_name: str) -> Dict[str, any]:
    """
    Analyze bus type distribution in the dataset.
    
    Args:
        bus_types: Array [timesteps, buses] with bus type codes [0=PQ, 1=PV, 2=Slack]
        case_name: Name of the test case (e.g., "case33")
    
    Returns:
        Dictionary with bus type statistics and warnings
    """
    if bus_types is None:
        return {
            'has_data': False,
            'warning': 'Bus types not available in dataset'
        }
    
    # Flatten to get all bus type assignments across all timesteps
    bus_types_flat = bus_types.flatten()
    
    # Count occurrences of each bus type
    unique, counts = np.unique(bus_types_flat, return_counts=True)
    bus_type_counts = dict(zip(unique, counts))
    
    # Map codes to names
    bus_type_names = {0: 'PQ', 1: 'PV', 2: 'Slack'}
    
    # Calculate statistics
    num_timesteps, num_buses = bus_types.shape
    total_assignments = num_timesteps * num_buses
    
    stats = {
        'has_data': True,
        'num_timesteps': num_timesteps,
        'num_buses': num_buses,
        'total_assignments': total_assignments,
        'bus_type_counts': {},
        'bus_type_percentages': {},
        'has_pq': False,
        'has_pv': False,
        'has_slack': False,
        'warnings': []
    }
    
    # Analyze each bus type
    for code, name in bus_type_names.items():
        count = bus_type_counts.get(code, 0)
        percentage = (count / total_assignments * 100) if total_assignments > 0 else 0.0
        
        stats['bus_type_counts'][name] = count
        stats['bus_type_percentages'][name] = percentage
        
        if code == 0:
            stats['has_pq'] = count > 0
        elif code == 1:
            stats['has_pv'] = count > 0
        elif code == 2:
            stats['has_slack'] = count > 0
    
    # Generate warnings for missing bus types
    if not stats['has_pv']:
        stats['warnings'].append(
            f"WARNING: {case_name} has NO PV buses (conventional generators). "
            f"Plots for PV buses will be empty. This is expected for distribution feeders like case33bw."
        )
    
    if not stats['has_pq']:
        stats['warnings'].append(
            f"WARNING: {case_name} has NO PQ buses (load buses). This is unusual."
        )
    
    if not stats['has_slack']:
        stats['warnings'].append(
            f"WARNING: {case_name} has NO Slack buses (reference buses). This is unusual."
        )
    
    return stats


def print_bus_type_summary(bus_types: np.ndarray, case_name: str):
    """
    Print a summary of bus type distribution.
    
    Args:
        bus_types: Array [timesteps, buses] with bus type codes
        case_name: Name of the test case
    """
    stats = analyze_bus_types(bus_types, case_name)
    
    if not stats['has_data']:
        print(f"[Bus Type EDA] {case_name}: {stats['warning']}")
        return
    
    print(f"\n{'='*60}")
    print(f" BUS TYPE ANALYSIS: {case_name}")
    print(f"{'='*60}")
    print(f" Total timesteps: {stats['num_timesteps']}")
    print(f" Total buses: {stats['num_buses']}")
    print(f"\n Bus Type Distribution:")
    
    for bus_type in ['PQ', 'PV', 'Slack']:
        count = stats['bus_type_counts'].get(bus_type, 0)
        pct = stats['bus_type_percentages'].get(bus_type, 0.0)
        status = "✓" if count > 0 else "✗"
        print(f"   {status} {bus_type:6s}: {count:8d} assignments ({pct:5.1f}%)")
    
    # Print warnings
    if stats['warnings']:
        print(f"\n Warnings:")
        for warning in stats['warnings']:
            print(f"   {warning}")
    
    print(f"{'='*60}\n")


def get_available_bus_types(bus_types: np.ndarray) -> Tuple[bool, bool, bool]:
    """
    Quick check for which bus types are available.
    
    Returns:
        Tuple of (has_pq, has_pv, has_slack) booleans
    """
    if bus_types is None:
        return False, False, False
    
    unique = np.unique(bus_types)
    return (0 in unique, 1 in unique, 2 in unique)

