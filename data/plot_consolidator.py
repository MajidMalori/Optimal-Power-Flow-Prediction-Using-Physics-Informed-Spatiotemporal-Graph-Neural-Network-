"""
Consolidated Professional Plot Generation

Generates 3 essential, publication-quality plots for all bus systems:
1. Data Profile - Load/generation patterns and data quality
2. Convergence Story - Data generation quality metrics
3. Physics Health - Voltage distribution and system health

NO data_loader dependency - loads numpy files directly for speed.
"""

import os
from typing import Dict, List
from tqdm import tqdm

# Import new professional plotting modules
from data.plot_data_profile import plot_data_profile
from data.plot_convergence import plot_convergence_story
from data.plot_physics_health import plot_physics_health


def generate_all_data_plots(config, bus_systems: List[int], data_plots_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Generate 3 professional plots for each bus system.
    
    Args:
        config: Configuration object
        bus_systems: List of bus system numbers (e.g., [33, 57, 118])
        data_plots_dir: Directory to save all plots
    
    Returns:
        Dictionary mapping bus system to plot paths
    """
    os.makedirs(data_plots_dir, exist_ok=True)
    all_plot_paths = {}
    
    for num_buses in tqdm(bus_systems, desc="Generating plots", unit="system"):
        case_name = f"case{num_buses}"
        plot_status = []
        case_plot_paths = {}
        
        # Set case name in config for system-specific parameters
        config.CASE_NAME = case_name
        
        # ========== PLOT 1: Data Profile ==========
        try:
            output_path = os.path.join(data_plots_dir, f'data_profile_{num_buses}bus.png')
            result = plot_data_profile(config, case_name, output_path)
            if result:
                case_plot_paths['data_profile'] = result
                plot_status.append('profile')
        except Exception as e:
            tqdm.write(f"  {num_buses}-bus data profile error: {e}")
            case_plot_paths['data_profile'] = None
        
        # ========== PLOT 2: Convergence Story ==========
        try:
            output_path = os.path.join(data_plots_dir, f'convergence_{num_buses}bus.png')
            result = plot_convergence_story(config.DATA_DIR, case_name, output_path, config)
            if result:
                case_plot_paths['convergence'] = result
                plot_status.append('convergence')
        except Exception as e:
            tqdm.write(f"  {num_buses}-bus convergence error: {e}")
            case_plot_paths['convergence'] = None
        
        # ========== PLOT 3: Physics Health ==========
        try:
            output_path = os.path.join(data_plots_dir, f'physics_health_{num_buses}bus.png')
            result = plot_physics_health(config, case_name, output_path)
            if result:
                case_plot_paths['physics_health'] = result
                plot_status.append('physics')
        except Exception as e:
            tqdm.write(f"  {num_buses}-bus physics error: {e}")
            case_plot_paths['physics_health'] = None
        
        all_plot_paths[num_buses] = case_plot_paths
        tqdm.write(f"  {num_buses}-bus: {' + '.join(plot_status) if plot_status else 'failed'}")
    
    return all_plot_paths