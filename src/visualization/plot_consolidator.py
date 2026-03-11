"""
Consolidated Professional Plot Generation (YAML-Compatible)

Generates 4 essential, publication-quality plots for all bus systems:
1. Data Profile - Load/generation patterns and data quality
2. Reliability & Topology - Combined stability and switching report
3. Physics Health - Voltage distribution and system health
4. System Efficiency - Power loss and system performance
"""

import os
from typing import Dict, List
from tqdm import tqdm

# Import new professional plotting modules
from src.visualization.plot_data_profile import plot_data_profile
from src.visualization.plot_physics_health import plot_physics_health
from src.visualization.plot_system_efficiency import plot_system_efficiency
from src.visualization.plot_topology import plot_topology_events


def generate_all_data_plots(config: dict, bus_systems: List[int], data_plots_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Generate professional plots using dictionary-based config.
    """
    os.makedirs(data_plots_dir, exist_ok=True)
    all_plot_paths = {}
    
    plot_types = 4
    total_items = len(bus_systems) * plot_types
    
    pbar = tqdm(
        total=total_items,
        desc="Generating plots",
        unit="plot",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} plots"
    )
    
    for num_buses in bus_systems:
        case_name = f"case{num_buses}"
        plot_status = []
        case_plot_paths = {}
        
        # Inject case_name into a copy of config to avoid side effects
        case_config = config.copy()
        case_config['CASE_NAME'] = case_name
        
        # Determine case-specific Data Dir and Plots Dir
        base_data_dir = config.get('output_dir', 'data/raw')
        data_dir = os.path.join(base_data_dir, case_name)
        case_plots_dir = os.path.join(data_plots_dir, case_name)
        os.makedirs(case_plots_dir, exist_ok=True)
        
        # ========== PLOT 1: Data Profile ==========
        pbar.set_description(f"Generating plots (case{num_buses}, profile)")
        try:
            output_path = os.path.join(case_plots_dir, f'data_profile_{num_buses}bus.png')
            result = plot_data_profile(case_config, case_name, output_path)
            if result:
                case_plot_paths['data_profile'] = result
                plot_status.append('profile')
            pbar.update(1)
        except Exception as e:
            tqdm.write(f"  {num_buses}-bus data profile error: {e}")
            case_plot_paths['data_profile'] = None
            pbar.update(1)
        
        # ========== PLOT 3: Physics Health ==========
        pbar.set_description(f"Generating plots (case{num_buses}, physics)")
        try:
            output_path = os.path.join(case_plots_dir, f'physics_health_{num_buses}bus.png')
            result = plot_physics_health(case_config, case_name, output_path)
            if result:
                case_plot_paths['physics_health'] = result
                plot_status.append('physics')
            pbar.update(1)
        except Exception as e:
            tqdm.write(f"  {num_buses}-bus physics error: {e}")
            case_plot_paths['physics_health'] = None
            pbar.update(1)
        
        # ========== PLOT 4: System Efficiency ==========
        pbar.set_description(f"Generating plots (case{num_buses}, efficiency)")
        try:
            output_path = os.path.join(case_plots_dir, f'system_efficiency_{num_buses}bus.png')
            result = plot_system_efficiency(case_config, case_name, output_path)
            if result:
                case_plot_paths['system_efficiency'] = result
                plot_status.append('efficiency')
            pbar.update(1)
        except Exception as e:
            tqdm.write(f"  {num_buses}-bus efficiency error: {e}")
            case_plot_paths['system_efficiency'] = None
            pbar.update(1)

        # ========== PLOT 5: Topology Events ==========
        pbar.set_description(f"Generating plots (case{num_buses}, topology)")
        try:
            output_path = os.path.join(case_plots_dir, f'topology_{num_buses}bus.png')
            # Topology reads from the raw data arrays, which are now inside the case subfolder
            result = plot_topology_events(data_dir, case_name, output_path, case_config)
            if result:
                case_plot_paths['topology'] = result
                plot_status.append('topology')
            pbar.update(1)
        except Exception as e:
            tqdm.write(f"  {num_buses}-bus topology error: {e}")
            case_plot_paths['topology'] = None
            pbar.update(1)
            
        all_plot_paths[num_buses] = case_plot_paths
        tqdm.write(f"  {num_buses}-bus: {' + '.join(plot_status) if plot_status else 'failed'}")
    
    pbar.set_description("Plot generation complete")
    pbar.close()
    
    return all_plot_paths