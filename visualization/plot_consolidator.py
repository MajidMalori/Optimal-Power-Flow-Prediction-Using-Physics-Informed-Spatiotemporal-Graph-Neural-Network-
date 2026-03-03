"""
Consolidated Professional Plot Generation (YAML-Compatible)

Generates 3 essential, publication-quality plots for all bus systems:
1. Data Profile - Load/generation patterns and data quality
2. Convergence Story - Data generation quality metrics
3. Physics Health - Voltage distribution and system health
"""

import os
from typing import Dict, List
from tqdm import tqdm

# Import new professional plotting modules
from visualization.plot_data_profile import plot_data_profile
from visualization.plot_convergence import plot_convergence_story
from visualization.plot_physics_health import plot_physics_health
from visualization.plot_system_efficiency import plot_system_efficiency
from visualization.plot_topology import plot_topology_events


def generate_all_data_plots(config: dict, bus_systems: List[int], data_plots_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Generate professional plots using dictionary-based config.
    """
    os.makedirs(data_plots_dir, exist_ok=True)
    all_plot_paths = {}
    
    plot_types = 5
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
        
        # Determine Data Dir from config
        data_dir = config.get('output_dir', 'data/01_raw')
        
        # ========== PLOT 1: Data Profile ==========
        pbar.set_description(f"Generating plots (case{num_buses}, profile)")
        try:
            output_path = os.path.join(data_plots_dir, f'data_profile_{num_buses}bus.png')
            # The plotting scripts expect an object with attributes or a dict?
            # Let's assume they handle dict if we've refactored them, 
            # or we pass an SimpleNamespace if they expect attributes.
            # Most of our refactored code now uses dict.get()
            result = plot_data_profile(case_config, case_name, output_path)
            if result:
                case_plot_paths['data_profile'] = result
                plot_status.append('profile')
            pbar.update(1)
        except Exception as e:
            tqdm.write(f"  {num_buses}-bus data profile error: {e}")
            case_plot_paths['data_profile'] = None
            pbar.update(1)
        
        # ========== PLOT 2: Convergence Story ==========
        pbar.set_description(f"Generating plots (case{num_buses}, convergence)")
        try:
            output_path = os.path.join(data_plots_dir, f'convergence_{num_buses}bus.png')
            result = plot_convergence_story(data_dir, case_name, output_path, case_config)
            if result:
                case_plot_paths['convergence'] = result
                plot_status.append('convergence')
            pbar.update(1)
        except Exception as e:
            tqdm.write(f"  {num_buses}-bus convergence error: {e}")
            case_plot_paths['convergence'] = None
            pbar.update(1)
        
        # ========== PLOT 3: Physics Health ==========
        pbar.set_description(f"Generating plots (case{num_buses}, physics)")
        try:
            output_path = os.path.join(data_plots_dir, f'physics_health_{num_buses}bus.png')
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
            output_path = os.path.join(data_plots_dir, f'system_efficiency_{num_buses}bus.png')
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
            output_path = os.path.join(data_plots_dir, f'topology_{num_buses}bus.png')
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