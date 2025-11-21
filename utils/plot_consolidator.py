"""
Consolidated plot generation for all bus systems.

Generates all data analysis plots (audit, data profile, convergence) for all bus systems
and saves them in a single data_plots folder before training starts.
"""

import os
import shutil
import traceback
import numpy as np
from typing import Dict, List, Optional
from utils.data_auditor import DataAuditor
from utils.data_profile_story import analyze_data_profiles
from utils.data_loader import load_power_system_data, PowerSystemLazyDataset
from config import FeatureIndices


def generate_all_data_plots(config, bus_systems: List[int], data_plots_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Generate all data analysis plots for all bus systems and save in consolidated folder.
    
    Args:
        config: Configuration object
        bus_systems: List of bus system numbers (e.g., [33, 57, 118])
        data_plots_dir: Directory to save all plots (e.g., experimental_results/run_XXX/data_plots)
    
    Returns:
        Dictionary mapping bus system to plot paths:
        {
            33: {
                'convergence_story': 'path/to/convergence_story_33.png',
                'data_profile': 'path/to/data_profile_33.png',
                'audit_physics': 'path/to/physics_health_33.png',
                'audit_composition': 'path/to/data_composition_33.png',
                'audit_curtailment': 'path/to/curtailment_impact_33.png',
                'audit_contingency': 'path/to/contingency_heatmap_33.png'
            },
            ...
        }
    """
    os.makedirs(data_plots_dir, exist_ok=True)
    all_plot_paths = {}
    
    print(f"\n[Plots] Generating consolidated data plots → {data_plots_dir}")
    
    for num_buses in bus_systems:
        case_name = f"case{num_buses}"
        print(f"  {case_name}: ", end="", flush=True)
        
        case_plot_paths = {}
        
        # Load data for this case (needed for plots)
        try:
            # Load data using lazy loading system
            data_tuple = load_power_system_data(config, case_name)
            # Handle both new (6 values) and old (4 values) return formats
            if len(data_tuple) == 6:
                file_metadata, adjacency, ybus_metadata, normalizer, topology_cache, topology_ids = data_tuple
            else:
                file_metadata, adjacency, ybus_metadata, normalizer = data_tuple[:4]
                topology_cache, topology_ids = None, None
            
            # Create lazy dataset
            dataset = PowerSystemLazyDataset(
                file_metadata=file_metadata,
                adjacency_matrix=adjacency,
                normalizer=normalizer,
                ybus_metadata=ybus_metadata,
                is_static=True,
                sequence_length=1,
                hours_per_day=config.HOURS_PER_DAY,
                topology_cache=topology_cache,
                topology_ids=topology_ids
            )
            
            # Load a subset of data for visualization (e.g., first 1000 samples or all if small)
            # For visualization, we need denormalized data
            # Load up to 2000 samples to keep it fast but representative
            num_samples_to_load = min(2000, len(dataset))
            
            # Pre-allocate arrays
            voltage_data_list = []
            p_ren_list = []
            
            # Use indices from config
            # Targets: [V, theta] for PQ, [Q, theta] for PV, [P, Q] for Slack
            # We want Voltage Magnitude (V). 
            # For PQ buses (type 0), target index 0 is V.
            # For PV buses (type 1), V is in features (index 8).
            # For Slack buses (type 2), V is in features (index 8).
            
            # We need to know bus types. They are in file_metadata or we can infer.
            # Let's just use the targets for PQ buses as a proxy for "System Health" 
            # or use feature index 8 (vm_meas) which should be close to actual V.
            
            for i in range(num_samples_to_load):
                sample = dataset[i]
                # Denormalize features and targets
                features = normalizer.denormalize(sample['features'].unsqueeze(0)).squeeze(0) # [buses, 10]
                targets = normalizer.denormalize_targets(sample['targets'].unsqueeze(0)).squeeze(0) # [buses, 2]
                
                # Extract Voltage: Use feature index 8 (vm_meas) as a good approximation for all buses
                # (since it's the "measured" voltage from power flow)
                vm_pu = features[:, 8].numpy()
                voltage_data_list.append(vm_pu)
                
                # Extract Renewable Generation
                p_ren = features[:, FeatureIndices.P_REN].numpy()
                p_ren_list.append(p_ren)
                
            voltage_data = np.stack(voltage_data_list) # [samples, buses]
            raw_solar = np.sum(np.stack(p_ren_list), axis=1) # [samples] (Total solar/wind)
            
            # For curtailment plot, we need a 24-hour slice
            if num_samples_to_load >= 24:
                time_slice = np.arange(24)
                raw_solar_slice = raw_solar[:24]
                # We don't have explicit curtailed data, so we'll use raw_solar for both
                # but add a small simulated curtailment for visualization if we detect it in audit
                curtailed_solar_slice = raw_solar_slice 
            else:
                time_slice = None
                raw_solar_slice = None
                curtailed_solar_slice = None
                
        except Exception as e:
            print(f"  Warning: Could not load data for plotting: {e}")
            voltage_data = None
            raw_solar_slice = None
            curtailed_solar_slice = None
            time_slice = None
        
        # 1. Convergence Story Plot
        try:
            convergence_path = DataAuditor.plot_convergence_story(
                data_dir=config.DATA_DIR,
                case_name=case_name,
                output_dir=data_plots_dir,  # Save directly to consolidated folder
                config=config
            )
            # Rename to include bus number for clarity
            if convergence_path and os.path.exists(convergence_path):
                new_convergence_path = os.path.join(data_plots_dir, f'convergence_story_{num_buses}bus.png')
                if convergence_path != new_convergence_path:
                    shutil.move(convergence_path, new_convergence_path)
                else:
                    new_convergence_path = convergence_path
                case_plot_paths['convergence_story'] = new_convergence_path
                print(f"  ✓ Convergence story: {new_convergence_path}")
            else:
                print(f"  ✗ Convergence story file not found")
                case_plot_paths['convergence_story'] = None
        except Exception as e:
            print(f"  ✗ Convergence story failed: {e}")
            traceback.print_exc()
            case_plot_paths['convergence_story'] = None
        
        # 2. Data Profile Story Plot
        try:
            issues = analyze_data_profiles(
                config=config,
                case_name=case_name,
                features=None,
                normalizer=None,
                renewable_fractions=None
            )
            # The function saves to bus-specific folder, we need to move it
            # Check multiple possible locations
            possible_paths = [
                os.path.join(config.CURRENT_RUN_DIR, f"{num_buses}bus", 'data_profile_story.png'),
                os.path.join(config.EXPERIMENTAL_RESULTS_DIR, f"{num_buses}bus", 'data_profile_story.png'),
                os.path.join(config.EXPERIMENTAL_RESULTS_DIR, f"{case_name.replace('case', '')}bus", 'data_profile_story.png'),
            ]
            
            new_profile_path = os.path.join(data_plots_dir, f'data_profile_story_{num_buses}bus.png')
            found = False
            for original_path in possible_paths:
                if os.path.exists(original_path):
                    shutil.move(original_path, new_profile_path)
                    case_plot_paths['data_profile'] = new_profile_path
                    print("profile", end=" ", flush=True)
                    found = True
                    break
            
            if not found:
                print("profile✗", end=" ", flush=True)
                case_plot_paths['data_profile'] = None
        except Exception as e:
            print("profile✗", end=" ", flush=True)
            case_plot_paths['data_profile'] = None
        
        # 3. Audit Plots (4 detailed plots per bus system)
        try:
            # Load audit data for this case
            audits = DataAuditor.load_all_audits(config.DATA_DIR, case_name)
            if audits:
                # Create DataAuditor instance with first audit (DataAuditor needs audit_dict, not data_dir)
                first_audit_key = list(audits.keys())[0]
                auditor = DataAuditor(audit_dict=audits[first_audit_key])
                
                # Generate all 4 audit plots
                audit_plot_paths = auditor.generate_all_plots(
                    output_dir=data_plots_dir,
                    config=config,
                    case_name=case_name,
                    voltage_data=voltage_data,
                    raw_solar=raw_solar_slice,
                    curtailed_solar=curtailed_solar_slice,
                    time_slice=time_slice
                )
                
                # Rename plots to include bus number
                renamed_paths = {}
                for plot_name, plot_path in audit_plot_paths.items():
                    if plot_path and os.path.exists(plot_path):
                        # Extract filename and add bus number
                        base_name = os.path.basename(plot_path)
                        name, ext = os.path.splitext(base_name)
                        new_name = f"{name}_{num_buses}bus{ext}"
                        new_path = os.path.join(data_plots_dir, new_name)
                        shutil.move(plot_path, new_path)
                        renamed_paths[plot_name] = new_path
                    else:
                        renamed_paths[plot_name] = None
                
                case_plot_paths['audit_physics'] = renamed_paths.get('physics_health')
                case_plot_paths['audit_composition'] = renamed_paths.get('data_composition')
                case_plot_paths['audit_curtailment'] = renamed_paths.get('curtailment_impact')
                case_plot_paths['audit_contingency'] = renamed_paths.get('contingency_heatmap')
                
                audit_count = len([p for p in renamed_paths.values() if p])
                print(f"audit({audit_count}/4)", end=" ", flush=True)
            else:
                print("audit(0/4)", end=" ", flush=True)
                case_plot_paths['audit_physics'] = None
                case_plot_paths['audit_composition'] = None
                case_plot_paths['audit_curtailment'] = None
                case_plot_paths['audit_contingency'] = None
        except Exception as e:
            print("audit✗", end=" ", flush=True)
            case_plot_paths['audit_physics'] = None
            case_plot_paths['audit_composition'] = None
            case_plot_paths['audit_curtailment'] = None
            case_plot_paths['audit_contingency'] = None
        
        all_plot_paths[num_buses] = case_plot_paths
        print("done", flush=True)
    
    print(f"[Plots] All plots saved to: {data_plots_dir}\n")
    
    return all_plot_paths

