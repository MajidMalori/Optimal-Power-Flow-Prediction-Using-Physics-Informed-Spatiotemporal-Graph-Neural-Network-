import os
import sys
import traceback
import json
import warnings
import copy
import gc
import time
import shutil
import tempfile
import random
import argparse
import yaml
import shutil
import time
from datetime import datetime
import numpy as np
import pandas as pd
import pandapower as pp
from tqdm import tqdm
from scipy import sparse


# Add parent directory to Python path so we can import utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from refactored modules
from data.profiles import (
    get_daily_load_profile, 
    get_solar_generation_profile, 
    get_wind_generation_profile,
    simulate_weather_sequence,
    calculate_renewable_reactive_power
)
from data.topology import (
    load_network, 
    configure_renewables, 
    apply_n1_contingency, 
    restore_contingency,
    calculate_ybus_from_net,
    calculate_adjacency_matrix,
    identify_bus_types,
    create_opf_targets
)
from data.validation import (
    SuppressPrints,
    validate_power_flow_inputs,
    validate_power_flow_outputs,
    apply_curtailment_with_retry,
    hard_reset_system,
    trip_renewable_generators
)
from utils.contingency_ybus import DataGenerationError

from data.data_auditor import transform_convergence_to_audit

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# CONFIGURATION
# Refactored to use Config class and CLI arguments
from config import Config

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Physics-Informed Data Generation')
    parser.add_argument('--time_steps', type=int, default=None, help='Number of time steps to generate')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save generated data')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML configuration file')
    parser.add_argument('--mode', type=str, default=None, choices=['train', 'test'], help='Data generation mode (train/test)')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar (useful when running from train.py)')
    return parser.parse_known_args()[0]

# Parse arguments
args = parse_arguments()

# Initialize Config
# CLI arguments override YAML configuration
try:
    # Determine data mode: CLI > Config > Default
    # We need to peek at Config default if CLI is None, but Config isn't loaded yet.
    # So we load Config with CLI override if present, otherwise let Config use its internal default logic (which reads YAML)
    # However, Config init requires data_mode.
    # Let's pass CLI mode if present, otherwise 'test' (or whatever default we want if YAML doesn't specify, but YAML is required).
    # Actually Config loads YAML.
    
    config_instance = Config(
        yaml_config_path=args.config,
        load_yaml=True,
        # If CLI args are provided, they override the config defaults
        data_mode=args.mode if args.mode is not None else getattr(Config, 'data_mode', 'test'),
        train_timesteps=args.time_steps, 
        test_timesteps=args.time_steps,
        save_results=True
    )
except Exception as e:
    print(f"Error loading configuration: {e}")
    sys.exit(1)

# Create CONFIG dictionary for compatibility with existing code
# Prioritize CLI args > Config object > Defaults
data_mode = config_instance.DATA_MODE # Config has already resolved mode
default_timesteps = config_instance.DATA_MODE_TIMESTEPS[data_mode]
timesteps_to_use = args.time_steps if args.time_steps is not None else default_timesteps
output_dir_to_use = args.output_dir if args.output_dir is not None else config_instance.DATA_DIR

# Debug: Print mode and output directory
# print(f"[Data Generation] Mode: {data_mode}, Output Directory: {output_dir_to_use}")

CONFIG = {
    "random_seed": 42,
    "test_cases": getattr(Config, 'test_cases', ["case33", "case57", "case118"]),
    "time_steps": timesteps_to_use,
    "output_dir": output_dir_to_use,
    "renewable_fractions_to_run": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "contingency_rate": getattr(Config, 'contingency_rate', 0.05),
    "voltage_error_std": 0.005,
    "power_error_std": 0.01,
    "angle_error_std": 0.02,
    "max_energy_utilization_coeff": 0.98,
    "loss_sensitivity": 0.01,
    "base_carbon_intensity_grid": 0.55,
    "max_carbon_reduction_from_renewables": 0.30,
    "hours_per_day": getattr(Config, 'hours_per_day', 24),
    "seed": 42,
    "chunk_size": 1000,
    "use_chunked_writing": True,
    "pmu_coverage": getattr(Config, 'pmu_coverage', 0.3)
}

def simulate_time_series(net: pp.pandapowerNet, config: dict, output_dir: str = None, 
                         case_name: str = None, renewable_fraction: float = None, 
                         timestamp: str = None) -> dict:
    """
    Runs the main time-series power flow simulation with convergence tracking.
    """
    num_buses = len(net.bus)
    time_steps = config['time_steps']
    
    # MEMORY OPTIMIZATION: Chunked writing mode
    use_chunked_writing = config.get('use_chunked_writing', True)
    chunk_size = config.get('chunk_size', 1000)
    
    chunked_mode = (use_chunked_writing and output_dir is not None and 
                    case_name is not None and renewable_fraction is not None and 
                    timestamp is not None)
    
    if chunked_mode:
        # print(f"[Memory Optimization] Using chunked writing (chunk_size={chunk_size})") # UI Cleanup
        temp_dir = tempfile.mkdtemp(prefix='gen_data_chunks_')
        
        feature_file = os.path.join(temp_dir, 'features_temp.npy')
        target_file = os.path.join(temp_dir, 'targets_temp.npy')
        bus_types_file = os.path.join(temp_dir, 'bus_types_temp.npy')
        topology_ids_file = os.path.join(temp_dir, 'topology_ids_temp.npy')
        energy_coeffs_file = os.path.join(temp_dir, 'energy_coeffs_temp.npy')
        carbon_coeffs_file = os.path.join(temp_dir, 'carbon_coeffs_temp.npy')
        
        feature_matrix = np.memmap(feature_file, mode='w+', dtype=np.float32, shape=(time_steps, num_buses, 10))
        target_matrix = np.memmap(target_file, mode='w+', dtype=np.float32, shape=(time_steps, num_buses, 2))
        bus_types_array = np.memmap(bus_types_file, mode='w+', dtype=np.int32, shape=(time_steps, num_buses))
        topology_ids = np.memmap(topology_ids_file, mode='w+', dtype=np.int32, shape=(time_steps,))
        time_energy_coeffs = np.memmap(energy_coeffs_file, mode='w+', dtype=np.float32, shape=(time_steps,))
        time_carbon_coeffs = np.memmap(carbon_coeffs_file, mode='w+', dtype=np.float32, shape=(time_steps,))
        
        chunks_written = 0
    else:
        feature_matrix = np.zeros((time_steps, num_buses, 10), dtype=np.float32)
        target_matrix = np.zeros((time_steps, num_buses, 2), dtype=np.float32)
        bus_types_array = np.zeros((time_steps, num_buses), dtype=np.int32)
        topology_ids = np.zeros(time_steps, dtype=np.int32)
        time_energy_coeffs = np.zeros(time_steps, dtype=np.float32)
        time_carbon_coeffs = np.zeros(time_steps, dtype=np.float32)
        temp_dir = None
    
    base_adjacency_matrix = calculate_adjacency_matrix(net)
    
    ybus_base = None
    contingency_timesteps = []
    contingency_ybus_list = []
    
    convergence_stats = {
        'total_timesteps': time_steps,
        'successful': 0,
        'failed': 0,
        'failed_no_contingency': [],
        'failed_with_contingency': [],
        'contingency_line_details': {},
        'successful_timesteps': [],
        'resolution_methods': {
            'strict_normal': 0,
            'strict_contingency': 0,
            'relaxed_contingency': 0,
            'restored_line': 0,
        },
        'timestep_resolution': {},
        'contingencies_attempted': 0,
        'contingencies_successful': 0,
        'contingencies_failed': 0,
        'contingencies_resolved_strict': 0,
        'contingencies_resolved_relaxed': 0,
        'contingencies_restored': 0,
        'critical_lines': {},
        'validation_stats': {
            'consecutive_failures': 0,
            'max_consecutive_failures': 0,
            'curtailment_attempts': 0,
            'curtailment_events': 0,
            'curtailment_successful': 0,
            'generator_trips': 0,
            'hard_resets': 0,
            'pre_validation_failed': 0,
            'post_validation_failed': 0,
            'garbage_discarded': 0,
            'voltage_violations': 0,
            'angle_violations': 0,
            'line_loading_violations': 0,
            'slack_power_violations': 0,
            'generator_capacity_violations': 0,
            'inverter_capability_violations': 0,
            'valid_stressed_states': 0,
        },
    }
    
    base_load_p, base_load_q = net.load.p_mw.copy(), net.load.q_mvar.copy()
    total_system_load_mw = base_load_p.sum()
    # print(f"Total system load: {total_system_load_mw:.2f} MW") # UI Cleanup
    
    solar_gens = net.sgen[net.sgen.type == 'solar'] if 'type' in net.sgen.columns else pd.DataFrame()
    wind_gens = net.sgen[net.sgen.type == 'wind'] if 'type' in net.sgen.columns else pd.DataFrame()
    
    num_solar = len(solar_gens)
    num_wind = len(wind_gens)
    num_total_renewable = num_solar + num_wind
    
    if num_total_renewable > 0:
        # Scale capacity to achieve target penetration accounting for capacity factor (~0.3)
        # Target Generation = Load * Fraction
        # Required Capacity = Target Generation / Capacity Factor
        # Required Capacity = Load * Fraction / 0.3
        assumed_capacity_factor = 0.3
        # Ensure we don't divide by zero if fraction is 0 (though num_total_renewable > 0 implies some renewables)
        target_fraction = renewable_fraction if renewable_fraction is not None else 0.0
        max_total_renewable_mw = (total_system_load_mw * target_fraction) / assumed_capacity_factor
        solar_fraction = num_solar / num_total_renewable if num_total_renewable > 0 else 0.5
        wind_fraction = num_wind / num_total_renewable if num_total_renewable > 0 else 0.5
        
        max_individual_solar_mw = (max_total_renewable_mw * solar_fraction) / num_solar if num_solar > 0 else 0
        max_individual_wind_mw = (max_total_renewable_mw * wind_fraction) / num_wind if num_wind > 0 else 0
        
        # print(f"  Renewable generators: {num_solar} solar + {num_wind} wind") # UI Cleanup
        # print(f"  Max individual capacity: Solar={max_individual_solar_mw:.3f} MW, Wind={max_individual_wind_mw:.3f} MW") # UI Cleanup
    else:
        max_individual_solar_mw = 0
        max_individual_wind_mw = 0
        max_total_renewable_mw = 1.0
        # print("  No renewable generators configured") # UI Cleanup
    
    if num_total_renewable > 0 and 'type' in net.sgen.columns:
        for i, sgen in net.sgen.iterrows():
            if sgen.type == 'solar' and max_individual_solar_mw > 0:
                net.sgen.at[i, 'sn_mva'] = max_individual_solar_mw * 1.1
            elif sgen.type == 'wind' and max_individual_wind_mw > 0:
                net.sgen.at[i, 'sn_mva'] = max_individual_wind_mw * 1.1
    
    # Always generate weather-driven renewable variability
    weather_sequence = simulate_weather_sequence(
        timesteps=time_steps,
        hours_per_day=config.get('hours_per_day', 24),
        seed=config.get('seed', None)
    )
        
    dropped_line_idx = None
    has_contingency = False
    consecutive_failures = 0
    max_consecutive_failures = 0
    detailed_metrics = []
    
    # Progress Bar Logic
    disable_pbar = False
    try:
        if 'args' in globals():
            disable_pbar = getattr(args, 'no_progress_bar', False)
    except Exception:
        pass
    
    # Also check config dict (for in-memory tests)
    if not disable_pbar and config.get('no_progress_bar', False):
        disable_pbar = True
        
    iterator = range(time_steps)
    if not disable_pbar:
        iterator = tqdm(range(time_steps), desc=f"Generating {case_name} ({renewable_fraction*100:.0f}%)", unit="step")
    
    for t in iterator:
        restore_contingency(net, dropped_line_idx)
        dropped_line_idx = None
        has_contingency = False
        
        if np.random.random() < config['contingency_rate']:
            dropped_line_idx = apply_n1_contingency(net)
            has_contingency = (dropped_line_idx is not None)
            if has_contingency:
                convergence_stats['contingencies_attempted'] += 1

        if has_contingency and dropped_line_idx is not None:
            topology_ids[t] = dropped_line_idx + 1
        else:
            topology_ids[t] = 0


        # Time-series simulation (always enabled)
        current_hour = t % config['hours_per_day']
        current_day = t // config['hours_per_day']
        
        load_multiplier = get_daily_load_profile(current_hour)
        net.load.p_mw = base_load_p * load_multiplier
        net.load.q_mvar = base_load_q * load_multiplier
        
        solar_weather = None
        wind_weather = None
        if weather_sequence is not None:
            solar_weather, wind_weather = weather_sequence[t]
        
        current_total_renewable_p_mw = 0
        if 'type' in net.sgen.columns and not net.sgen.empty:
            for i, sgen in net.sgen.iterrows():
                p_gen = 0
                if sgen.type == 'solar':
                    solar_profile = get_solar_generation_profile(
                        current_hour, 
                        day_of_year=180 + current_day % 180,
                        weather_state=solar_weather
                    )
                    p_gen = solar_profile * max_individual_solar_mw
                elif sgen.type == 'wind':
                    wind_profile = get_wind_generation_profile(
                        current_hour, 
                        day=current_day,
                        weather_state=wind_weather
                    )
                    p_gen = wind_profile * max_individual_wind_mw
                
                net.sgen.at[i, 'p_mw'] = p_gen
                q_gen = calculate_renewable_reactive_power(p_gen, sgen.bus, net, t > 0)
                net.sgen.at[i, 'q_mvar'] = q_gen
                current_total_renewable_p_mw += p_gen
        
        convergence_successful = False
        resolution_method = None
        violation_flags = {}
        
        if consecutive_failures >= 3:
            print(f"  [Hard Reset] {consecutive_failures} consecutive failures detected - triggering hard reset")
            base_renewable_p_mw_for_reset = {}
            if not net.sgen.empty:
                for i, sgen in net.sgen.iterrows():
                    # Time-series simulation (always enabled)
                    current_hour = t % config['hours_per_day']
                    current_day = t // config['hours_per_day']
                    if sgen.type == 'solar':
                        solar_profile = get_solar_generation_profile(
                            current_hour, 
                            day_of_year=180 + current_day % 180,
                            weather_state=weather_sequence[t][0] if weather_sequence else None
                        )
                        base_renewable_p_mw_for_reset[i] = solar_profile * max_individual_solar_mw
                    elif sgen.type == 'wind':
                        wind_profile = get_wind_generation_profile(
                            current_hour, 
                            day=current_day,
                            weather_state=weather_sequence[t][1] if weather_sequence else None
                        )
                        base_renewable_p_mw_for_reset[i] = wind_profile * max_individual_wind_mw
                    else:
                        base_renewable_p_mw_for_reset[i] = net.sgen.at[i, 'p_mw']
            
            reset_success, new_dropped_line_idx = hard_reset_system(
                net, base_load_p, base_load_q, base_renewable_p_mw_for_reset, 
                convergence_stats, dropped_line_idx, case_name
            )
            
            if reset_success:
                dropped_line_idx = new_dropped_line_idx
                has_contingency = (dropped_line_idx is not None)
                consecutive_failures = 0
                convergence_successful = True
                resolution_method = 'hard_reset'
                _, _, violation_flags = validate_power_flow_outputs(net, convergence_stats, case_name)
                convergence_stats['successful'] += 1
                convergence_stats['successful_timesteps'].append(t)
                convergence_stats['resolution_methods']['hard_reset'] = convergence_stats['resolution_methods'].get('hard_reset', 0) + 1
            else:
                consecutive_failures += 1
                max_consecutive_failures = max(max_consecutive_failures, consecutive_failures)
                convergence_stats['validation_stats']['consecutive_failures'] = consecutive_failures
                convergence_stats['validation_stats']['max_consecutive_failures'] = max_consecutive_failures
                convergence_stats['failed'] += 1
                if has_contingency:
                    convergence_stats['failed_with_contingency'].append(t)
                else:
                    convergence_stats['failed_no_contingency'].append(t)
                raise RuntimeError(f"Hard reset failed - timestep {t} cannot be recovered")
        
        if not convergence_successful:
            resolution_method = None
            violation_flags = {}
            
            input_valid, input_reason = validate_power_flow_inputs(net)
            if not input_valid:
                convergence_stats['validation_stats']['pre_validation_failed'] += 1
                print(f"  WARNING: Timestep {t} failed pre-validation: {input_reason}, attempting generator trip")
                
                if trip_renewable_generators(net, convergence_stats, case_name):
                    convergence_successful = True
                    resolution_method = 'trip_generators'
                    _, _, violation_flags = validate_power_flow_outputs(net, convergence_stats, case_name)
                else:
                    consecutive_failures += 1
                    max_consecutive_failures = max(max_consecutive_failures, consecutive_failures)
                    convergence_stats['validation_stats']['consecutive_failures'] = consecutive_failures
                    convergence_stats['validation_stats']['max_consecutive_failures'] = max_consecutive_failures
                    convergence_stats['failed'] += 1
                    convergence_stats['failed_no_contingency'].append(t)
                    raise RuntimeError(f"Timestep {t} failed even after generator trip - no valid power flow solution")
        
        base_renewable_p_mw = {}
        if not net.sgen.empty and not convergence_successful:
            for i, sgen in net.sgen.iterrows():
                base_renewable_p_mw[i] = net.sgen.at[i, 'p_mw']
        
        if not convergence_successful:
            curtailment_success, curtailment_scaling, violation_flags = apply_curtailment_with_retry(
                net, base_renewable_p_mw, max_attempts=10, 
                convergence_stats=convergence_stats, has_contingency=has_contingency,
                case_name=case_name
            )
            
            if curtailment_success:
                convergence_successful = True
                convergence_stats['successful'] += 1
                convergence_stats['successful_timesteps'].append(t)
                consecutive_failures = 0
                
                if has_contingency:
                    resolution_method = 'curtailment_contingency' if curtailment_scaling < 1.0 else 'strict_contingency'
                    convergence_stats['resolution_methods']['strict_contingency'] += 1
                    convergence_stats['contingencies_successful'] += 1
                    convergence_stats['contingencies_resolved_strict'] += 1
                else:
                    resolution_method = 'curtailment_normal' if curtailment_scaling < 1.0 else 'strict_normal'
                    convergence_stats['resolution_methods']['strict_normal'] += 1
            else:
                if has_contingency and dropped_line_idx is not None:
                    line_key = f"line_{dropped_line_idx}"
                    if line_key not in convergence_stats['critical_lines']:
                        convergence_stats['critical_lines'][line_key] = {
                            'line_id': int(dropped_line_idx),
                            'failure_count': 0,
                            'resolution_methods': {'relaxed_curtailment': 0, 'restored_curtailment': 0, 'trip': 0}
                        }
                    convergence_stats['critical_lines'][line_key]['failure_count'] += 1
                    
                    for i, base_p_mw in base_renewable_p_mw.items():
                        net.sgen.at[i, 'p_mw'] = base_p_mw
                    
                    relaxed_success, relaxed_scaling, relaxed_violations = apply_curtailment_with_retry(
                        net, base_renewable_p_mw, max_attempts=10,
                        convergence_stats=convergence_stats, has_contingency=True
                    )
                    
                    if relaxed_success:
                        try:
                            with SuppressPrints():
                                pp.runpp(net, numba=True, enforce_q_lims=False, algorithm='nr', 
                                        tolerance_mva=1e-6, max_iteration=20)
                            relaxed_valid, _, relaxed_violations = validate_power_flow_outputs(net, convergence_stats)
                            if relaxed_valid:
                                convergence_successful = True
                                violation_flags = relaxed_violations
                                convergence_stats['successful'] += 1
                                convergence_stats['successful_timesteps'].append(t)
                                consecutive_failures = 0
                                
                                resolution_method = 'relaxed_curtailment_contingency'
                                convergence_stats['resolution_methods']['relaxed_contingency'] += 1
                                convergence_stats['contingencies_successful'] += 1
                                convergence_stats['contingencies_resolved_relaxed'] += 1
                                convergence_stats['critical_lines'][line_key]['resolution_methods']['relaxed_curtailment'] += 1
                            else:
                                relaxed_success = False
                        except pp.LoadflowNotConverged:
                            relaxed_success = False
                    
                    if not relaxed_success:
                        restore_contingency(net, dropped_line_idx)
                        dropped_line_idx = None
                        has_contingency = False
                        convergence_stats['critical_lines'][line_key]['resolution_methods']['restored_curtailment'] += 1
                        
                        for i, base_p_mw in base_renewable_p_mw.items():
                            net.sgen.at[i, 'p_mw'] = base_p_mw
                        
                        restored_success, restored_scaling, restored_violations = apply_curtailment_with_retry(
                            net, base_renewable_p_mw, max_attempts=10,
                            convergence_stats=convergence_stats, has_contingency=False
                        )
                        
                        if restored_success:
                            convergence_successful = True
                            violation_flags = restored_violations
                            convergence_stats['successful'] += 1
                            convergence_stats['successful_timesteps'].append(t)
                            consecutive_failures = 0
                            
                            resolution_method = 'restored_curtailment'
                            convergence_stats['resolution_methods']['restored_line'] += 1
                            convergence_stats['contingencies_restored'] += 1
                        else:
                            if trip_renewable_generators(net, convergence_stats):
                                convergence_successful = True
                                _, _, violation_flags = validate_power_flow_outputs(net, convergence_stats)
                                convergence_stats['successful'] += 1
                                convergence_stats['successful_timesteps'].append(t)
                                consecutive_failures = 0
                                resolution_method = 'trip_after_restore'
                                convergence_stats['critical_lines'][line_key]['resolution_methods']['trip'] += 1
                            else:
                                consecutive_failures += 1
                                max_consecutive_failures = max(max_consecutive_failures, consecutive_failures)
                                convergence_stats['validation_stats']['consecutive_failures'] = consecutive_failures
                                convergence_stats['validation_stats']['max_consecutive_failures'] = max_consecutive_failures
                                convergence_stats['failed'] += 1
                                convergence_stats['failed_with_contingency'].append(t)
                                convergence_stats['contingencies_failed'] += 1
                                resolution_method = 'failed_completely'
                                print(f"  ERROR: Timestep {t} failed completely after all strategies")
                                continue
                else:
                    if trip_renewable_generators(net, convergence_stats):
                        convergence_successful = True
                        _, _, violation_flags = validate_power_flow_outputs(net, convergence_stats)
                        convergence_stats['successful'] += 1
                        convergence_stats['successful_timesteps'].append(t)
                        consecutive_failures = 0
                        resolution_method = 'trip_normal'
                    else:
                        consecutive_failures += 1
                        max_consecutive_failures = max(max_consecutive_failures, consecutive_failures)
                        convergence_stats['validation_stats']['consecutive_failures'] = consecutive_failures
                        convergence_stats['validation_stats']['max_consecutive_failures'] = max_consecutive_failures
                        convergence_stats['failed'] += 1
                        convergence_stats['failed_no_contingency'].append(t)
                        resolution_method = 'failed'
                        print(f"  ERROR: Timestep {t} failed completely (no contingency)")
                        continue
        
        if not convergence_successful:
            consecutive_failures += 1
            max_consecutive_failures = max(max_consecutive_failures, consecutive_failures)
            convergence_stats['validation_stats']['consecutive_failures'] = consecutive_failures
            convergence_stats['validation_stats']['max_consecutive_failures'] = max_consecutive_failures
            convergence_stats['failed'] += 1
            if has_contingency:
                convergence_stats['failed_with_contingency'].append(t)
            else:
                convergence_stats['failed_no_contingency'].append(t)
            print(f"  ERROR: Timestep {t} failed - no successful path found")
            continue
        
        if resolution_method:
            convergence_stats['timestep_resolution'][str(t)] = resolution_method
        
        if violation_flags:
            if violation_flags.get('voltage_violation', False):
                convergence_stats['validation_stats']['voltage_violations'] += 1
            if violation_flags.get('angle_violation', False):
                convergence_stats['validation_stats']['angle_violations'] += 1
            if violation_flags.get('line_loading_violation', False):
                convergence_stats['validation_stats']['line_loading_violations'] += 1
            if violation_flags.get('slack_power_violation', False):
                convergence_stats['validation_stats']['slack_power_violations'] += 1
            if violation_flags.get('generator_capacity_violation', False):
                convergence_stats['validation_stats']['generator_capacity_violations'] += 1
            if violation_flags.get('inverter_capability_violation', False):
                convergence_stats['validation_stats']['inverter_capability_violations'] += 1
            
            if any([violation_flags.get('voltage_violation', False), 
                   violation_flags.get('angle_violation', False),
                   violation_flags.get('line_loading_violation', False), 
                   violation_flags.get('slack_power_violation', False)]):
                convergence_stats['validation_stats']['valid_stressed_states'] += 1
        
        vm_pu = net.res_bus.vm_pu.values
        va_rad = np.deg2rad(net.res_bus.va_degree.values)
        
        load_p_by_bus = net.res_load.groupby(net.load.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)
        load_q_by_bus = net.res_load.groupby(net.load.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0)
        p_load = load_p_by_bus.values
        q_load = load_q_by_bus.values

        ext_grid_p_by_bus = net.res_ext_grid.groupby(net.ext_grid.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)
        ext_grid_q_by_bus = net.res_ext_grid.groupby(net.ext_grid.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0)
        
        gen_p_by_bus = net.res_gen.groupby(net.gen.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)
        gen_q_by_bus = net.res_gen.groupby(net.gen.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0)

        sgen_p_by_bus = net.res_sgen.groupby(net.sgen.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)
        sgen_q_by_bus = net.res_sgen.groupby(net.sgen.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0)

        current_ybus = calculate_ybus_from_net(net)
        
        if ybus_base is None:
            ybus_base = current_ybus.copy()
        elif has_contingency:
            contingency_timesteps.append(t)
            contingency_ybus_list.append(current_ybus.copy())
        
        detailed_metrics.append({
            'timestep': t,
            'p_load_mw': p_load.sum(),
            'q_load_mvar': q_load.sum(),
            'p_gen_mw': (ext_grid_p_by_bus + gen_p_by_bus).sum(),
            'q_gen_mvar': (ext_grid_q_by_bus + gen_q_by_bus).sum(),
            'p_ren_mw': sgen_p_by_bus.sum(),
            'q_ren_mvar': sgen_q_by_bus.sum(),
            'v_min_pu': vm_pu.min(),
            'v_max_pu': vm_pu.max(),
            'v_mean_pu': vm_pu.mean(),
            'v_std_pu': vm_pu.std(),
            'angle_min_deg': np.rad2deg(va_rad).min(),
            'angle_max_deg': np.rad2deg(va_rad).max(),
            'max_line_loading_pct': net.res_line.loading_percent.max(),
            'converged': 1 if convergence_successful else 0,
            'resolution_method': resolution_method or 'Normal',
            'violations': ";".join([k for k, v in violation_flags.items() if v]) if violation_flags else "None"
        })
        
        renewable_util_frac = current_total_renewable_p_mw / max_total_renewable_mw
        time_carbon_coeffs[t] = config['base_carbon_intensity_grid'] - (renewable_util_frac * config['max_carbon_reduction_from_renewables'])
        time_energy_coeffs[t] = config['max_energy_utilization_coeff'] - (net.res_line.pl_mw.sum() * config['loss_sensitivity'])
        
        bus_types = identify_bus_types(net)
        bus_types_array[t] = bus_types
        
        opf_targets = create_opf_targets(net, bus_types)
        target_matrix[t] = opf_targets
        
        positive_noise_vm = np.abs(np.random.normal(0, config['voltage_error_std'], num_buses))
        positive_noise_angle = np.abs(np.random.normal(0, config['angle_error_std'], num_buses))
        positive_noise_power = np.abs(np.random.normal(0, config['power_error_std'], num_buses))
        
        meas_pl = p_load * (1 + positive_noise_power)
        meas_ql = q_load * (1 + positive_noise_power)
        meas_p_ext = ext_grid_p_by_bus.values * (1 + positive_noise_power)
        meas_q_ext = ext_grid_q_by_bus.values * (1 + positive_noise_power)
        meas_p_conv = gen_p_by_bus.values * (1 + positive_noise_power)
        meas_q_conv = gen_q_by_bus.values * (1 + positive_noise_power)
        meas_p_ren = sgen_p_by_bus.values * (1 + positive_noise_power)
        meas_q_ren = sgen_q_by_bus.values * (1 + positive_noise_power)
        
        gen_indices = np.where(bus_types > 0)[0]
        num_pmu_buses = max(1, int(num_buses * config.get('pmu_coverage', 0.3)))
        pmu_indices = set(gen_indices)
        remaining_slots = num_pmu_buses - len(pmu_indices)
        if remaining_slots > 0:
            pq_indices = np.where(bus_types == 0)[0]
            if len(pq_indices) > 0:
                chosen_pq = np.random.choice(pq_indices, size=min(len(pq_indices), remaining_slots), replace=False)
                pmu_indices.update(chosen_pq)
        
        pmu_buses = np.array(list(pmu_indices))
        meas_vm = np.full(num_buses, np.nan)
        meas_va = np.full(num_buses, np.nan)
        meas_vm[pmu_buses] = vm_pu[pmu_buses] * (1 + positive_noise_vm[pmu_buses])
        meas_va[pmu_buses] = va_rad[pmu_buses] * (1 + positive_noise_angle[pmu_buses])
        meas_vm = np.nan_to_num(meas_vm, nan=0.0)
        meas_va = np.nan_to_num(meas_va, nan=0.0)
        
        feature_matrix[t] = np.stack([
            meas_pl, meas_ql,
            meas_p_ext, meas_q_ext,
            meas_p_conv, meas_q_conv,
            meas_p_ren, meas_q_ren,
            meas_vm, meas_va
        ], axis=1)
        
        if chunked_mode and (t + 1) % chunk_size == 0:
            feature_matrix.flush()
            target_matrix.flush()
            bus_types_array.flush()
            topology_ids.flush()
            time_energy_coeffs.flush()
            time_carbon_coeffs.flush()
            chunks_written += 1
            # print(f"  [Chunked Writing] Flushed chunk {chunks_written} ({(t + 1)}/{time_steps} timesteps written)") # UI Cleanup
    
    if chunked_mode:
        feature_matrix.flush()
        target_matrix.flush()
        bus_types_array.flush()
        topology_ids.flush()
        time_energy_coeffs.flush()
        time_carbon_coeffs.flush()
        # print(f"  [Chunked Writing] All data flushed to disk ({time_steps}/{time_steps} timesteps)") # UI Cleanup
    
    convergence_stats['success_rate'] = (convergence_stats['successful'] / time_steps * 100) if time_steps > 0 else 0
    
    if ybus_base is None:
        raise RuntimeError(
            f"CRITICAL ERROR: ybus_base is None after {convergence_stats['successful']} successful power flows! "
        )
    else:
        # print(f"  [DEBUG] *** YBUS_BASE VERIFIED *** shape={ybus_base.shape}, diagonal[0]={ybus_base[0,0]:.6f}")
        convergence_stats['ybus_fallback_used'] = False
    
    ybus_data = {
        "base": ybus_base,
        "contingency_timesteps": np.array(contingency_timesteps, dtype=np.int32),
        "contingency_matrices": np.array(contingency_ybus_list) if contingency_ybus_list else np.array([]).reshape(0, num_buses, num_buses).astype(np.complex128)
    }
    
    if chunked_mode:
        # print(f"  [Chunked Writing] Copying memmap arrays to regular arrays for cleanup...")
        features_return = np.array(feature_matrix)
        targets_return = np.array(target_matrix)
        bus_types_return = np.array(bus_types_array)
        topology_ids_return = np.array(topology_ids)
        energy_coeffs_return = np.array(time_energy_coeffs)
        carbon_coeffs_return = np.array(time_carbon_coeffs)
        
        del feature_matrix, target_matrix, bus_types_array, topology_ids
        del time_energy_coeffs, time_carbon_coeffs
        gc.collect()
        
        convergence_stats['_temp_dir'] = temp_dir
        convergence_stats['_temp_files'] = {
            'features': feature_file,
            'targets': target_file,
            'bus_types': bus_types_file,
            'topology_ids': topology_ids_file,
            'energy_coeffs': energy_coeffs_file,
            'carbon_coeffs': carbon_coeffs_file
        }
    else:
        features_return = feature_matrix
        targets_return = target_matrix
        bus_types_return = bus_types_array
        topology_ids_return = topology_ids
        energy_coeffs_return = time_energy_coeffs
        carbon_coeffs_return = time_carbon_coeffs
    
    # Save detailed metrics CSV (skip if output_dir is None for in-memory tests)
    if output_dir is not None:
        try:
            df_metrics = pd.DataFrame(detailed_metrics)
            csv_filename = f"{case_name}_detailed_metrics_frac{renewable_fraction:.1f}_{timestamp}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            df_metrics.to_csv(csv_path, index=False)
            # print(f"  [CSV] Saved detailed metrics to {csv_filename}")
        except Exception as e:
            raise RuntimeError(f"Could not save detailed metrics CSV: {e}")

    return {
        "features": features_return,
        "targets": targets_return,
        "bus_types": bus_types_return,
        "base_adjacency": base_adjacency_matrix,
        "topology_ids": topology_ids_return,
        "ybus_data": ybus_data,
        "time_energy_coeffs": energy_coeffs_return, 
        "time_carbon_coeffs": carbon_coeffs_return,
        "convergence_stats": convergence_stats
    }

def save_data(data_dict: dict, case_name: str, renewable_fraction: float, output_dir: str, timestamp: str = None):
    """
    Saves generated data arrays.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    convergence_stats = data_dict.get('convergence_stats', {})
    temp_files = convergence_stats.get('_temp_files', None)
    use_temp_files = (temp_files is not None and os.path.exists(temp_files.get('features', '')))
    
    if use_temp_files:
        # print(f"[Memory Optimization] Copying chunked data files to final location...")
        
        file_mappings = {
            'features': f"{case_name}_features_frac{renewable_fraction:.1f}_{timestamp}.npy",
            'targets': f"{case_name}_targets_frac{renewable_fraction:.1f}_{timestamp}.npy",
            'bus_types': f"{case_name}_bus_types_frac{renewable_fraction:.1f}_{timestamp}.npy",
            'topology_ids': f"{case_name}_topology_ids_frac{renewable_fraction:.1f}_{timestamp}.npy",
            'energy_coeffs': f"{case_name}_time_energy_coeffs_frac{renewable_fraction:.1f}_{timestamp}.txt",
            'carbon_coeffs': f"{case_name}_time_carbon_coeffs_frac{renewable_fraction:.1f}_{timestamp}.txt"
        }
        
        for temp_key, final_filename in file_mappings.items():
                final_path = os.path.join(output_dir, final_filename)
                
                if temp_key in ['energy_coeffs', 'carbon_coeffs']:
                    coeff_key = 'time_energy_coeffs' if 'energy' in temp_key else 'time_carbon_coeffs'
                    if coeff_key in data_dict:
                        data = data_dict[coeff_key]
                        if isinstance(data, np.memmap):
                            data = np.array(data)
                        np.savetxt(final_path, data)
                        # print(f"  Copied {temp_key} -> {final_filename} (converted to .txt)")
                else:
                    data_key_map = {
                        'features': 'features',
                        'targets': 'targets',
                        'bus_types': 'bus_types',
                        'topology_ids': 'topology_ids'
                    }
                    
                    data_key = data_key_map.get(temp_key)
                    if data_key and data_key in data_dict:
                        data = data_dict[data_key]
                        if isinstance(data, np.memmap):
                            data = np.array(data)
                        
                        if temp_key == 'features':
                            np.save(final_path, np.array(data, dtype=np.float32), allow_pickle=False)
                        elif temp_key == 'targets':
                            np.save(final_path, np.array(data, dtype=np.float32), allow_pickle=False)
                        elif temp_key == 'bus_types':
                            np.save(final_path, np.array(data, dtype=np.int32), allow_pickle=False)
                        elif temp_key == 'topology_ids':
                            np.save(final_path, np.array(data, dtype=np.int32), allow_pickle=False)
                        else:
                            np.save(final_path, np.array(data), allow_pickle=False)
                        # print(f"  Copied {temp_key} -> {final_filename}")
        
        # print(f"[Memory Optimization] All chunked data files copied successfully")
    
    for key, data in data_dict.items():
        if use_temp_files and key in ['features', 'targets', 'bus_types', 'topology_ids', 
                                      'time_energy_coeffs', 'time_carbon_coeffs']:
            continue
        if key == "ybus_data":
            for sub_key, sub_data in data.items():
                sub_filename = f"{case_name}_ybus_{sub_key}_frac{renewable_fraction:.1f}_{timestamp}.npy"
                filepath = os.path.join(output_dir, sub_filename)
                # print(f"Saving Ybus component '{sub_key}' to '{filepath}'...")
                np.save(filepath, sub_data, allow_pickle=False)
            continue
        
        if key == "convergence_stats":
            stats_to_save = {k: v for k, v in data.items() if not k.startswith('_temp')}
            audit_data = transform_convergence_to_audit(
                stats_to_save, case_name, renewable_fraction, timestamp
            )
            stats_filename = f"{case_name}_data_quality_audit_frac{renewable_fraction:.1f}_{timestamp}.json"
            filepath = os.path.join(output_dir, stats_filename)
            # print(f"Saving data quality audit to '{filepath}'...")
            with open(filepath, 'w') as f:
                json.dump(audit_data, f, indent=2)
            continue
        
        base_filename = f"{case_name}_{key}_frac{renewable_fraction:.1f}_{timestamp}"
        
        if "coeffs" in key:
            filename = os.path.join(output_dir, base_filename + ".txt")
            # print(f"Saving coefficient data to '{filename}'...")
            if isinstance(data, np.memmap):
                np.savetxt(filename, np.array(data))
            else:
                np.savetxt(filename, data)
        elif key == "topology_ids":
            filename = os.path.join(output_dir, base_filename + ".npy")
            # print(f"Saving topology IDs to '{filename}'...")
            if isinstance(data, np.memmap):
                np.save(filename, np.array(data), allow_pickle=False)
            else:
                np.save(filename, data, allow_pickle=False)
        elif key == "base_adjacency":
            if isinstance(data, np.memmap):
                adj_data = np.array(data)
            else:
                adj_data = data
            adj_sparse = sparse.coo_matrix(adj_data)
            edge_index = np.array([adj_sparse.row, adj_sparse.col])
            filename = os.path.join(output_dir, base_filename + ".npy")
            # print(f"Saving base adjacency matrix to '{filename}'...")
            np.save(filename, np.array([edge_index], dtype=object), allow_pickle=True)
        else:
            filename = os.path.join(output_dir, base_filename + ".npy")
            # print(f"Saving array data to '{filename}'...")
            if isinstance(data, np.memmap):
                np.save(filename, np.array(data), allow_pickle=True)
            else:
                np.save(filename, data, allow_pickle=True)

if __name__ == "__main__":
    
    if CONFIG["random_seed"] is not None:
        # print(f"\nSetting random seed: {CONFIG['random_seed']} (for reproducibility)")
        np.random.seed(CONFIG["random_seed"])
        random.seed(CONFIG["random_seed"])
    else:
        raise ValueError("No random seed set - results will not be reproducible! Set CONFIG['seed'] or use --seed argument")
    
    data_mode = 'train'
    timesteps = None
    cases_to_run = None

    args = sys.argv[1:]
    if len(args) > 0 and not args[0].startswith('--'):
        data_mode = args[0].lower()
        if data_mode not in ['train', 'test']:
            raise ValueError(f"Invalid data_mode '{data_mode}'. Use 'train' or 'test'")
        args = args[1:]
    
    if len(args) > 0 and not args[0].startswith('--'):
        try:
            timesteps = int(args[0])
            args = args[1:]
        except ValueError:
            pass
            
    i = 0
    while i < len(args):
        if args[i] == '--cases':
            if i + 1 < len(args):
                cases_str = args[i+1]
                if cases_str.lower() == 'all':
                     cases_to_run = ["case33", "case57", "case118"]
                else:
                    cases_to_run = [c.strip() for c in cases_str.split(',')]
                i += 1
            else:
                raise ValueError("--cases requires a value (e.g. case33,case57 or all)")
        i += 1

    try:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(parent_dir, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if 'system' in yaml_config and 'test_cases' in yaml_config['system']:
                    CONFIG['test_cases'] = yaml_config['system']['test_cases']
                    # print(f"Loaded test_cases from config.yaml: {CONFIG['test_cases']}")
    except Exception as e:
        raise RuntimeError(f"Could not load config.yaml: {e}. Configuration file is required.")

    if cases_to_run is None:
        cases_to_run = CONFIG["test_cases"]
        
    # print(f"Cases to run: {cases_to_run}")
    
    if timesteps is None:
        if data_mode == 'train':
            timesteps = CONFIG["time_steps"]
        else:
            timesteps = 120
            
    # print(f"Generating {data_mode} data for {timesteps} timesteps...")
    
    CONFIG['time_steps'] = timesteps
    
    generation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # print(f"\nStarting data generation [{data_mode.upper()} MODE - {timesteps} timesteps]")
    # print(f"Timestamp: {generation_timestamp}")
    
    # Auto-cleanup: Delete old data in the target directory before generating new data
    output_path = CONFIG['output_dir']
    if os.path.exists(output_path):
        try:
            print(f"\n[Auto-Cleanup] Deleting old data in {output_path}...")
            shutil.rmtree(output_path)
            
            # Wait for directory to be fully deleted (Windows async deletion issue)
            max_wait = 10  # seconds
            wait_time = 0
            while os.path.exists(output_path) and wait_time < max_wait:
                time.sleep(0.1)
                wait_time += 0.1
            
            if os.path.exists(output_path):
                raise RuntimeError(f"Failed to delete directory after {max_wait}s: {output_path}")
            
            print(f"[Auto-Cleanup] Successfully cleaned {data_mode} data directory")
        except Exception as e:
            if "Failed to delete directory" in str(e):
                raise  # Re-raise deletion timeout errors
            # If deletion failed for any other reason, this is critical
            raise RuntimeError(f"Auto-cleanup failed: Could not delete old data: {e}")
    
    # Recreate the directory (now guaranteed to be deleted)
    os.makedirs(output_path, exist_ok=True)
    
    for case in cases_to_run:
        try:
            base_net = load_network(case)
            for frac in CONFIG["renewable_fractions_to_run"]:
                # print(f"\n{'='*60}\nProcessing {case} with {frac*100:.0f}% renewable fraction\n{'='*60}")
                
                net_for_run = copy.deepcopy(base_net)
                save_case_name = case.replace('bw', '')
                net_for_run.name = f"{save_case_name}_frac{frac:.1f}"
                
                net_with_renewables = configure_renewables(net_for_run, frac, CONFIG)
                
                # Use the configured output directory (respects CLI args and mode)
                output_path = CONFIG['output_dir']
                
                os.makedirs(output_path, exist_ok=True)
                
                generated_data = simulate_time_series(
                    net_with_renewables, CONFIG,
                    output_dir=output_path,
                    case_name=save_case_name,
                    renewable_fraction=frac,
                    timestamp=generation_timestamp
                )
                    
                save_data(generated_data, save_case_name, frac, output_path, generation_timestamp)
                
                if '_temp_dir' in generated_data.get('convergence_stats', {}):
                    temp_dir = generated_data['convergence_stats']['_temp_dir']
                    if os.path.exists(temp_dir):
                        max_retries = 3
                        for retry in range(max_retries):
                            try:
                                shutil.rmtree(temp_dir)
                                # print(f"  [Chunked Writing] Cleaned up temporary files")
                                break
                            except PermissionError as e:
                                if retry < max_retries - 1:
                                    time.sleep(0.1)
                                    gc.collect()
                                else:
                                    raise RuntimeError(f"Could not delete temp directory {temp_dir}: {e}")

        except DataGenerationError as e:
            print(f"\n{'='*80}")
            print(f"SEVERE ERROR: Data generation cannot continue!")
            print(f"{'='*80}")
            print(f"Error details:")
            traceback.print_exc()
            sys.exit(1)
        except Exception as e:
            print(f"\nWARNING: An error occurred while processing {case}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            traceback.print_exc()
            print(f"\nSkipping to the next test case.")
            continue
    
    try:
        # Use the configured output directory (respects CLI args and mode)
        output_path = CONFIG['output_dir']
        
        os.makedirs(output_path, exist_ok=True)
        
        # Always time-series mode
        metadata = {
            'generation_mode': 'time_series',
            'data_mode': data_mode,
            'timesteps': timesteps,
            'timestamp': generation_timestamp,
            'hours_per_day': CONFIG.get('hours_per_day', 24),
            'test_cases': cases_to_run,
            'renewable_fractions': CONFIG["renewable_fractions_to_run"],
            'generation_date': datetime.now().isoformat()
        }
        
        metadata_file = os.path.join(output_path, "data_generation_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
       # print(f"\n[Metadata] Saved generation metadata to: {metadata_file}")
    except Exception as e:
        raise RuntimeError(f"Could not save metadata file: {e}")
            
    print("\nAll data generation processes are complete.")
