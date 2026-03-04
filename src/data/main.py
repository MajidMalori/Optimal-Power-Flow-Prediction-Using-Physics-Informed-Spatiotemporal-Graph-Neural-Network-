import os
import sys
import traceback
import json
import argparse
import warnings
import copy
import random
import glob
import shutil
import tempfile
import numpy as np
import pandas as pd
import pandapower as pp
from tqdm import tqdm
from scipy import sparse
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir is 'src/data'
# root_dir is one more level up
root_dir = os.path.dirname(os.path.dirname(script_dir))
if root_dir not in sys.path: sys.path.insert(0, root_dir)

from src.data.profiles import (
    get_daily_load_profile, get_solar_generation_profile, 
    get_wind_generation_profile, simulate_weather_sequence,
    calculate_renewable_reactive_power
)
from src.data.topology import (
    load_network, configure_renewables, apply_n1_contingency, 
    restore_contingency, calculate_ybus_from_net, calculate_adjacency_matrix,
    identify_bus_types, create_opf_targets
)
from src.data.validation import (
    SuppressPrints, validate_power_flow_inputs, validate_power_flow_outputs,
    apply_curtailment_with_retry, hard_reset_system, trip_renewable_generators
)

warnings.filterwarnings('ignore', category=FutureWarning)

def load_data_config(config_path='configs/data_generation.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# CLI Parser
parser = argparse.ArgumentParser(description="Generate Spatio-Temporal Data")
parser.add_argument('--case', '--cases', '--buses', type=str, dest='buses', default=None, help="Cases (e.g., '33,57' or 'all')")
parser.add_argument('--timesteps', '--timestep', type=int, default=None, help="Number of timesteps")
args = parser.parse_args()

# Load YAML
gen_config = load_data_config()

# Professional Path Setup
data_mode = 'train' # Default
output_dir = gen_config.get('output_dir', 'data/01_raw')
reports_dir = gen_config.get('reports_dir', 'reports/figures/01_raw_data')

timesteps_to_use = args.timesteps if args.timesteps is not None else gen_config.get('time_steps', 10008)

CONFIG = {
    "random_seed": gen_config.get('random_seed', 42),
    "test_cases": gen_config.get('test_cases', ["case33", "case57", "case118"]),
    "time_steps": timesteps_to_use,
    "output_dir": output_dir,
    "reports_dir": reports_dir,
    "renewable_fractions_to_run": gen_config.get('renewable_fractions_to_run', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    "contingency_rate": gen_config.get('contingency_rate', 0.05),
    "voltage_error_std": gen_config.get('voltage_error_std', 0.005),
    "power_error_std": gen_config.get('power_error_std', 0.01),
    "angle_error_std": gen_config.get('angle_error_std', 0.02),
    "max_energy_utilization_coeff": gen_config.get('max_energy_utilization_coeff', 0.98),
    "loss_sensitivity": gen_config.get('loss_sensitivity', 0.01),
    "base_carbon_intensity_grid": gen_config.get('base_carbon_intensity_grid', 0.55),
    "max_carbon_reduction_from_renewables": gen_config.get('max_carbon_reduction_from_renewables', 0.30),
    "hours_per_day": gen_config.get('hours_per_day', 24),
    "seed": 42,
    "chunk_size": 1000,
    "use_chunked_writing": True,
    "pmu_coverage": gen_config.get('pmu_coverage', 0.3),
    "system_limits": gen_config.get('system_limits', {}),
    "solar_weather_weights": gen_config.get('solar_weather_weights', [0.3, 0.4, 0.25, 0.05]),
    "wind_weather_weights": gen_config.get('wind_weather_weights', [0.15, 0.45, 0.30, 0.10]),
}

def simulate_time_series(net: pp.pandapowerNet, config: dict, output_dir: str = None, 
                         case_name: str = None, renewable_fraction: float = None, pbar=None) -> dict:
    num_buses = len(net.bus)
    time_steps = config['time_steps']
    chunk_size = config.get('chunk_size', 1000)
    chunked_mode = (config.get('use_chunked_writing', True) and output_dir and 
                    case_name and renewable_fraction is not None)
    
    if chunked_mode:
        temp_dir = tempfile.mkdtemp(prefix='gen_data_chunks_')
        files = {k: os.path.join(temp_dir, f'{k}_temp.npy') for k in ['features', 'targets', 'bus_types', 'topology_ids', 'carbon_coeffs']}
        
        feature_matrix = np.memmap(files['features'], mode='w+', dtype=np.float32, shape=(time_steps, num_buses, 11))
        target_matrix = np.memmap(files['targets'], mode='w+', dtype=np.float32, shape=(time_steps, num_buses, 10))
        bus_types_array = np.memmap(files['bus_types'], mode='w+', dtype=np.int32, shape=(time_steps, num_buses))
        topology_ids = np.memmap(files['topology_ids'], mode='w+', dtype=np.int32, shape=(time_steps,))
        time_carbon_coeffs = np.memmap(files['carbon_coeffs'], mode='w+', dtype=np.float32, shape=(time_steps,))
    else:
        feature_matrix = np.zeros((time_steps, num_buses, 11), dtype=np.float32)
        target_matrix = np.zeros((time_steps, num_buses, 10), dtype=np.float32)
        bus_types_array = np.zeros((time_steps, num_buses), dtype=np.int32)
        topology_ids = np.zeros(time_steps, dtype=np.int32)
        time_carbon_coeffs = np.zeros(time_steps, dtype=np.float32)
        temp_dir, files = None, {}

    base_adjacency_matrix = calculate_adjacency_matrix(net)
    ybus_base = None
    contingency_timesteps = []
    contingency_ybus_list = []
    
    convergence_stats = {
        'total_timesteps': time_steps, 'successful': 0, 'failed': 0,
        'failed_no_contingency': [], 'failed_with_contingency': [],
        'resolution_methods': {'strict_normal': 0, 'strict_contingency': 0, 'relaxed_contingency': 0, 'restored_line': 0, 'hard_reset': 0},
        'reverted_contingencies': 0,
        'validation_stats': {'consecutive_failures': 0, 'max_consecutive_failures': 0, 'curtailment_attempts': 0, 'curtailment_successful': 0, 
                             'generator_trips': 0, 'hard_resets': 0, 'voltage_violations': 0, 'angle_violations': 0, 'line_loading_violations': 0}
    }
    
    base_load_p, base_load_q = net.load.p_mw.copy(), net.load.q_mvar.copy()
    
    num_solar = len(net.sgen[net.sgen.type == 'solar']) if 'type' in net.sgen.columns else 0
    num_wind = len(net.sgen[net.sgen.type == 'wind']) if 'type' in net.sgen.columns else 0
    num_renewables = num_solar + num_wind
    
    max_solar_mw = max_wind_mw = max_total_ren_mw = 0
    if num_renewables > 0:
        total_load = base_load_p.sum()
        target_frac = renewable_fraction if renewable_fraction is not None else 0.0
        max_total_ren_mw = (total_load * target_frac) / 0.3
        max_solar_mw = (max_total_ren_mw * (num_solar/num_renewables)) / num_solar if num_solar else 0
        max_wind_mw = (max_total_ren_mw * (num_wind/num_renewables)) / num_wind if num_wind else 0
        
        for i, sgen in net.sgen.iterrows():
            if sgen.type == 'solar': net.sgen.at[i, 'sn_mva'] = max_solar_mw * 1.1
            elif sgen.type == 'wind': net.sgen.at[i, 'sn_mva'] = max_wind_mw * 1.1
    else:
        max_total_ren_mw = 1.0

    weather_seq = simulate_weather_sequence(time_steps, config.get('hours_per_day', 24), config.get('random_seed', None))
    dropped_line_idx = None
    detailed_metrics = []
    
    if pbar is not None:
        pbar.set_description(f"Gen {case_name} (frac {renewable_fraction*100:.0f}%)")

    for t in range(time_steps):
        restore_contingency(net, dropped_line_idx)
        dropped_line_idx = None
        has_contingency = False
        
        if np.random.random() < config['contingency_rate']:
            dropped_line_idx = apply_n1_contingency(net)
            has_contingency = (dropped_line_idx is not None)
            if not has_contingency:
                convergence_stats['reverted_contingencies'] += 1

        topology_ids[t] = (dropped_line_idx + 1) if has_contingency else 0
        
        cur_hour, cur_day = t % config['hours_per_day'], t // config['hours_per_day']
        load_mult = get_daily_load_profile(cur_hour)
        net.load.p_mw = base_load_p * load_mult
        net.load.q_mvar = base_load_q * load_mult
        
        cur_total_ren_p = 0
        if not net.sgen.empty and 'type' in net.sgen.columns:
            solar_w, wind_w = weather_seq[t] if weather_seq else (None, None)
            solar_prof = get_solar_generation_profile(cur_hour, 180 + cur_day % 180, solar_w, config)
            wind_prof = get_wind_generation_profile(cur_hour, cur_day, wind_w, config)
            
            mask_s, mask_w = net.sgen.type == 'solar', net.sgen.type == 'wind'
            net.sgen.loc[mask_s, 'p_mw'] = solar_prof * max_solar_mw
            net.sgen.loc[mask_w, 'p_mw'] = wind_prof * max_wind_mw
            
            for i in net.sgen.index:
                net.sgen.at[i, 'q_mvar'] = calculate_renewable_reactive_power(net.sgen.at[i, 'p_mw'], net.sgen.at[i, 'bus'], net, t > 0)
            cur_total_ren_p = net.sgen.p_mw.sum()

        success = False
        method = None
        flags = {}
        
        if convergence_stats['validation_stats']['consecutive_failures'] >= 3:
            base_ren_reset = {i: net.sgen.at[i, 'p_mw'] for i in net.sgen.index}
            reset_ok, new_drop = hard_reset_system(net, base_load_p, base_load_q, base_ren_reset, convergence_stats, dropped_line_idx, case_name, config)
            if reset_ok:
                dropped_line_idx, has_contingency = new_drop, (new_drop is not None)
                convergence_stats['validation_stats']['consecutive_failures'] = 0
                success, method = True, 'hard_reset'
                convergence_stats['resolution_methods']['hard_reset'] += 1
            else:
                convergence_stats['failed'] += 1
                raise RuntimeError(f"Hard reset failed at {t}")

        if not success:
            valid_in, reason = validate_power_flow_inputs(net)
            if not valid_in:
                if trip_renewable_generators(net, convergence_stats, case_name, config):
                    success, method = True, 'trip_generators'
                else:
                    convergence_stats['failed'] += 1
                    raise RuntimeError(f"Pre-validation failed at {t}: {reason}")

        base_ren = {i: net.sgen.at[i, 'p_mw'] for i in net.sgen.index} if not success else {}
        if not success:
            curtail_ok, _, flags = apply_curtailment_with_retry(net, base_ren, 10, convergence_stats, has_contingency, case_name, config)
            if curtail_ok:
                success, method = True, ('strict_' + ('contingency' if has_contingency else 'normal'))
                convergence_stats['resolution_methods']['strict_contingency' if has_contingency else 'strict_normal'] += 1
            elif has_contingency:
                try:
                    with SuppressPrints(): pp.runpp(net, numba=True, enforce_q_lims=False, algorithm='nr', tolerance_mva=1e-6, max_iteration=20)
                    valid_r, _, flags_r = validate_power_flow_outputs(net, convergence_stats, case_name, config)
                    if valid_r:
                        success, method, flags = True, 'relaxed_contingency', flags_r
                        convergence_stats['resolution_methods']['relaxed_contingency'] += 1
                except: pass
                
                if not success:
                    restore_contingency(net, dropped_line_idx)
                    convergence_stats['reverted_contingencies'] += 1
                    dropped_line_idx, has_contingency = None, False
                    topology_ids[t] = 0  # Reverted
                    for i, p in base_ren.items(): net.sgen.at[i, 'p_mw'] = p
                    restore_ok, _, flags_rest = apply_curtailment_with_retry(net, base_ren, 10, convergence_stats, False, case_name, config)
                    if restore_ok:
                        success, method, flags = True, 'restored_curtailment', flags_rest
                        convergence_stats['resolution_methods']['restored_line'] += 1
                    elif trip_renewable_generators(net, convergence_stats, case_name, config):
                        success, method = True, 'trip_after_restore'
            
            if not success and not has_contingency:
                if trip_renewable_generators(net, convergence_stats, case_name, config):
                    success, method = True, 'trip_normal'

        if not success:
            convergence_stats['validation_stats']['consecutive_failures'] += 1
            convergence_stats['failed'] += 1
            continue

        convergence_stats['successful'] += 1
        convergence_stats['validation_stats']['consecutive_failures'] = 0
        if flags:
            for k in ['voltage_violation', 'angle_violation', 'line_loading_violation']:
                if flags.get(k): convergence_stats['validation_stats'][k+'s'] += 1

        # Data Extraction
        vm_pu = net.res_bus.vm_pu.values
        va_rad = np.deg2rad(net.res_bus.va_degree.values)
        bus_idx = net.bus.index
        
        load_p = net.res_load.groupby(net.load.bus).p_mw.sum().reindex(bus_idx, fill_value=0).values
        load_q = net.res_load.groupby(net.load.bus).q_mvar.sum().reindex(bus_idx, fill_value=0).values
        ext_p = net.res_ext_grid.groupby(net.ext_grid.bus).p_mw.sum().reindex(bus_idx, fill_value=0).values
        ext_q = net.res_ext_grid.groupby(net.ext_grid.bus).q_mvar.sum().reindex(bus_idx, fill_value=0).values
        gen_p = net.res_gen.groupby(net.gen.bus).p_mw.sum().reindex(bus_idx, fill_value=0).values
        gen_q = net.res_gen.groupby(net.gen.bus).q_mvar.sum().reindex(bus_idx, fill_value=0).values
        sgen_p = net.res_sgen.groupby(net.sgen.bus).p_mw.sum().reindex(bus_idx, fill_value=0).values
        sgen_q = net.res_sgen.groupby(net.sgen.bus).q_mvar.sum().reindex(bus_idx, fill_value=0).values

        ybus = calculate_ybus_from_net(net)
        if ybus_base is None: ybus_base = ybus.copy()
        elif has_contingency:
            contingency_timesteps.append(t)
            contingency_ybus_list.append(ybus.copy())

        detailed_metrics.append({'timestep': t, 'p_load': load_p.sum(), 'v_min': vm_pu.min(), 'v_max': vm_pu.max(), 'converged': 1, 'method': method})

        ren_util = cur_total_ren_p / max_total_ren_mw if max_total_ren_mw > 0 else 0
        time_carbon_coeffs[t] = config['base_carbon_intensity_grid'] - (ren_util * config['max_carbon_reduction_from_renewables'])
        
        bus_types = identify_bus_types(net)
        bus_types_array[t] = bus_types
        target_matrix[t] = create_opf_targets(net, bus_types)
        
        rng = np.random.default_rng()
        noise_vm = np.abs(rng.normal(0, config['voltage_error_std'], num_buses))
        noise_va = np.abs(rng.normal(0, config['angle_error_std'], num_buses))
        noise_p = np.abs(rng.normal(0, config['power_error_std'], num_buses))
        
        meas_data = [load_p, load_q, ext_p, ext_q, gen_p, gen_q, sgen_p, sgen_q]
        meas_data = [d * (1 + noise_p) for d in meas_data]
        
        meas_vm, meas_va = np.full(num_buses, np.nan), np.full(num_buses, np.nan)
        gen_idx = np.where(bus_types > 0)[0]
        n_pmu = max(1, int(num_buses * config.get('pmu_coverage', 0.3)))
        pmu_idx = set(gen_idx)
        if len(pmu_idx) < n_pmu:
            pq_idx = np.where(bus_types == 0)[0]
            if len(pq_idx) > 0: pmu_idx.update(np.random.choice(pq_idx, min(len(pq_idx), n_pmu-len(pmu_idx)), replace=False))
        
        pmu_buses = np.array(list(pmu_idx))
        meas_vm[pmu_buses] = vm_pu[pmu_buses] * (1 + noise_vm[pmu_buses])
        meas_va[pmu_buses] = va_rad[pmu_buses] * (1 + noise_va[pmu_buses])
        
        # Compute bus degree (number of active connections) for the current topology
        if has_contingency:
            # Degree from current Ybus: count non-zero off-diagonal entries per row
            ybus_abs = np.abs(ybus)
            np.fill_diagonal(ybus_abs, 0)
            degree_vals = (ybus_abs > 1e-6).sum(axis=1).astype(np.float32)
        else:
            degree_vals = base_adjacency_matrix.sum(axis=1).astype(np.float32)
            
        feature_matrix[t] = np.stack(meas_data + [np.nan_to_num(meas_vm), np.nan_to_num(meas_va), degree_vals], axis=1)
        if pbar is not None:
            pbar.update(1)
        if chunked_mode and (t + 1) % chunk_size == 0:
            for arr in [feature_matrix, target_matrix, bus_types_array, topology_ids, time_carbon_coeffs]: arr.flush()

    if chunked_mode:
        for arr in [feature_matrix, target_matrix, bus_types_array, topology_ids, time_carbon_coeffs]: arr.flush()
    
    ybus_data = {"base": ybus_base, "contingency_timesteps": np.array(contingency_timesteps, dtype=np.int32),
                 "contingency_matrices": np.array(contingency_ybus_list) if contingency_ybus_list else np.array([]).reshape(0, num_buses, num_buses)}
    
    ret = { "features": np.array(feature_matrix) if chunked_mode else feature_matrix, 
            "targets": np.array(target_matrix) if chunked_mode else target_matrix, 
            "bus_types": np.array(bus_types_array) if chunked_mode else bus_types_array,
            "base_adjacency": base_adjacency_matrix, "topology_ids": np.array(topology_ids) if chunked_mode else topology_ids,
            "ybus_data": ybus_data, "time_carbon_coeffs": np.array(time_carbon_coeffs) if chunked_mode else time_carbon_coeffs,
            "convergence_stats": convergence_stats }

    if chunked_mode:
        shutil.rmtree(temp_dir, ignore_errors=True)
    if output_dir:
        pd.DataFrame(detailed_metrics).to_csv(os.path.join(output_dir, f"{case_name}_detailed_metrics_frac{renewable_fraction:.1f}.csv"), index=False)
    return ret

def save_data(data: dict, case: str, frac: float, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for k, v in data.items():
        base = f"{case}_{k}_frac{frac:.1f}"
        if k == 'ybus_data':
            for sk, sv in v.items(): np.save(os.path.join(out_dir, f"{case}_ybus_{sk}_frac{frac:.1f}.npy"), sv)
        elif k == 'convergence_stats':
            with open(os.path.join(out_dir, f"{case}_data_quality_audit_frac{frac:.1f}.json"), 'w') as f: json.dump(v, f, indent=2)
        elif 'coeffs' in k: np.savetxt(os.path.join(out_dir, base+".txt"), v)
        elif k == 'base_adjacency':
            adj = sparse.coo_matrix(v)
            np.save(os.path.join(out_dir, base+".npy"), np.array([[adj.row, adj.col]], dtype=object), allow_pickle=True)
        else: np.save(os.path.join(out_dir, base+".npy"), v)

if __name__ == "__main__":
    cases = CONFIG["test_cases"]
    if args.buses and args.buses.lower() != 'all':
        req = [int(b) for b in args.buses.split(',')]
        cases = [f"case{b}" for b in req if f"case{b}" in cases]
    
    np.random.seed(CONFIG["random_seed"])
    random.seed(CONFIG["random_seed"])

    
    # Cleanup legacy and professional directories
    legacy_plots = os.path.join(root_dir, 'data', 'plots_train')
    if os.path.exists(legacy_plots): shutil.rmtree(legacy_plots)
    
    # Ensure directories exist without wiping them
    os.makedirs(CONFIG['reports_dir'], exist_ok=True)
    out_path = CONFIG['output_dir']
    os.makedirs(out_path, exist_ok=True)
    
    for case in cases:
        # 1. Cleanup old data files for this case
        old_data = glob.glob(os.path.join(out_path, f"{case}_*"))
        for f in old_data:
            try: os.remove(f)
            except: pass
            
        # 2. Cleanup old plots for this case (pattern: *_33bus.png)
        num_buses = case.replace('case', '').replace('bw', '')
        old_plots = glob.glob(os.path.join(CONFIG['reports_dir'], f"*_{num_buses}bus.png"))
        for f in old_plots:
            try: os.remove(f)
            except: pass
                
        try:
            base_net = load_network(case)
            fracs = CONFIG["renewable_fractions_to_run"]
            total_steps = len(fracs) * CONFIG['time_steps']
            num_buses = case.replace('case', '').replace('bw', '')
            print(f"\n{num_buses}-bus | {len(fracs)} fracs × {CONFIG['time_steps']} steps")
            pbar = tqdm(total=total_steps, desc=f"Gen {case}",
                        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} steps",
                        unit="step")
            total_reverted = 0
            for frac in fracs:
                net = copy.deepcopy(base_net)
                s_case = case.replace('bw', '')
                net.name = f"{s_case}_frac{frac:.1f}"
                configure_renewables(net, frac, CONFIG)
                data = simulate_time_series(net, CONFIG, out_path, s_case, frac, pbar=pbar)
                save_data(data, s_case, frac, out_path)
                total_reverted += data.get('convergence_stats', {}).get('reverted_contingencies', 0)
            pbar.close()
            
            # Print feedback if contingencies were heavily rejected (Radial systems)
            if total_reverted > 0:
                print(f"  Note: {total_reverted} contingencies were attempted but reverted due to system instability (normal for radial grids).")
        except Exception as e:
            print(f"Error processing {case}: {e}")
            traceback.print_exc()

    # Plots
    print()
    try:
        from src.visualization.plot_consolidator import generate_all_data_plots
        buses = [int(c.replace('case', '').replace('bw', '')) for c in cases]
        if buses:
            generate_all_data_plots(gen_config, buses, CONFIG['reports_dir'])
    except Exception as e:
        print(f"Plot error: {e}")
