import os
import sys
import traceback
import json
import warnings
import copy
import gc
import shutil
import tempfile
import random
import glob
import hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import pandapower as pp
from tqdm import tqdm
from scipy import sparse

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)

from data.profiles import (
    get_daily_load_profile, get_solar_generation_profile, 
    get_wind_generation_profile, simulate_weather_sequence,
    calculate_renewable_reactive_power
)
from data.topology import (
    load_network, configure_renewables, apply_n1_contingency, 
    restore_contingency, calculate_ybus_from_net, calculate_adjacency_matrix,
    identify_bus_types, create_opf_targets
)
from data.validation import (
    SuppressPrints, validate_power_flow_inputs, validate_power_flow_outputs,
    apply_curtailment_with_retry, hard_reset_system, trip_renewable_generators
)
from utils.contingency_ybus import DataGenerationError
from config import Config

warnings.filterwarnings('ignore', category=FutureWarning)

args = Config.parse_cli_args()
try:
    config_instance = Config(cli_args=args, load_yaml=True, data_mode='train')
except Exception as e:
    raise RuntimeError(f"Critical config error: {e}") from e

data_mode = config_instance.DATA_MODE
timesteps_to_use = args.timesteps if args.timesteps is not None else config_instance.DATA_MODE_TIMESTEPS[data_mode]
output_dir_to_use = config_instance.DATA_DIR

CONFIG = {
    "random_seed": 42,
    "test_cases": getattr(config_instance, 'TEST_CASES', ["case33", "case57", "case118"]),
    "time_steps": timesteps_to_use,
    "output_dir": output_dir_to_use,
    "renewable_fractions_to_run": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "contingency_rate": getattr(config_instance, 'CONTINGENCY_RATE', 0.05),
    "voltage_error_std": 0.005,
    "power_error_std": 0.01,
    "angle_error_std": 0.02,
    "max_energy_utilization_coeff": 0.98,
    "loss_sensitivity": 0.01,
    "base_carbon_intensity_grid": 0.55,
    "max_carbon_reduction_from_renewables": 0.30,
    "hours_per_day": getattr(config_instance, 'HOURS_PER_DAY', 24),
    "seed": 42,
    "chunk_size": 1000,
    "use_chunked_writing": True,
    "pmu_coverage": getattr(config_instance, 'PMU_COVERAGE', 0.3)
}

def simulate_time_series(net: pp.pandapowerNet, config: dict, output_dir: str = None, 
                         case_name: str = None, renewable_fraction: float = None, 
                         timestamp: str = None) -> dict:
    num_buses = len(net.bus)
    time_steps = config['time_steps']
    chunk_size = config.get('chunk_size', 1000)
    chunked_mode = (config.get('use_chunked_writing', True) and output_dir and 
                    case_name and renewable_fraction is not None and timestamp)
    
    if chunked_mode:
        temp_dir = tempfile.mkdtemp(prefix='gen_data_chunks_')
        files = {k: os.path.join(temp_dir, f'{k}_temp.npy') for k in ['features', 'targets', 'bus_types', 'topology_ids', 'carbon_coeffs']}
        
        feature_matrix = np.memmap(files['features'], mode='w+', dtype=np.float32, shape=(time_steps, num_buses, 10))
        target_matrix = np.memmap(files['targets'], mode='w+', dtype=np.float32, shape=(time_steps, num_buses, 10))
        bus_types_array = np.memmap(files['bus_types'], mode='w+', dtype=np.int32, shape=(time_steps, num_buses))
        topology_ids = np.memmap(files['topology_ids'], mode='w+', dtype=np.int32, shape=(time_steps,))
        time_carbon_coeffs = np.memmap(files['carbon_coeffs'], mode='w+', dtype=np.float32, shape=(time_steps,))
    else:
        feature_matrix = np.zeros((time_steps, num_buses, 10), dtype=np.float32)
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
        'validation_stats': {'consecutive_failures': 0, 'max_consecutive_failures': 0, 'curtailment_attempts': 0, 'curtailment_successful': 0, 
                             'generator_trips': 0, 'hard_resets': 0, 'voltage_violations': 0, 'angle_violations': 0, 'line_loading_violations': 0}
    }
    
    base_load_p, base_load_q = net.load.p_mw.copy(), net.load.q_mvar.copy()
    total_load = base_load_p.sum()
    
    num_solar = len(net.sgen[net.sgen.type == 'solar']) if 'type' in net.sgen.columns else 0
    num_wind = len(net.sgen[net.sgen.type == 'wind']) if 'type' in net.sgen.columns else 0
    num_renewables = num_solar + num_wind
    
    max_solar_mw = max_wind_mw = max_total_ren_mw = 0
    if num_renewables > 0:
        target_frac = renewable_fraction if renewable_fraction is not None else 0.0
        max_total_ren_mw = (total_load * target_frac) / 0.3
        max_solar_mw = (max_total_ren_mw * (num_solar/num_renewables)) / num_solar if num_solar else 0
        max_wind_mw = (max_total_ren_mw * (num_wind/num_renewables)) / num_wind if num_wind else 0
        
        for i, sgen in net.sgen.iterrows():
            if sgen.type == 'solar': net.sgen.at[i, 'sn_mva'] = max_solar_mw * 1.1
            elif sgen.type == 'wind': net.sgen.at[i, 'sn_mva'] = max_wind_mw * 1.1
    else:
        max_total_ren_mw = 1.0

    weather_seq = simulate_weather_sequence(time_steps, config.get('hours_per_day', 24), config.get('seed', None))
    dropped_line_idx = None
    detailed_metrics = []
    
    disable_pbar = config.get('no_progress_bar', False) or getattr(args, 'no_progress_bar', False)
    iterator = range(time_steps) if disable_pbar else tqdm(range(time_steps), desc=f"Gen {case_name} ({renewable_fraction*100:.0f}%)", unit="step", miniters=1)

    for t in iterator:
        restore_contingency(net, dropped_line_idx)
        dropped_line_idx = None
        has_contingency = False
        
        if np.random.random() < config['contingency_rate']:
            dropped_line_idx = apply_n1_contingency(net)
            has_contingency = (dropped_line_idx is not None)

        topology_ids[t] = (dropped_line_idx + 1) if has_contingency else 0
        
        cur_hour, cur_day = t % config['hours_per_day'], t // config['hours_per_day']
        load_mult = get_daily_load_profile(cur_hour)
        net.load.p_mw = base_load_p * load_mult
        net.load.q_mvar = base_load_q * load_mult
        
        cur_total_ren_p = 0
        if not net.sgen.empty and 'type' in net.sgen.columns:
            solar_w, wind_w = weather_seq[t] if weather_seq else (None, None)
            
            solar_prof = get_solar_generation_profile(cur_hour, 180 + cur_day % 180, solar_w)
            wind_prof = get_wind_generation_profile(cur_hour, cur_day, wind_w)
            
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
            base_ren_reset = {} 
            if not net.sgen.empty:
                for i, sgen in net.sgen.iterrows():
                    base_ren_reset[i] = net.sgen.at[i, 'p_mw']
            
            reset_ok, new_drop = hard_reset_system(net, base_load_p, base_load_q, base_ren_reset, convergence_stats, dropped_line_idx, case_name)
            if reset_ok:
                dropped_line_idx, has_contingency = new_drop, (new_drop is not None)
                convergence_stats['validation_stats']['consecutive_failures'] = 0
                success, method = True, 'hard_reset'
                convergence_stats['resolution_methods']['hard_reset'] += 1
            else:
                convergence_stats['validation_stats']['consecutive_failures'] += 1
                convergence_stats['failed'] += 1
                raise RuntimeError(f"Hard reset failed at {t}")

        if not success:
            valid_in, reason = validate_power_flow_inputs(net)
            if not valid_in:
                if trip_renewable_generators(net, convergence_stats, case_name):
                    success, method = True, 'trip_generators'
                else:
                    convergence_stats['failed'] += 1
                    raise RuntimeError(f"Pre-validation failed at {t}: {reason}")

        base_ren = {i: net.sgen.at[i, 'p_mw'] for i in net.sgen.index} if not success else {}
        
        if not success:
            curtail_ok, scale, flags = apply_curtailment_with_retry(net, base_ren, 10, convergence_stats, has_contingency, case_name)
            if curtail_ok:
                success = True
                method = ('curtailment_' + ('contingency' if has_contingency else 'normal')) if scale < 1.0 else ('strict_' + ('contingency' if has_contingency else 'normal'))
                convergence_stats['resolution_methods']['strict_contingency' if has_contingency else 'strict_normal'] += 1
            elif has_contingency:
                # Try relaxed contingency
                try:
                    with SuppressPrints(): pp.runpp(net, numba=True, enforce_q_lims=False, algorithm='nr', tolerance_mva=1e-6, max_iteration=20)
                    valid_r, _, flags_r = validate_power_flow_outputs(net, convergence_stats)
                    if valid_r:
                        success, method, flags = True, 'relaxed_contingency', flags_r
                        convergence_stats['resolution_methods']['relaxed_contingency'] += 1
                except pp.LoadflowNotConverged:
                    # Expected failure mode for relaxed contingency; just continue to restoration
                    pass
                except Exception as e:
                     print(f"Warning: Unexpected error in relaxed contingency: {e}")
                
                if not success: # Restore line
                    restore_contingency(net, dropped_line_idx)
                    dropped_line_idx, has_contingency = None, False
                    for i, p in base_ren.items(): net.sgen.at[i, 'p_mw'] = p
                    
                    restore_ok, _, flags_rest = apply_curtailment_with_retry(net, base_ren, 10, convergence_stats, False)
                    if restore_ok:
                        success, method, flags = True, 'restored_curtailment', flags_rest
                        convergence_stats['resolution_methods']['restored_line'] += 1
                    elif trip_renewable_generators(net, convergence_stats):
                        success, method = True, 'trip_after_restore'
            
            if not success and not has_contingency:
                if trip_renewable_generators(net, convergence_stats):
                    success, method = True, 'trip_normal'

        if not success:
            convergence_stats['validation_stats']['consecutive_failures'] += 1
            convergence_stats['failed'] += 1
            (convergence_stats['failed_with_contingency'] if has_contingency else convergence_stats['failed_no_contingency']).append(t)
            print(f"  ERROR: Timestep {t} failed")
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

        detailed_metrics.append({
            'timestep': t, 'p_load': load_p.sum(), 'p_gen': (ext_p+gen_p).sum(), 'p_ren': sgen_p.sum(),
            'v_min': vm_pu.min(), 'v_max': vm_pu.max(), 'converged': 1, 'method': method or 'Normal'
        })

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
        
        meas_vm = np.full(num_buses, np.nan)
        meas_va = np.full(num_buses, np.nan)
        
        gen_idx = np.where(bus_types > 0)[0]
        n_pmu = max(1, int(num_buses * config.get('pmu_coverage', 0.3)))
        pmu_idx = set(gen_idx)
        if len(pmu_idx) < n_pmu:
            pq_idx = np.where(bus_types == 0)[0]
            if len(pq_idx) > 0: pmu_idx.update(np.random.choice(pq_idx, min(len(pq_idx), n_pmu-len(pmu_idx)), replace=False))
        
        pmu_buses = np.array(list(pmu_idx))
        meas_vm[pmu_buses] = vm_pu[pmu_buses] * (1 + noise_vm[pmu_buses])
        meas_va[pmu_buses] = va_rad[pmu_buses] * (1 + noise_va[pmu_buses])
        
        feature_matrix[t] = np.stack(meas_data + [np.nan_to_num(meas_vm), np.nan_to_num(meas_va)], axis=1)
        
        if chunked_mode and (t + 1) % chunk_size == 0:
            for arr in [feature_matrix, target_matrix, bus_types_array, topology_ids, time_carbon_coeffs]: arr.flush()

    if chunked_mode:
        for arr in [feature_matrix, target_matrix, bus_types_array, topology_ids, time_carbon_coeffs]: arr.flush()
    
    if ybus_base is None: raise RuntimeError("Ybus base is None")
    
    ybus_data = {"base": ybus_base, "contingency_timesteps": np.array(contingency_timesteps, dtype=np.int32),
                 "contingency_matrices": np.array(contingency_ybus_list) if contingency_ybus_list else np.array([]).reshape(0, num_buses, num_buses)}
    
    ret_features = np.array(feature_matrix) if chunked_mode else feature_matrix
    ret_targets = np.array(target_matrix) if chunked_mode else target_matrix
    ret_bus_types = np.array(bus_types_array) if chunked_mode else bus_types_array
    ret_topo = np.array(topology_ids) if chunked_mode else topology_ids
    ret_carbon = np.array(time_carbon_coeffs) if chunked_mode else time_carbon_coeffs
    
    if chunked_mode:
        del feature_matrix, target_matrix, bus_types_array, topology_ids, time_carbon_coeffs
        gc.collect()
        convergence_stats.update({
            '_temp_dir': temp_dir, 
            '_temp_files': files,
            '_shapes': {
                'features': (time_steps, num_buses, 10),
                'targets': (time_steps, num_buses, 10),
                'bus_types': (time_steps, num_buses),
                'topology_ids': (time_steps,),
                'carbon_coeffs': (time_steps,)
            }
        })

    if output_dir:
        pd.DataFrame(detailed_metrics).to_csv(os.path.join(output_dir, f"{case_name}_detailed_metrics_frac{renewable_fraction:.1f}_{timestamp}.csv"), index=False)

    return {
        "features": ret_features, "targets": ret_targets, "bus_types": ret_bus_types,
        "base_adjacency": base_adjacency_matrix, "topology_ids": ret_topo,
        "ybus_data": ybus_data, "time_carbon_coeffs": ret_carbon,
        "convergence_stats": convergence_stats
    }

def save_data(data: dict, case: str, frac: float, out_dir: str, ts: str = None):
    os.makedirs(out_dir, exist_ok=True)
    ts = ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    stats = data.get('convergence_stats', {})
    temp_files = stats.get('_temp_files')
    shapes = stats.get('_shapes', {})
    
    file_map = {
        'features': f"{case}_features_frac{frac:.1f}_{ts}.npy",
        'targets': f"{case}_targets_frac{frac:.1f}_{ts}.npy",
        'bus_types': f"{case}_bus_types_frac{frac:.1f}_{ts}.npy",
        'topology_ids': f"{case}_topology_ids_frac{frac:.1f}_{ts}.npy",
        'carbon_coeffs': f"{case}_time_carbon_coeffs_frac{frac:.1f}_{ts}.txt"
    }
    
    if temp_files and os.path.exists(temp_files['features']):
        for k, fname in file_map.items():
            src = temp_files[k]
            dst = os.path.join(out_dir, fname)
            if k == 'carbon_coeffs':
                np.savetxt(dst, np.array(np.memmap(src, mode='r', dtype=np.float32)))
            else:
                dtype = np.int32 if k in ['bus_types', 'topology_ids'] else np.float32
                shape = shapes.get(k)
                if shape:
                    arr = np.memmap(src, mode='r', dtype=dtype, shape=shape)
                    np.save(dst, np.array(arr), allow_pickle=False)
                else:
                    np.save(dst, np.array(np.memmap(src, mode='r', dtype=dtype)), allow_pickle=False)
    
    for k, v in data.items():
        if temp_files and k in ['features', 'targets', 'bus_types', 'topology_ids', 'time_carbon_coeffs']: continue
        
        base = f"{case}_{k}_frac{frac:.1f}_{ts}"
        if k == 'ybus_data':
            for sk, sv in v.items():
                np.save(os.path.join(out_dir, f"{case}_ybus_{sk}_frac{frac:.1f}_{ts}.npy"), sv, allow_pickle=False)
        elif k == 'convergence_stats':
            clean_stats = {sk: sv for sk, sv in v.items() if not sk.startswith('_temp')}
            with open(os.path.join(out_dir, f"{case}_data_quality_audit_frac{frac:.1f}_{ts}.json"), 'w') as f:
                json.dump(clean_stats, f, indent=2)
        elif 'coeffs' in k:
            np.savetxt(os.path.join(out_dir, base+".txt"), v)
        elif k == 'base_adjacency':
            adj = sparse.coo_matrix(v)
            np.save(os.path.join(out_dir, base+".npy"), np.array([[adj.row, adj.col]], dtype=object), allow_pickle=True)
        else:
            np.save(os.path.join(out_dir, base+".npy"), v, allow_pickle=True)

if __name__ == "__main__":
    if CONFIG["random_seed"]:
        np.random.seed(CONFIG["random_seed"])
        random.seed(CONFIG["random_seed"])
    
    cases = CONFIG["test_cases"]
    if args.buses and args.buses.lower() != 'all':
        req_buses = [int(b) for b in args.buses.split(',')]
        cases = [f"case{b}" for b in req_buses if f"case{b}" in cases]
        if not cases: raise ValueError(f"No valid buses in {req_buses}")
    
    # CLI args override config
    timesteps_cli = args.timesteps
    CONFIG['time_steps'] = timesteps_cli if timesteps_cli is not None else CONFIG['time_steps']
    
    # If no CLI arg and no config value, default based on mode
    if CONFIG['time_steps'] is None:
        CONFIG['time_steps'] = 10008 if data_mode == 'train' else 240
        
    gen_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n[Data Gen] {data_mode.upper()} | {CONFIG['time_steps']} steps | Buses: {', '.join(cases)}")
    
    out_path = CONFIG['output_dir']
    os.makedirs(out_path, exist_ok=True)
    
    # Cleanup old files only if explicitly requested
    if getattr(config_instance, 'clear_results', False):
        print("Cleaning up old data files...")
        for c in cases:
            for f in glob.glob(os.path.join(out_path, f"{c.replace('bw', '')}_*")):
                if os.path.exists(f): os.remove(f)
            
    # Always cleanup old files for the specific cases being generated
    # main.py is the explicit generator script, so running it implies "generate new data"
    print(f"Cleaning old data files for: {', '.join(cases)}...")
    
    # Preserve metadata for parallel execution safety
    for c in cases:
        for f in glob.glob(os.path.join(out_path, f"{c.replace('bw', '')}_*")):
            if os.path.exists(f): os.remove(f)
            
    for case in cases:
        try:
            base_net = load_network(case)
            for frac in CONFIG["renewable_fractions_to_run"]:
                net = copy.deepcopy(base_net)
                s_case = case.replace('bw', '')
                net.name = f"{s_case}_frac{frac:.1f}"
                configure_renewables(net, frac, CONFIG)
                
                data = simulate_time_series(net, CONFIG, out_path, s_case, frac, gen_ts)
                save_data(data, s_case, frac, out_path, gen_ts)
                
                if '_temp_dir' in data.get('convergence_stats', {}):
                    shutil.rmtree(data['convergence_stats']['_temp_dir'], ignore_errors=True)
                    
        except Exception as e:
            print(f"Error processing {case}: {e}")
            traceback.print_exc()

    # Metadata
    try:
        meta = {
            'data_mode': data_mode, 'generation_mode': 'time_series',
            'timesteps': CONFIG['time_steps'], 'ts': gen_ts,
            'test_cases': cases, 'fracs': CONFIG["renewable_fractions_to_run"],
            'date': datetime.now().isoformat(),
            'config_hash': hashlib.md5(json.dumps({
                'data_mode': data_mode,
                'timesteps': CONFIG['time_steps'],
                'hours_per_day': CONFIG.get('hours_per_day', 24),
                'contingency_rate': CONFIG.get('contingency_rate', 0.05),
                'pmu_coverage': CONFIG.get('pmu_coverage', 0.3)
            }, sort_keys=True).encode()).hexdigest()
        }
        pid = os.getpid()
        with open(os.path.join(out_path, f"data_generation_metadata_{pid}_{gen_ts}.json"), 'w') as f:
            json.dump({'runs': [meta], 'pid': pid}, f, indent=2)
    except Exception as e:
        print(f"Metadata error: {e}")

    # Plots
    try:
        from data.plot_consolidator import generate_all_data_plots
        plots_dir = os.path.join(os.path.dirname(out_path), f'plots_{data_mode}')
        buses = [int(c.replace('case', '').replace('bw', '')) for c in cases]
        if buses:
            print(f"\nGenerating plots for: {buses}")
            generate_all_data_plots(config_instance, buses, plots_dir)
    except Exception as e:
        print(f"Plot error: {e}")
