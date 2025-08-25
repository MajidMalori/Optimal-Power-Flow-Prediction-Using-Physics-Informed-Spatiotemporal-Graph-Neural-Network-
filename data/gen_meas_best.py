# File: data/gen_meas_best.py

import os
import traceback
import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import copy

# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================
CONFIG = {
    "test_cases": ["case33bw", "case57", "case118"],
    "time_steps": 100,
    "output_dir": "./data", # Save to a dedicated data folder
    "renewable_fractions_to_run": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 
    "max_solar_mw": 5.0,
    "max_wind_mw": 8.0,
    "contingency_rate": 0.05,
    "voltage_error_std": 0.005,
    "power_error_std": 0.01,
    "angle_error_std": 0.02,
    "max_energy_utilization_coeff": 0.98,
    "loss_sensitivity": 0.01,
    "base_carbon_intensity_grid": 0.55,
    "max_carbon_reduction_from_renewables": 0.30
}

# =============================================================================
# SECTION 2: HELPER FUNCTIONS
# =============================================================================

def load_network(case_name: str) -> pp.pandapowerNet:
    """Loads a pandapower network based on its name."""
    print(f"\n----- Loading Base Test Case: {case_name} -----")
    if case_name == "case33bw": return pn.case33bw()
    if case_name == "case57": return pn.case57()
    if case_name == "case118": return pn.case118()
    raise ValueError(f"Unknown test case: {case_name}")

def configure_renewables(net: pp.pandapowerNet, renewable_fraction_for_run: float, config: dict) -> pp.pandapowerNet:
    """Adds renewable static generators (sgen) to the network for a specific fraction."""
    num_buses = len(net.bus)
    num_renewables = int(num_buses * renewable_fraction_for_run)
    slack_buses = set(net.ext_grid.bus)
    possible_buses = list(set(net.bus.index) - slack_buses)
    
    net.sgen.drop(net.sgen.index, inplace=True)
    
    if len(possible_buses) < num_renewables:
        print(f"Warning: Not enough non-slack buses. Using {len(possible_buses)} of {num_renewables} requested.")
        num_renewables = len(possible_buses)
        
    if num_renewables == 0:
        print("Configuring network with 0 renewable generators.")
        return net

    renewable_buses = np.random.choice(possible_buses, size=num_renewables, replace=False)
    
    if 'type' not in net.sgen.columns: net.sgen['type'] = pd.Series(dtype=str)
        
    for bus_idx in renewable_buses:
        gen_type = np.random.choice(['solar', 'wind'])
        pp.create_sgen(net, bus=bus_idx, p_mw=0, q_mvar=0, name=f"{gen_type.capitalize()}@{bus_idx}", type=gen_type)
        
    print(f"Configured {len(net.sgen)} renewable generators for a {renewable_fraction_for_run*100:.0f}% fraction.")
    return net

def apply_n1_contingency(net: pp.pandapowerNet) -> int:
    """Randomly takes one active line out of service if it doesn't cause islanding."""
    active_lines = net.line.index[net.line.in_service]
    if not active_lines.any(): return None
    
    for line_to_drop in np.random.permutation(active_lines.values):
        net.line.loc[line_to_drop, 'in_service'] = False
        if nx.is_connected(top.create_nxgraph(net, include_trafos=True)):
            return line_to_drop
        net.line.loc[line_to_drop, 'in_service'] = True
    return None

def restore_contingency(net: pp.pandapowerNet, dropped_line_idx: int):
    """Restores a line that was previously taken out of service."""
    if dropped_line_idx is not None:
        net.line.loc[dropped_line_idx, 'in_service'] = True


def calculate_ybus_from_net(net: pp.pandapowerNet) -> np.ndarray:
    """
    Calculates the Ybus matrix from scratch based on the pandapower network data.
    This function ensures the Ybus is always ordered by the external bus indices (0 to N-1),
    avoiding the internal/external indexing problem.

    Args:
        net: The pandapower network object.

    Returns:
        A dense numpy array representing the Ybus matrix.
    """
    num_buses = len(net.bus)
    ybus = np.zeros((num_buses, num_buses), dtype=np.complex128)

    # --- 1. Process Lines ---
    # Get only lines that are currently in service
    active_lines = net.line[net.line.in_service]
    for _, line in active_lines.iterrows():
        from_bus = int(line.from_bus)
        to_bus = int(line.to_bus)
        
        # Series impedance and admittance
        r_ohm = line.r_ohm_per_km * line.length_km
        x_ohm = line.x_ohm_per_km * line.length_km
        z_series = r_ohm + 1j * x_ohm
        y_series = 1.0 / z_series if z_series != 0 else 1e9 # Avoid division by zero
        
        # Shunt admittance (line charging) - half at each end
        b_shunt_siemens = 1j * line.c_nf_per_km * line.length_km * 2 * np.pi * net.f_hz * 1e-9
        y_shunt_half = b_shunt_siemens / 2.0
        
        # Add to off-diagonal elements
        ybus[from_bus, to_bus] -= y_series
        ybus[to_bus, from_bus] -= y_series
        
        # Add to diagonal elements
        ybus[from_bus, from_bus] += y_series + y_shunt_half
        ybus[to_bus, to_bus] += y_series + y_shunt_half

    # --- 2. Process Transformers (if any) ---
    # This part is simplified; for a full implementation, 3-winding transformers
    # and more complex tap settings would be needed.
    if 'trafo' in net and not net.trafo.empty:
        active_trafos = net.trafo[net.trafo.in_service]
        for _, trafo in active_trafos.iterrows():
            hv_bus = int(trafo.hv_bus)
            lv_bus = int(trafo.lv_bus)
            
            # Simplified impedance calculation
            z_trafo = (trafo.vk_percent / 100.0) * (net.sn_mva / trafo.sn_mva)
            y_trafo = 1.0 / (1j * z_trafo) # Assuming primarily reactive
            
            # Tap ratio (simplified)
            tap_ratio = 1.0 # Assume 1.0 if not specified
            
            # Add to matrix
            ybus[hv_bus, hv_bus] += y_trafo / (tap_ratio**2)
            ybus[lv_bus, lv_bus] += y_trafo
            ybus[hv_bus, lv_bus] -= y_trafo / tap_ratio
            ybus[lv_bus, hv_bus] -= y_trafo / tap_ratio

    # --- 3. Process Shunt Elements (e.g., capacitors, reactors) ---
    if 'shunt' in net and not net.shunt.empty:
        active_shunts = net.shunt[net.shunt.in_service]
        for _, shunt in active_shunts.iterrows():
            bus = int(shunt.bus)
            # Shunt admittance is p_mw + j*q_mvar (at 1.0 pu voltage)
            y_shunt = (shunt.p_mw + 1j * shunt.q_mvar) / net.sn_mva
            ybus[bus, bus] += y_shunt
            
    return ybus

# =============================================================================
# SECTION 3: SIMULATION AND SAVING
# =============================================================================

def simulate_time_series(net: pp.pandapowerNet, config: dict) -> dict:
    """Runs the main time-series power flow simulation."""
    num_buses = len(net.bus)
    time_steps = config['time_steps']
    
    feature_matrix = np.zeros((time_steps, num_buses, 6))
    target_matrix = np.zeros((time_steps, num_buses, 6))
    adjacency_array = np.empty((time_steps,), dtype=object)
    ybus_array = np.zeros((time_steps, num_buses, num_buses), dtype=np.complex128)
    time_energy_coeffs = np.zeros(time_steps)
    time_carbon_coeffs = np.zeros(time_steps)
    
    base_load_p, base_load_q = net.load.p_mw.copy(), net.load.q_mvar.copy()
    
    solar_gens = net.sgen[net.sgen.type == 'solar'] if 'type' in net.sgen.columns else pd.DataFrame()
    wind_gens = net.sgen[net.sgen.type == 'wind'] if 'type' in net.sgen.columns else pd.DataFrame()
    max_total_renewable_mw = (len(solar_gens) * config['max_solar_mw'] + len(wind_gens) * config['max_wind_mw']) or 1.0
        
    dropped_line_idx = None
    with tqdm(total=time_steps, desc=f"Simulating {net.name}", unit="step") as pbar:
        for t in range(time_steps):
            # Restore any previous contingency
            restore_contingency(net, dropped_line_idx)
            dropped_line_idx = None
            
            # Apply a new N-1 contingency based on the configured rate
            if np.random.random() < config['contingency_rate']:
                dropped_line_idx = apply_n1_contingency(net)

            # Create the graph adjacency matrix for the current topology
            graph = top.create_nxgraph(net, include_lines=True, include_trafos=True)
            adj_coo = nx.to_scipy_sparse_array(graph, format='coo')
            adjacency_array[t] = np.vstack([adj_coo.row, adj_coo.col])

            # Apply random variations to loads
            net.load.p_mw = base_load_p * np.random.uniform(0.8, 1.2, len(base_load_p))
            net.load.q_mvar = base_load_q * np.random.uniform(0.8, 1.2, len(base_load_q))

            # Apply random variations to renewable generation based on time of day
            current_total_renewable_p_mw = 0
            if 'type' in net.sgen.columns and not net.sgen.empty:
                for i, sgen in net.sgen.iterrows():
                    p_gen = 0
                    if sgen.type == 'solar':
                        p_gen = np.random.uniform(0, config['max_solar_mw']) if 7 <= (t % 24) < 19 else 0
                    elif sgen.type == 'wind':
                        p_gen = np.random.uniform(0, config['max_wind_mw'])
                    net.sgen.at[i, 'p_mw'] = p_gen
                    current_total_renewable_p_mw += p_gen
            
            try:
                # Run the power flow calculation
                pp.runpp(net, numba=True, enforce_q_lims=True, algorithm='nr', tolerance_mva=1e-8)
            except pp.LoadflowNotConverged:
                # If it fails, copy the state from the previous timestep
                if t > 0:
                    feature_matrix[t], target_matrix[t] = feature_matrix[t-1], target_matrix[t-1]
                    time_energy_coeffs[t], time_carbon_coeffs[t] = time_energy_coeffs[t-1], time_carbon_coeffs[t-1]
                    adjacency_array[t] = adjacency_array[t-1]
                    ybus_array[t] = ybus_array[t-1]
                pbar.update(1)
                continue
            
            # --- START DATA AGGREGATION (CONSISTENT 0 to N-1 ORDERING) ---
            
            # 1. Get bus voltages and angles. These are already ordered correctly by net.bus.index.
            vm_pu = net.res_bus.vm_pu.values
            va_rad = np.deg2rad(net.res_bus.va_degree.values)
            
            # 2. Aggregate loads. `reindex` ensures we have a value for every bus, in order.
            load_p_by_bus = net.res_load.groupby(net.load.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)
            load_q_by_bus = net.res_load.groupby(net.load.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0)
            p_load = load_p_by_bus.values
            q_load = load_q_by_bus.values

            # 3. Aggregate conventional generators
            gen_p_by_bus = net.res_gen.groupby(net.gen.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)
            gen_q_by_bus = net.res_gen.groupby(net.gen.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0)

            # 4. Aggregate static (renewable) generators
            sgen_p_by_bus = net.res_sgen.groupby(net.sgen.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)
            sgen_q_by_bus = net.res_sgen.groupby(net.sgen.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0)

            # 5. Combine generator types to get total injection per bus
            p_gen = (gen_p_by_bus + sgen_p_by_bus).values
            q_gen = (gen_q_by_bus + sgen_q_by_bus).values
            
            # 6. Calculate the Ybus matrix using our custom function to ensure correct ordering
            ybus_array[t] = calculate_ybus_from_net(net)
            
            # --- END DATA AGGREGATION ---
            
            # Calculate time-varying coefficients for multi-objective evaluation
            renewable_util_frac = current_total_renewable_p_mw / max_total_renewable_mw
            time_carbon_coeffs[t] = config['base_carbon_intensity_grid'] - (renewable_util_frac * config['max_carbon_reduction_from_renewables'])
            time_energy_coeffs[t] = config['max_energy_utilization_coeff'] - (net.res_line.pl_mw.sum() * config['loss_sensitivity'])
            
            # Assemble the ground truth state vector (targets)
            true_state = np.stack([vm_pu, va_rad, p_load, q_load, p_gen, q_gen], axis=1)
            target_matrix[t] = true_state

            # Create noisy measurements for the model features
            meas_vm = true_state[:,0] * (1 + np.random.normal(0, config['voltage_error_std'], num_buses))
            meas_va = true_state[:,1] + np.random.normal(0, config['angle_error_std'], num_buses)
            meas_pl = true_state[:,2] * (1 + np.random.normal(0, config['power_error_std'], num_buses))
            meas_ql = true_state[:,3] * (1 + np.random.normal(0, config['power_error_std'], num_buses))
            meas_pg = true_state[:,4] * (1 + np.random.normal(0, config['power_error_std'], num_buses))
            meas_qg = true_state[:,5] * (1 + np.random.normal(0, config['power_error_std'], num_buses))
            
            feature_matrix[t] = np.stack([meas_vm, meas_va, meas_pl, meas_ql, meas_pg, meas_qg], axis=1)
            pbar.update(1)
            
    return {
        "features": feature_matrix, 
        "targets": target_matrix, 
        "adjacency": adjacency_array, 
        "ybus_matrices": ybus_array, 
        "time_energy_coeffs": time_energy_coeffs, 
        "time_carbon_coeffs": time_carbon_coeffs
    }

    
def save_data(data_dict: dict, case_name: str, renewable_fraction: float, output_dir: str):
    """
    Saves generated data arrays. Multi-dimensional arrays are saved as binary .npy files,
    while 1D coefficient arrays are saved as human-readable .txt files.
    """
    os.makedirs(output_dir, exist_ok=True)
    for key, data in data_dict.items():
        # Create a base filename that includes the case and renewable fraction
        base_filename = f"{case_name}_{key}_frac{renewable_fraction:.1f}"
        
        # Check if the key indicates a coefficient file
        if "coeffs" in key:
            # Save these 1D arrays as .txt files
            filename = os.path.join(output_dir, base_filename + ".txt")
            print(f"Saving coefficient data to '{filename}'...")
            np.savetxt(filename, data)
        else:
            # Save all other multi-dimensional arrays as .npy files
            filename = os.path.join(output_dir, base_filename + ".npy")
            print(f"Saving array data to '{filename}'...")
            np.save(filename, data, allow_pickle=True)

# =============================================================================
# SECTION 4: MAIN EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    for case in CONFIG["test_cases"]:
        try:
            base_net = load_network(case)
            for frac in CONFIG["renewable_fractions_to_run"]:
                print(f"\n{'='*60}\nProcessing {case} with {frac*100:.0f}% renewable fraction\n{'='*60}")
                
                net_for_run = copy.deepcopy(base_net)
                save_case_name = case.replace('bw', '')
                net_for_run.name = f"{save_case_name}_frac{frac:.1f}"
                
                net_with_renewables = configure_renewables(net_for_run, frac, CONFIG)
                generated_data = simulate_time_series(net_with_renewables, CONFIG)
                
                output_path = CONFIG.get("output_dir", os.path.dirname(os.path.abspath(__file__)))
                save_data(generated_data, save_case_name, frac, output_path)

        except Exception as e:
            print(f"\nAn unrecoverable error occurred while processing {case}:")
            traceback.print_exc()
            print("\nSkipping to the next test case.")
            continue
            
    print("\n\nAll data generation processes are complete.")