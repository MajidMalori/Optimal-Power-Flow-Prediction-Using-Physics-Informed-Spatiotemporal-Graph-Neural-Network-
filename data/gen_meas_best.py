# File: data/gen_meas_best.py

import os
import traceback
import json
import warnings

# Suppress numba/pandapower warnings
warnings.filterwarnings('ignore', message='.*numba.*')
warnings.filterwarnings('ignore', category=FutureWarning)

import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top
import numpy as np
import pandas as pd
import networkx as nx
import copy
from datetime import datetime

# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================
CONFIG = {
    "random_seed": 42,  # For reproducibility - set to None for non-deterministic behavior
    "test_cases": ["case33", "case57", "case118"],  # Focus on larger systems since 33-bus is confirmed working
    "time_steps": 10080,  # Default: 420 days (10080 hours) - will be overridden by command-line argument if provided
    "output_dir": "./data", # Base directory - mode-specific subdirectory will be appended
    "renewable_fractions_to_run": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 
    "max_solar_mw": 0.025,  # Per-unit scaling: 2.5% of total load per generator
    "max_wind_mw": 0.04,    # Per-unit scaling: 4% of total load per generator
    "contingency_rate": 0.05,
    "voltage_error_std": 0.005,
    "power_error_std": 0.01,
    "angle_error_std": 0.02,
    "max_energy_utilization_coeff": 0.98,
    "loss_sensitivity": 0.01,
    "base_carbon_intensity_grid": 0.55,
    "max_carbon_reduction_from_renewables": 0.30,
    
    # Time-series generation settings
    "use_time_series": True,  # True: Generate realistic daily cycles, False: Random scenarios (Monte Carlo)
    "hours_per_day": 24,  # Number of hours in a day
    "num_days": None,  # Will be calculated from time_steps and hours_per_day
    
    # Weather-driven renewable variability (NEW FIX for realistic uncertainty)
    "use_weather_driven_renewables": True,  # True: Weather-based variability (realistic), False: Deterministic time-of-day patterns (legacy)
    "seed": 42,  # Random seed for weather simulation (reproducibility)
}

# =============================================================================
# SECTION 2: HELPER FUNCTIONS
# =============================================================================

def get_daily_load_profile(hour: int, season: str = 'summer') -> float:
    """
    Returns realistic hourly load multiplier (0.0-1.0) based on time of day.
    
    Args:
        hour: Hour of day (0-23)
        season: 'summer', 'winter', 'spring', 'fall' (affects shape slightly)
    
    Returns:
        Load multiplier relative to peak demand
    """
    # Base hourly load pattern (typical residential/commercial mix)
    # Values represent percentage of peak load
    hourly_pattern = {
        0: 0.40,   # Midnight - low demand
        1: 0.35,   # 1 AM - lowest demand
        2: 0.33,   # 2 AM
        3: 0.32,   # 3 AM
        4: 0.35,   # 4 AM - starting to rise
        5: 0.42,   # 5 AM - morning ramp begins
        6: 0.55,   # 6 AM - people waking up
        7: 0.70,   # 7 AM - morning peak begins
        8: 0.85,   # 8 AM - high morning demand
        9: 0.90,   # 9 AM - business hours
        10: 0.92,  # 10 AM
        11: 0.95,  # 11 AM - approaching noon
        12: 0.97,  # Noon - high demand
        13: 0.95,  # 1 PM
        14: 0.93,  # 2 PM
        15: 0.92,  # 3 PM
        16: 0.94,  # 4 PM - load rising again
        17: 0.98,  # 5 PM - evening ramp
        18: 1.00,  # 6 PM - PEAK DEMAND (evening peak)
        19: 0.98,  # 7 PM - still high
        20: 0.90,  # 8 PM - starting to decline
        21: 0.80,  # 9 PM
        22: 0.65,  # 10 PM
        23: 0.50,  # 11 PM - winding down
    }
    
    base_load = hourly_pattern.get(hour, 0.5)
    
    # Add small random variation (±5%) for realism
    variation = np.random.uniform(0.95, 1.05)
    
    return base_load * variation


def get_solar_generation_profile(hour: int, day_of_year: int = 180, weather_state: str = None) -> float:
    """
    Returns realistic hourly solar generation multiplier (0.0-1.0) with weather-driven variability.
    
    Args:
        hour: Hour of day (0-23)
        day_of_year: Day of year (1-365, affects sun strength and day length)
        weather_state: 'clear', 'partly_cloudy', 'cloudy', 'storm' (if None, randomly chosen)
    
    Returns:
        Solar generation multiplier (0 = night/storm, 1 = peak clear solar)
    """
    # Solar only during daylight hours (roughly 5 AM to 7 PM)
    if hour < 5 or hour > 19:
        return 0.0  # Night time
    
    # Solar follows a bell curve peaking at noon (geometric/astronomical component)
    hour_from_noon = abs(hour - 12)
    
    if hour_from_noon > 7:
        return 0.0  # Too far from noon
    
    # Bell curve: peak at noon (12), declining towards sunrise/sunset
    # Using cosine function for smooth curve
    solar_angle = (hour - 12) * (np.pi / 14)  # Map to -pi/2 to pi/2
    base_solar = max(0, np.cos(solar_angle))  # 0 at edges, 1 at noon
    
    # Season factor (summer stronger, winter weaker)
    season_factor = 0.85 + 0.15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    # WEATHER-DRIVEN VARIABILITY (replaces fixed 0.7-1.0 range)
    # This is the KEY FIX for realistic renewable uncertainty
    if weather_state is None:
        # If no weather state provided, randomly choose (backwards compatible)
        weather_state = np.random.choice(['clear', 'partly_cloudy', 'cloudy', 'storm'], 
                                        p=[0.3, 0.4, 0.25, 0.05])
    
    # Weather impact on solar generation (much wider range than before!)
    if weather_state == 'clear':
        cloud_factor = np.random.uniform(0.90, 1.0)  # Near full sun
    elif weather_state == 'partly_cloudy':
        cloud_factor = np.random.uniform(0.35, 0.85)  # Highly variable (clouds moving)
    elif weather_state == 'cloudy':
        cloud_factor = np.random.uniform(0.08, 0.35)  # Low but not zero (diffuse light)
    else:  # storm
        cloud_factor = np.random.uniform(0.0, 0.08)   # Near zero (dark clouds)
    
    return base_solar * cloud_factor * season_factor


def get_wind_generation_profile(hour: int, day: int = 0, weather_state: str = None) -> float:
    """
    Returns realistic hourly wind generation multiplier (0.0-1.0) with weather-driven variability.
    
    Args:
        hour: Hour of day (0-23)
        day: Day number (for day-to-day persistence)
        weather_state: 'calm', 'breezy', 'windy', 'storm' (if None, randomly chosen with persistence)
    
    Returns:
        Wind generation multiplier (0 = calm, 1 = maximum wind output)
    """
    # WEATHER-DRIVEN MODEL (replaces fixed time-of-day patterns)
    # Wind is primarily weather-driven, NOT time-driven
    
    if weather_state is None:
        # If no weather state provided, randomly choose (backwards compatible)
        # Use day seed for day-to-day persistence (weather doesn't change every hour)
        day_seed = np.random.RandomState(day)
        weather_state = day_seed.choice(['calm', 'breezy', 'windy', 'storm'], 
                                       p=[0.15, 0.45, 0.30, 0.10])
    
    # Weather impact on wind generation (MUCH wider range than before!)
    if weather_state == 'calm':
        base_wind = np.random.uniform(0.0, 0.20)      # Very low wind
    elif weather_state == 'breezy':
        base_wind = np.random.uniform(0.20, 0.55)     # Moderate wind
    elif weather_state == 'windy':
        base_wind = np.random.uniform(0.55, 0.90)     # High wind
    else:  # storm
        base_wind = np.random.uniform(0.85, 1.0)      # Maximum output (before cutoff)
    
    # Small thermal diurnal effect (realistic but minor compared to weather)
    # Daytime: slight increase due to convective winds
    # Night: slight decrease due to boundary layer stabilization
    thermal_factor = 1.0 + 0.08 * np.sin(2 * np.pi * (hour - 6) / 24)
    
    # Hourly micro-variation (gusts, local effects)
    micro_variation = np.random.uniform(0.85, 1.15)
    
    wind = base_wind * thermal_factor * micro_variation
    
    # Clip to valid range (turbines cut out at very high winds)
    return np.clip(wind, 0.0, 1.0)


def simulate_weather_sequence(timesteps: int, hours_per_day: int = 24, seed: int = None) -> list:
    """
    Simulate realistic weather patterns using Markov chain with persistence.
    Weather states persist for several hours (realistic weather patterns).
    
    Args:
        timesteps: Total number of timesteps to simulate
        hours_per_day: Number of hours per day (for diurnal effects)
        seed: Random seed for reproducibility
    
    Returns:
        List of weather state strings for each timestep
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Weather states for solar (cloud cover)
    solar_states = ['clear', 'partly_cloudy', 'cloudy', 'storm']
    
    # Weather states for wind (wind speed)
    wind_states = ['calm', 'breezy', 'windy', 'storm']
    
    # Transition probability matrix for solar weather (weather persists!)
    # Rows = current state, Columns = next state
    solar_transitions = {
        'clear':         {'clear': 0.65, 'partly_cloudy': 0.30, 'cloudy': 0.05, 'storm': 0.0},
        'partly_cloudy': {'clear': 0.25, 'partly_cloudy': 0.45, 'cloudy': 0.25, 'storm': 0.05},
        'cloudy':        {'clear': 0.10, 'partly_cloudy': 0.30, 'cloudy': 0.50, 'storm': 0.10},
        'storm':         {'clear': 0.0,  'partly_cloudy': 0.10, 'cloudy': 0.40, 'storm': 0.50}
    }
    
    # Transition probability matrix for wind weather
    wind_transitions = {
        'calm':   {'calm': 0.60, 'breezy': 0.30, 'windy': 0.08, 'storm': 0.02},
        'breezy': {'calm': 0.20, 'breezy': 0.50, 'windy': 0.25, 'storm': 0.05},
        'windy':  {'calm': 0.05, 'breezy': 0.30, 'windy': 0.50, 'storm': 0.15},
        'storm':  {'calm': 0.02, 'breezy': 0.10, 'windy': 0.40, 'storm': 0.48}
    }
    
    # Initialize sequences
    solar_sequence = []
    wind_sequence = []
    
    # Start with typical conditions
    current_solar = 'partly_cloudy'
    current_wind = 'breezy'
    
    for t in range(timesteps):
        # Store current states
        solar_sequence.append(current_solar)
        wind_sequence.append(current_wind)
        
        # Transition to next state (hourly changes, but with persistence)
        solar_probs = solar_transitions[current_solar]
        current_solar = np.random.choice(
            list(solar_probs.keys()),
            p=list(solar_probs.values())
        )
        
        wind_probs = wind_transitions[current_wind]
        current_wind = np.random.choice(
            list(wind_probs.keys()),
            p=list(wind_probs.values())
        )
    
    # Return both sequences as tuples
    return [(solar_sequence[i], wind_sequence[i]) for i in range(timesteps)]


def calculate_renewable_reactive_power(p_mw: float, bus_idx: int, net: pp.pandapowerNet, 
                                       use_voltage_control: bool = True) -> float:
    """
    Calculate reactive power for renewable generators based on IEEE 1547 volt-var control.
    Modern inverters adjust reactive power to support voltage regulation.
    
    Args:
        p_mw: Active power generation [MW]
        bus_idx: Bus index where generator is connected
        net: Pandapower network object
        use_voltage_control: If True, use voltage-dependent control; else use fixed power factor
    
    Returns:
        q_mvar: Reactive power [Mvar]
    """
    # If no active power, no reactive power capability
    if p_mw < 1e-6:
        return 0.0
    
    # Inverter reactive power capability (typically ±0.33 * P for modern inverters)
    # This corresponds to power factor range of 0.95 leading to 0.95 lagging
    max_q_capability = 0.33 * p_mw
    
    if use_voltage_control and hasattr(net, 'res_bus') and not net.res_bus.empty:
        # Voltage-dependent reactive power (volt-var control per IEEE 1547-2018)
        # This is how modern grid-tied inverters actually operate
        
        try:
            # Get voltage at generator bus from previous power flow
            v_pu = net.res_bus.loc[bus_idx, 'vm_pu']
            
            # IEEE 1547 volt-var curve (simplified):
            # - V > 1.05 pu: Absorb Q (lower voltage)
            # - V < 0.95 pu: Inject Q (raise voltage)
            # - 0.95 < V < 1.05 pu: Proportional control
            
            # Deadband: ±0.02 pu around nominal (no Q injection in normal range)
            v_deadband_low = 0.98
            v_deadband_high = 1.02
            
            if v_pu < v_deadband_low:
                # Low voltage: Inject reactive power (support voltage)
                # Linear ramp: Max Q at V=0.95, zero Q at V=0.98
                q_factor = min(1.0, (v_deadband_low - v_pu) / 0.03)
                q_mvar = q_factor * max_q_capability  # Positive Q (inject)
            elif v_pu > v_deadband_high:
                # High voltage: Absorb reactive power (reduce voltage)
                # Linear ramp: Max Q at V=1.05, zero Q at V=1.02
                q_factor = min(1.0, (v_pu - v_deadband_high) / 0.03)
                q_mvar = -q_factor * max_q_capability  # Negative Q (absorb)
            else:
                # Normal voltage range: No reactive power support needed
                q_mvar = 0.0
            
            # Apply small random variation (control is not perfect)
            q_mvar *= np.random.uniform(0.95, 1.05)
            
        except (KeyError, AttributeError):
            # Fallback: If voltage not available, use fixed power factor
            # This happens on first timestep before any power flow results exist
            power_factor = 0.98  # Slightly lagging (typical for inverters)
            q_mvar = p_mw * np.tan(np.arccos(power_factor))
    
    else:
        # Fixed power factor mode (fallback or first timestep)
        # Use 0.98 power factor (slightly lagging, typical for inverters)
        power_factor = 0.98
        q_mvar = p_mw * np.tan(np.arccos(power_factor))
        
        # Add small random variation to avoid all generators having identical Q
        q_mvar *= np.random.uniform(0.95, 1.05)
    
    # Clip to inverter capability limits
    return np.clip(q_mvar, -max_q_capability, max_q_capability)


def load_network(case_name: str) -> pp.pandapowerNet:
    """Loads a pandapower network based on its name."""
    print(f"\n----- Loading Base Test Case: {case_name} -----")
    if case_name == "case33": return pn.case33bw()
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

def calculate_adjacency_matrix(net: pp.pandapowerNet) -> np.ndarray:
    """Calculate adjacency matrix from network topology."""
    num_buses = len(net.bus)
    adj_matrix = np.zeros((num_buses, num_buses), dtype=np.float32)
    
    # Add edges from lines
    for _, line in net.line.iterrows():
        from_bus = int(line['from_bus'])
        to_bus = int(line['to_bus'])
        adj_matrix[from_bus, to_bus] = 1.0
        adj_matrix[to_bus, from_bus] = 1.0
    
    return adj_matrix

# =============================================================================
# SECTION 3: SIMULATION AND SAVING
# =============================================================================

def simulate_time_series(net: pp.pandapowerNet, config: dict) -> dict:
    """
    Runs the main time-series power flow simulation with convergence tracking.
    
    Returns:
        Dictionary containing simulation data and convergence statistics
    """
    num_buses = len(net.bus)
    time_steps = config['time_steps']
    
    feature_matrix = np.zeros((time_steps, num_buses, 10))
    target_matrix = np.zeros((time_steps, num_buses, 10))
    adjacency_array = np.zeros((time_steps, num_buses, num_buses))
    time_energy_coeffs = np.zeros(time_steps)
    time_carbon_coeffs = np.zeros(time_steps)
    
    # Note: Generation components are now included in the feature/target matrices
    # No need for separate storage
    
    # Sparse Ybus storage: base + contingencies only
    ybus_base = None  # Base topology Ybus (set on first successful power flow)
    contingency_timesteps = []  # Timesteps where contingencies occurred
    contingency_ybus_list = []  # Ybus matrices for contingency timesteps
    
    # Enhanced convergence tracking for detailed reporting
    convergence_stats = {
        'total_timesteps': time_steps,
        'successful': 0,
        'failed': 0,
        'failed_no_contingency': [],  # Failed with normal topology
        'failed_with_contingency': [],  # Failed with contingency topology
        'contingency_line_details': {},  # Details about which lines caused failures
        'successful_timesteps': [],  # Track which timesteps were successful
        
        # NEW: Resolution tracking
        'resolution_methods': {
            'strict_normal': 0,         # Converged with strict settings (normal topology)
            'strict_contingency': 0,    # Converged with strict settings (contingency topology)
            'relaxed_contingency': 0,   # Had to relax settings during contingency
            'restored_line': 0,         # Had to restore contingency line
        },
        'timestep_resolution': {},  # Per-timestep resolution method
        
        # NEW: Contingency statistics
        'contingencies_attempted': 0,     # Total contingencies tried
        'contingencies_successful': 0,    # Successfully handled
        'contingencies_failed': 0,        # Failed even after restoration (NEW!)
        'contingencies_resolved_strict': 0,   # Handled with strict settings
        'contingencies_resolved_relaxed': 0,  # Required relaxed settings
        'contingencies_restored': 0,      # Too severe, had to restore line
        'critical_lines': {},             # Lines that frequently cause failures
    }
    
    base_load_p, base_load_q = net.load.p_mw.copy(), net.load.q_mvar.copy()
    
    # Calculate realistic renewable capacity based on system load
    total_system_load_mw = base_load_p.sum()
    print(f"Total system load: {total_system_load_mw:.2f} MW")
    
    solar_gens = net.sgen[net.sgen.type == 'solar'] if 'type' in net.sgen.columns else pd.DataFrame()
    wind_gens = net.sgen[net.sgen.type == 'wind'] if 'type' in net.sgen.columns else pd.DataFrame()
    
    # FIXED CAPACITY SCALING (prevents generation-load imbalance)
    # Key insight: Total renewable capacity should be sized to serve peak load, not per-generator
    # At 100% renewable fraction, total capacity should be ~80-90% of peak load (realistic with diversity)
    
    num_solar = len(solar_gens)
    num_wind = len(wind_gens)
    num_total_renewable = num_solar + num_wind
    
    if num_total_renewable > 0:
        # Maximum instantaneous renewable generation should be ~85% of peak load
        # This accounts for: (1) grid stability requirements, (2) spinning reserve, (3) realistic dispatch
        max_total_renewable_mw = total_system_load_mw * 0.85
        
        # Distribute capacity across generators
        # Solar/wind ratio: ~60/40 split (typical renewable portfolio)
        solar_fraction = num_solar / num_total_renewable if num_total_renewable > 0 else 0.5
        wind_fraction = num_wind / num_total_renewable if num_total_renewable > 0 else 0.5
        
        # Individual generator capacity sized so total matches target
        # Solar capacity factor: ~20-25% (accounting for weather, night, etc.)
        # Wind capacity factor: ~30-35% (more consistent but still variable)
        # Inverter sizing: Must handle peak generation (capacity factor ~0.25 for solar, 0.35 for wind)
        
        if num_solar > 0:
            max_individual_solar_mw = (max_total_renewable_mw * solar_fraction * 0.25) / num_solar
        else:
            max_individual_solar_mw = 0
            
        if num_wind > 0:
            max_individual_wind_mw = (max_total_renewable_mw * wind_fraction * 0.35) / num_wind
        else:
            max_individual_wind_mw = 0
        
        print(f"  Renewable generators: {num_solar} solar + {num_wind} wind")
        print(f"  Max individual capacity: Solar={max_individual_solar_mw:.3f} MW, Wind={max_individual_wind_mw:.3f} MW")
        print(f"  Max total renewable capacity: {max_total_renewable_mw:.2f} MW ({max_total_renewable_mw/total_system_load_mw*100:.1f}% of peak load)")
    else:
        max_individual_solar_mw = 0
        max_individual_wind_mw = 0
        max_total_renewable_mw = 1.0  # Avoid division by zero
        print("  No renewable generators configured")
    
    # WEATHER-DRIVEN RENEWABLE GENERATION
    # Generate weather sequence for entire simulation (realistic persistence)
    use_weather_driven = config.get('use_weather_driven_renewables', True)  # Default: ON
    weather_sequence = None
    
    if use_weather_driven and config.get('use_time_series', False):
        print("Simulating weather-driven renewable variability...")
        weather_sequence = simulate_weather_sequence(
            timesteps=time_steps,
            hours_per_day=config.get('hours_per_day', 24),
            seed=config.get('seed', None)
        )
        print(f"   Weather simulation complete: {len(weather_sequence)} timesteps")
    else:
        print("Using legacy deterministic renewable patterns (weather_driven=False)")
        
    dropped_line_idx = None
    has_contingency = False  # Track if current timestep has contingency
    
    # Simulate all timesteps (progress tracked externally by data_validation.py)
    for t in range(time_steps):
        # Restore any previous contingency
        restore_contingency(net, dropped_line_idx)
        dropped_line_idx = None
        has_contingency = False
        
        # Apply a new N-1 contingency based on the configured rate
        if np.random.random() < config['contingency_rate']:
            dropped_line_idx = apply_n1_contingency(net)
            has_contingency = (dropped_line_idx is not None)
            if has_contingency:
                convergence_stats['contingencies_attempted'] += 1

        # ALWAYS calculate adjacency matrix for current topology (even if power flow fails)
        # This ensures data integrity - adjacency reflects actual network state
        current_adjacency_matrix = calculate_adjacency_matrix(net)
        
        # Create the graph adjacency matrix for the current topology
        graph = top.create_nxgraph(net, include_lines=True, include_trafos=True)
        adj_coo = nx.to_scipy_sparse_array(graph, format='coo')
        # Store as dense matrix to avoid object dtype issues
        adjacency_array[t] = current_adjacency_matrix

        # Apply load and generation profiles (time-series or random)
        if config.get('use_time_series', False):
            # TIME-SERIES MODE: Realistic daily patterns
            current_hour = t % config['hours_per_day']
            current_day = t // config['hours_per_day']
            
            # Apply realistic hourly load profile
            load_multiplier = get_daily_load_profile(current_hour)
            net.load.p_mw = base_load_p * load_multiplier
            net.load.q_mvar = base_load_q * load_multiplier
            
            # Apply realistic renewable generation profiles
            # Weather-driven if available, otherwise deterministic
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
                            weather_state=solar_weather  # Weather-driven!
                        )
                        p_gen = solar_profile * max_individual_solar_mw
                    elif sgen.type == 'wind':
                        wind_profile = get_wind_generation_profile(
                            current_hour, 
                            day=current_day,
                            weather_state=wind_weather  # Weather-driven!
                        )
                        p_gen = wind_profile * max_individual_wind_mw
                    
                    # Set active power
                    net.sgen.at[i, 'p_mw'] = p_gen
                    
                    # Calculate reactive power (volt-var control per IEEE 1547)
                    # Modern inverters adjust Q based on voltage to support grid
                    q_gen = calculate_renewable_reactive_power(
                        p_gen, sgen.bus, net, 
                        t > 0  # Use voltage control after first timestep
                    )
                    net.sgen.at[i, 'q_mvar'] = q_gen
                    
                    current_total_renewable_p_mw += p_gen
        else:
            # MONTE CARLO MODE: Random scenarios (original approach)
            net.load.p_mw = base_load_p * np.random.uniform(0.8, 1.2, len(base_load_p))
            net.load.q_mvar = base_load_q * np.random.uniform(0.8, 1.2, len(base_load_q))
            
            current_total_renewable_p_mw = 0
            if 'type' in net.sgen.columns and not net.sgen.empty:
                for i, sgen in net.sgen.iterrows():
                    p_gen = 0
                    if sgen.type == 'solar':
                        p_gen = np.random.uniform(0, max_individual_solar_mw) if 7 <= (t % 24) < 19 else 0
                    elif sgen.type == 'wind':
                        p_gen = np.random.uniform(0, max_individual_wind_mw)
                    
                    # Set active power
                    net.sgen.at[i, 'p_mw'] = p_gen
                    
                    # Calculate reactive power (same as time-series mode)
                    q_gen = calculate_renewable_reactive_power(
                        p_gen, sgen.bus, net, 
                        t > 0  # Use voltage control after first timestep
                    )
                    net.sgen.at[i, 'q_mvar'] = q_gen
                    
                    current_total_renewable_p_mw += p_gen
        
        # === EFFICIENT GUARANTEED CONVERGENCE STRATEGY ===
        # Try contingency first, only fall back if needed
        # This avoids unnecessary power flow calculations
        
        convergence_successful = False
        resolution_method = None
        
        # Try power flow with current topology (contingency or normal)
        try:
            pp.runpp(net, numba=True, enforce_q_lims=True, algorithm='nr', tolerance_mva=1e-8)
            convergence_successful = True
            convergence_stats['successful'] += 1
            convergence_stats['successful_timesteps'].append(t)
            
            # Track resolution method
            if has_contingency:
                resolution_method = 'strict_contingency'
                convergence_stats['resolution_methods']['strict_contingency'] += 1
                convergence_stats['contingencies_successful'] += 1
                convergence_stats['contingencies_resolved_strict'] += 1
            else:
                resolution_method = 'strict_normal'
                convergence_stats['resolution_methods']['strict_normal'] += 1
            
        except pp.LoadflowNotConverged:
            # CONTINGENCY FALLBACK: If failed during contingency, try relaxed settings
            if has_contingency and dropped_line_idx is not None:
                # Track critical line
                line_key = f"line_{dropped_line_idx}"
                if line_key not in convergence_stats['critical_lines']:
                    convergence_stats['critical_lines'][line_key] = {
                        'line_id': int(dropped_line_idx),
                        'failure_count': 0,
                        'resolution_methods': {'relaxed': 0, 'restored': 0}
                    }
                convergence_stats['critical_lines'][line_key]['failure_count'] += 1
                
                try:
                    # Relaxed settings for contingency (realistic - grid under stress)
                    pp.runpp(net, numba=False, enforce_q_lims=False, algorithm='nr', 
                            tolerance_mva=1e-6, max_iteration=20)
                    convergence_successful = True
                    convergence_stats['successful'] += 1
                    convergence_stats['successful_timesteps'].append(t)
                    
                    # Track resolution method
                    resolution_method = 'relaxed_contingency'
                    convergence_stats['resolution_methods']['relaxed_contingency'] += 1
                    convergence_stats['contingencies_successful'] += 1
                    convergence_stats['contingencies_resolved_relaxed'] += 1
                    convergence_stats['critical_lines'][line_key]['resolution_methods']['relaxed'] += 1
                    
                except pp.LoadflowNotConverged:
                    # Contingency too severe - restore line and run without contingency
                    restore_contingency(net, dropped_line_idx)
                    dropped_line_idx = None
                    has_contingency = False
                    convergence_stats['critical_lines'][line_key]['resolution_methods']['restored'] += 1
                    
                    try:
                        # Retry with normal topology
                        pp.runpp(net, numba=True, enforce_q_lims=True, algorithm='nr', tolerance_mva=1e-8)
                        convergence_successful = True
                        convergence_stats['successful'] += 1
                        convergence_stats['successful_timesteps'].append(t)
                        
                        # Track resolution method
                        resolution_method = 'restored_line'
                        convergence_stats['resolution_methods']['restored_line'] += 1
                        convergence_stats['contingencies_restored'] += 1
                        
                    except pp.LoadflowNotConverged:
                        # Even normal topology failed! This is a fundamental problem
                        # This should NEVER happen with proper capacity scaling
                        convergence_stats['failed'] += 1
                        convergence_stats['failed_no_contingency'].append(t)
                        convergence_stats['contingencies_failed'] += 1
                        resolution_method = 'failed_completely'
                        
                        print(f"  ERROR: Timestep {t} failed completely (even normal topology!)")
                        print(f"         This indicates a problem with load/generation balance")
                        print(f"         Total load: {net.load.p_mw.sum():.1f} MW, Total renewable: {current_total_renewable_p_mw:.1f} MW")
                        
                        # Skip this timestep - but this should be extremely rare!
                        continue
            else:
                # Failed without contingency - this shouldn't happen with fixed capacity
                convergence_stats['failed'] += 1
                convergence_stats['failed_no_contingency'].append(t)
                resolution_method = 'failed'
                print(f"  WARNING: Timestep {t} failed (no contingency), skipping")
                continue
        
        # Store resolution method for this timestep
        if resolution_method:
            convergence_stats['timestep_resolution'][str(t)] = resolution_method
        
        # --- START DATA AGGREGATION (CONSISTENT 0 to N-1 ORDERING) ---
        
        # 1. Get bus voltages and angles. These are already ordered correctly by net.bus.index.
        vm_pu = net.res_bus.vm_pu.values
        va_rad = np.deg2rad(net.res_bus.va_degree.values)
        
        # 2. Aggregate loads. `reindex` ensures we have a value for every bus, in order.
        load_p_by_bus = net.res_load.groupby(net.load.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)
        load_q_by_bus = net.res_load.groupby(net.load.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0)
        p_load = load_p_by_bus.values
        q_load = load_q_by_bus.values

        # 3. Aggregate slack bus (external grid) generation - THE MAIN POWER SOURCE!
        ext_grid_p_by_bus = net.res_ext_grid.groupby(net.ext_grid.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)
        ext_grid_q_by_bus = net.res_ext_grid.groupby(net.ext_grid.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0)
        
        # 4. Aggregate conventional generators
        gen_p_by_bus = net.res_gen.groupby(net.gen.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)
        gen_q_by_bus = net.res_gen.groupby(net.gen.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0)

        # 5. Aggregate static (renewable) generators
        sgen_p_by_bus = net.res_sgen.groupby(net.sgen.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)
        sgen_q_by_bus = net.res_sgen.groupby(net.sgen.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0)

        # Note: Generation components are now stored in the feature/target matrices
        
        # 7. Combine ALL generator types to get total injection per bus
        p_gen = (ext_grid_p_by_bus + gen_p_by_bus + sgen_p_by_bus).values
        q_gen = (ext_grid_q_by_bus + gen_q_by_bus + sgen_q_by_bus).values
        
        # 7. Calculate Ybus matrix (sparse storage: only base + contingencies)
        current_ybus = calculate_ybus_from_net(net)
        
        if ybus_base is None:
            # First successful power flow - store as base Ybus
            ybus_base = current_ybus.copy()
        elif has_contingency:
            # Topology changed due to contingency - store this variant
            contingency_timesteps.append(t)
            contingency_ybus_list.append(current_ybus.copy())
        # else: Normal topology, same as base - no need to store
        
        # --- END DATA AGGREGATION ---
        
        # Calculate time-varying coefficients for multi-objective evaluation
        renewable_util_frac = current_total_renewable_p_mw / max_total_renewable_mw
        time_carbon_coeffs[t] = config['base_carbon_intensity_grid'] - (renewable_util_frac * config['max_carbon_reduction_from_renewables'])
        time_energy_coeffs[t] = config['max_energy_utilization_coeff'] - (net.res_line.pl_mw.sum() * config['loss_sensitivity'])
        
        # Assemble the ground truth state vector (targets) with separated generation components
        true_state = np.stack([vm_pu, va_rad, p_load, q_load, 
                              ext_grid_p_by_bus.values, ext_grid_q_by_bus.values,  # Slack bus generation
                              gen_p_by_bus.values, gen_q_by_bus.values,            # Conventional generation
                              sgen_p_by_bus.values, sgen_q_by_bus.values], axis=1) # Renewable generation
        target_matrix[t] = true_state

        # Create noisy measurements for the model features with separated generation components
        # Use positive noise for ALL values to preserve original information and prevent sign changes
        
        # Generate positive noise for all values
        positive_noise_vm = np.abs(np.random.normal(0, config['voltage_error_std'], num_buses))
        positive_noise_angle = np.abs(np.random.normal(0, config['angle_error_std'], num_buses))
        positive_noise_power = np.abs(np.random.normal(0, config['power_error_std'], num_buses))
        
        # Voltage magnitude: positive noise ensures non-negative result
        meas_vm = true_state[:,0] * (1 + positive_noise_vm)
        
        # Voltage angle: positive noise preserves sign and magnitude relationship
        meas_va = true_state[:,1] * (1 + positive_noise_angle)
        
        # Loads: positive noise ensures non-negative result
        meas_pl = true_state[:,2] * (1 + positive_noise_power)
        meas_ql = true_state[:,3] * (1 + positive_noise_power)
        
        # External grid: positive noise preserves sign and magnitude relationship
        meas_p_ext = true_state[:,4] * (1 + positive_noise_power)
        meas_q_ext = true_state[:,5] * (1 + positive_noise_power)
        
        # Conventional generation: positive noise ensures non-negative result
        meas_p_conv = true_state[:,6] * (1 + positive_noise_power)
        meas_q_conv = true_state[:,7] * (1 + positive_noise_power)
        
        # Renewable generation: positive noise ensures non-negative result
        meas_p_ren = true_state[:,8] * (1 + positive_noise_power)
        meas_q_ren = true_state[:,9] * (1 + positive_noise_power)
        
        feature_matrix[t] = np.stack([meas_vm, meas_va, meas_pl, meas_ql, 
                                     meas_p_ext, meas_q_ext,           # Slack bus generation
                                     meas_p_conv, meas_q_conv,         # Conventional generation
                                     meas_p_ren, meas_q_ren], axis=1)   # Renewable generation
        
        # Adjacency matrix already stored above
    
    # Finalize convergence statistics
    convergence_stats['success_rate'] = (convergence_stats['successful'] / time_steps * 100) if time_steps > 0 else 0
    
    # Handle edge case: no successful power flow (very rare)
    if ybus_base is None:
        ybus_base = np.zeros((num_buses, num_buses), dtype=np.complex128)
        convergence_stats['ybus_fallback_used'] = True
    else:
        convergence_stats['ybus_fallback_used'] = False
    
    # Prepare sparse Ybus data structure
    ybus_data = {
        "base": ybus_base,
        "contingency_timesteps": np.array(contingency_timesteps, dtype=np.int32),
        "contingency_matrices": np.array(contingency_ybus_list) if contingency_ybus_list else np.array([]).reshape(0, num_buses, num_buses).astype(np.complex128)
    }
    
    # Apply truncation to ensure consistent shapes across all renewable fractions
    # This will be handled in the main execution block after all scenarios are generated
    
    return {
        "features": feature_matrix, 
        "targets": target_matrix, 
        "adjacency": adjacency_array, 
        "ybus_data": ybus_data,  # Sparse format
        "time_energy_coeffs": time_energy_coeffs, 
        "time_carbon_coeffs": time_carbon_coeffs,
        "convergence_stats": convergence_stats  # Detailed convergence report
        # Note: Generation components are now included in features/targets matrices
    }

    

def save_data(data_dict: dict, case_name: str, renewable_fraction: float, output_dir: str, timestamp: str = None):
    """
    Saves generated data arrays with support for sparse Ybus format and convergence reports.
    
    Args:
        data_dict: Dictionary containing data arrays
        case_name: Name of the test case (e.g., 'case33')
        renewable_fraction: Renewable energy fraction
        output_dir: Directory to save files
        timestamp: Optional timestamp string to ensure data consistency
    """
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for key, data in data_dict.items():
        # Handle sparse Ybus data specially
        if key == "ybus_data":
            # Save each component of the sparse Ybus separately
            for sub_key, sub_data in data.items():
                sub_filename = f"{case_name}_ybus_{sub_key}_frac{renewable_fraction:.1f}_{timestamp}.npy"
                filepath = os.path.join(output_dir, sub_filename)
                print(f"Saving Ybus component '{sub_key}' to '{filepath}'...")
                np.save(filepath, sub_data, allow_pickle=False)
            continue
        
        # Handle convergence statistics specially
        if key == "convergence_stats":
            stats_filename = f"{case_name}_convergence_report_frac{renewable_fraction:.1f}_{timestamp}.json"
            filepath = os.path.join(output_dir, stats_filename)
            print(f"Saving convergence report to '{filepath}'...")
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            continue
        
        # Create a base filename that includes the case, renewable fraction, and timestamp
        base_filename = f"{case_name}_{key}_frac{renewable_fraction:.1f}_{timestamp}"
        
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
    import sys
    import random
    
    # Set random seeds for reproducibility
    if CONFIG["random_seed"] is not None:
        print(f"\nSetting random seed: {CONFIG['random_seed']} (for reproducibility)")
        np.random.seed(CONFIG["random_seed"])
        random.seed(CONFIG["random_seed"])
        # Note: pandapower uses numpy's random state internally, so np.random.seed() covers it
    else:
        print("\nWARNING: No random seed set - results will not be reproducible!")
    
    # Parse command-line arguments for data mode and timesteps
    data_mode = 'train'  # Default
    timesteps = None  # Will use CONFIG default or mode-specific default
    
    if len(sys.argv) > 1:
        data_mode = sys.argv[1].lower()
        if data_mode not in ['train', 'test']:
            print(f"ERROR: Invalid data_mode '{data_mode}'. Use 'train' or 'test'.")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            timesteps = int(sys.argv[2])
        except ValueError:
            print(f"ERROR: Invalid timesteps '{sys.argv[2]}'. Must be an integer.")
            sys.exit(1)
    
    # Set mode-specific defaults if timesteps not provided
    # Use values that result in COMPLETE 24-hour days for 60/20/20 split
    # Total days must be divisible by 5 for clean splits
    if timesteps is None:
        timesteps = 10080 if data_mode == 'train' else 1080  # 420 days (train) or 45 days (test)
    
    CONFIG['time_steps'] = timesteps
    
    # Generate a single timestamp for this entire data generation run
    generation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nStarting data generation [{data_mode.upper()} MODE - {timesteps} timesteps]")
    print(f"Timestamp: {generation_timestamp}")
    print("All files will be tagged with this timestamp to ensure data consistency.")
    
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
                
                # Ensure we save to the generation_mode/data_mode directory structure
                # Structure: data/monte_carlo/train or data/time_series/test
                generation_mode = 'time_series' if CONFIG.get('use_time_series', False) else 'monte_carlo'
                
                script_dir = os.path.dirname(os.path.abspath(__file__))
                if "data" in script_dir:
                    # Script is being run from data/ subdirectory
                    output_path = os.path.join(script_dir, generation_mode, data_mode)
                else:
                    # Script is being run from main directory
                    output_path = os.path.join(script_dir, "data", generation_mode, data_mode)
                
                # Create directory if it doesn't exist
                os.makedirs(output_path, exist_ok=True)
                    
                # Pass the generation timestamp to ensure all files have the same timestamp
                save_data(generated_data, save_case_name, frac, output_path, generation_timestamp)

        except Exception as e:
            print(f"\nAn unrecoverable error occurred while processing {case}:")
            traceback.print_exc()
            print("\nSkipping to the next test case.")
            continue
    
    # Write metadata file for smart data detection
    try:
        generation_mode = 'time_series' if CONFIG.get('use_time_series', False) else 'monte_carlo'
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if "data" in script_dir:
            output_path = os.path.join(script_dir, generation_mode, data_mode)
        else:
            output_path = os.path.join(script_dir, "data", generation_mode, data_mode)
        
        os.makedirs(output_path, exist_ok=True)
        
        metadata = {
            'generation_mode': 'time_series' if CONFIG.get('use_time_series', False) else 'monte_carlo',
            'data_mode': data_mode,
            'timesteps': timesteps,
            'timestamp': generation_timestamp,
            'hours_per_day': CONFIG.get('hours_per_day', 24),
            'use_time_series': CONFIG.get('use_time_series', False),
            'test_cases': CONFIG["test_cases"],
            'renewable_fractions': CONFIG["renewable_fractions_to_run"],
            'generation_date': datetime.now().isoformat()
        }
        
        metadata_file = os.path.join(output_path, "data_generation_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n[Metadata] Saved generation metadata to: {metadata_file}")
        print(f"  Mode: {metadata['generation_mode']}")
        print(f"  Data type: {metadata['data_mode']}")
        print(f"  Timesteps: {metadata['timesteps']}")
    except Exception as e:
        print(f"\n[Warning] Could not save metadata file: {e}")
            
    print("\n\nAll data generation processes are complete.")