import os
import traceback
import json
import warnings
import sys

# Add parent directory to Python path so we can import utils
# This is needed when gen_meas_best.py is run as a script from the data/ directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import DataGenerationError from utils (shared exception)
from utils.contingency_ybus import DataGenerationError

# Suppress numba/pandapower warnings
warnings.filterwarnings('ignore', message='.*numba.*')
warnings.filterwarnings('ignore', message='.*Please install numba.*')
warnings.filterwarnings('ignore', message='.*numba cannot be imported.*')
warnings.filterwarnings('ignore', message='.*Probably the execution is slow.*')
warnings.filterwarnings('ignore', category=FutureWarning)

# Context manager to suppress print statements during pandapower operations
class SuppressPrints:
    """Context manager to suppress stdout print statements."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._devnull = open(os.devnull, 'w')
        sys.stdout = self._devnull
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        self._devnull.close()

# Import pandapower (suppress any print statements during import)
with SuppressPrints():
    import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top
import numpy as np
import pandas as pd
import networkx as nx
import copy
from datetime import datetime

# SECTION 1: CONFIGURATION
CONFIG = {
    "random_seed": 42,  # For reproducibility - set to None for non-deterministic behavior
    "test_cases": ["case33", "case57", "case118"],  # Focus on larger systems since 33-bus is confirmed working
    "time_steps": 10080,  # Default: 420 days (10080 hours) - will be overridden by command-line argument if provided
    "output_dir": "./data", # Base directory - mode-specific subdirectory will be appended
    "renewable_fractions_to_run": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 
    # NOTE: max_solar_mw and max_wind_mw removed - capacity is calculated dynamically
    # based on total system load and renewable fraction (see simulate_time_series)
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
    
    # Weather-driven renewable variability
    "use_weather_driven_renewables": True,  # True: Weather-based variability (realistic), False: Deterministic time-of-day patterns (legacy)
    "seed": 42,  # Random seed for weather simulation (reproducibility)
    
    # Memory optimization: Chunked writing
    "chunk_size": 1000,  # Write data in chunks of this many timesteps (reduces RAM usage)
    "use_chunked_writing": True,  # Enable chunked writing to disk (memory efficient for large datasets)
}

# SECTION 2: HELPER FUNCTIONS

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
    # Also handle very small power to avoid numerical precision issues
    if p_mw < 1e-5:  # Less than 0.00001 MW (0.01 kW) - too small for meaningful reactive power
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
            # BUT: Ensure variation respects physical limits
            # Calculate bounds for random variation to stay within limits
            if abs(q_mvar) > 1e-6:  # Non-zero q_mvar
                # Calculate max allowed multiplier to stay within limits
                max_multiplier = max_q_capability / abs(q_mvar)
                # Random variation between 0.95 and min(1.05, max_multiplier)
                # This ensures we never exceed the physical limit
                variation_range = min(1.05, max_multiplier * 0.99)  # 99% of max to leave small safety margin
                variation = np.random.uniform(0.95, variation_range)
                q_mvar *= variation
            else:
                # q_mvar is near zero, apply normal variation (won't exceed limits)
                q_mvar *= np.random.uniform(0.95, 1.05)
            
        except (KeyError, AttributeError):
            # Fallback: If voltage not available, use fixed power factor
            # This happens on first timestep before any power flow results exist
            power_factor = 0.98  # Slightly lagging (typical for inverters)
            q_mvar = p_mw * np.tan(np.arccos(power_factor))
            # Apply bounded random variation
            if abs(q_mvar) > 1e-6:
                max_multiplier = max_q_capability / abs(q_mvar)
                variation_range = min(1.05, max_multiplier * 0.99)
                variation = np.random.uniform(0.95, variation_range)
                q_mvar *= variation
            else:
                q_mvar *= np.random.uniform(0.95, 1.05)
    
    else:
        # Fixed power factor mode (fallback or first timestep)
        # Use 0.98 power factor (slightly lagging, typical for inverters)
        power_factor = 0.98
        q_mvar = p_mw * np.tan(np.arccos(power_factor))
        
        # Add small random variation to avoid all generators having identical Q
        # BUT: Ensure variation respects physical limits
        if abs(q_mvar) > 1e-6:
            max_multiplier = max_q_capability / abs(q_mvar)
            variation_range = min(1.05, max_multiplier * 0.99)
            variation = np.random.uniform(0.95, variation_range)
            q_mvar *= variation
        else:
            q_mvar *= np.random.uniform(0.95, 1.05)
    
    # Final safety check: Clip to physical limits (inverters cannot exceed these limits)
    q_mvar_clipped = np.clip(q_mvar, -max_q_capability, max_q_capability)
    
    # Check if significant clipping occurred (indicates a logic error, not just random variation)
    # With bounded random variation, clipping should be minimal (< 0.1% of capability)
    # Skip check if capability is too small (numerical precision issues)
    if max_q_capability >= 1e-5:  # Only check if capability is meaningful (> 0.00001 Mvar)
        clipping_amount = abs(q_mvar) - abs(q_mvar_clipped)
        if clipping_amount > max_q_capability * 0.001:  # More than 0.1% of capability clipped
            raise DataGenerationError(
                f"SEVERE ERROR: Inverter reactive power limit violation: q_mvar={q_mvar:.6f} Mvar exceeds "
                f"capability ±{max_q_capability:.6f} Mvar (based on p_mw={p_mw:.6f} MW) at bus {bus_idx}. "
                f"Clipping amount: {clipping_amount:.6f} Mvar ({clipping_amount/max_q_capability*100:.2f}% of capability). "
                f"This indicates a control logic error in calculate_renewable_reactive_power. "
                f"Check voltage control parameters or power factor calculations. "
                f"Data generation STOPPED - cannot generate valid data with invalid control logic."
            )
    
    # Log minor clipping (within tolerance, but still worth noting)
    if abs(q_mvar_clipped) < abs(q_mvar) - 1e-6:
        import warnings
        warnings.warn(
            f"Inverter reactive power clipped: requested {q_mvar:.6f} Mvar, "
            f"limited to {q_mvar_clipped:.6f} Mvar (capability: ±{max_q_capability:.6f} Mvar). "
            f"Bus {bus_idx}, P={p_mw:.6f} MW.",
            UserWarning
        )
    
    return q_mvar_clipped


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
        
        # NO FALLBACK: Zero impedance is unphysical (superconductor) and indicates data corruption
        if abs(z_series) < 1e-10:  # Check for near-zero (accounting for floating point)
            raise DataGenerationError(
                f"SEVERE ERROR: Line {line.name} (index {line.name}) has zero or near-zero impedance: "
                f"r={r_ohm:.6e} ohm, x={x_ohm:.6e} ohm, z={z_series:.6e} ohm. "
                f"This is unphysical and indicates network data corruption. "
                f"Check line parameters: r_ohm_per_km={line.r_ohm_per_km}, length_km={line.length_km}. "
                f"Data generation STOPPED - cannot generate valid data with corrupted network."
            )
        
        y_series = 1.0 / z_series
        
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

def identify_bus_types(net: pp.pandapowerNet) -> np.ndarray:
    """
    Identify bus types for Optimal Power Flow (OPF) from pandapower network state.
    Bus types are determined AFTER power flow solution (pandapower decides).
    
    - Slack bus: Bus with ext_grid (reference bus, V and θ known/specified)
    - PV bus: Bus with gen (generator with voltage control, V known, P specified)
    - PQ bus: Bus with load or sgen only (load bus, V and θ unknown)
    
    Note: A bus type can change dynamically (e.g., if gen hits Q limits, becomes PQ),
    but we use static classification based on network elements for simplicity.
    
    Returns:
        bus_types: Array of bus type codes [0=PQ, 1=PV, 2=Slack] for each bus
    """
    num_buses = len(net.bus)
    bus_types = np.zeros(num_buses, dtype=np.int32)  # Default: PQ bus
    
    # Identify slack buses (external grid) - these are always slack
    slack_buses = set(net.ext_grid.bus.values)
    for bus_idx in slack_buses:
        bus_types[bus_idx] = 2  # Slack bus
    
    # Identify PV buses (conventional generators with voltage control)
    # PV buses have gen connected (not ext_grid, not just load/sgen)
    # but we use static classification for training data consistency
    gen_buses = set(net.gen.bus.values)
    for bus_idx in gen_buses:
        if bus_idx not in slack_buses:  # Don't override slack
            bus_types[bus_idx] = 1  # PV bus
    
    return bus_types

def create_opf_targets(net: pp.pandapowerNet, bus_types: np.ndarray) -> np.ndarray:
    """
    Create OPF-style targets based on bus type (predict only unknowns):
    - PQ bus: Predict [V, θ] (unknowns)
    - PV bus: Predict [Q, θ] (unknowns, V is known/specified)
    - Slack bus: Predict [P, Q] (unknowns, V and θ are known/specified)
    
    All targets are normalized to per-unit for consistent scaling:
    - V: Already in per-unit (vm_pu)
    - θ: In radians (typically -0.5 to 0.5)
    - P, Q: Converted to per-unit by dividing by net.sn_mva
    
    Args:
        net: Pandapower network AFTER power flow solution
        bus_types: Array of bus type codes [0=PQ, 1=PV, 2=Slack] from identify_bus_types()
    
    Returns:
        targets: Array [num_buses, 2] with unknowns for each bus (all in consistent units)
    """
    num_buses = len(net.bus)
    targets = np.zeros((num_buses, 2), dtype=np.float64)
    
    # Get system base power (MVA) for per-unit conversion
    s_base_mva = net.sn_mva  # Base power in MVA
    
    # Get power flow results
    vm_pu = net.res_bus.vm_pu.values
    va_rad = np.deg2rad(net.res_bus.va_degree.values)
    
    # Get power injections (net injection = generation - load)
    ext_grid_p_by_bus = net.res_ext_grid.groupby(net.ext_grid.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)
    ext_grid_q_by_bus = net.res_ext_grid.groupby(net.ext_grid.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0)
    gen_p_by_bus = net.res_gen.groupby(net.gen.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)
    gen_q_by_bus = net.res_gen.groupby(net.gen.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0)
    sgen_p_by_bus = net.res_sgen.groupby(net.sgen.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)
    sgen_q_by_bus = net.res_sgen.groupby(net.sgen.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0)
    load_p_by_bus = net.res_load.groupby(net.load.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)
    load_q_by_bus = net.res_load.groupby(net.load.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0)
    
    # Net power injection at each bus (generation - load) in MW/MVar
    p_inj_mw = (ext_grid_p_by_bus + gen_p_by_bus + sgen_p_by_bus - load_p_by_bus).values
    q_inj_mvar = (ext_grid_q_by_bus + gen_q_by_bus + sgen_q_by_bus - load_q_by_bus).values
    
    # Convert power to per-unit for consistent normalization
    p_inj_pu = p_inj_mw / s_base_mva
    q_inj_pu = q_inj_mvar / s_base_mva
    
    # Create targets based on bus type (only unknowns, all in consistent units)
    for bus_idx in range(num_buses):
        if bus_types[bus_idx] == 0:  # PQ bus: unknowns = [V, θ]
            targets[bus_idx, 0] = vm_pu[bus_idx]  # Already in per-unit
            targets[bus_idx, 1] = va_rad[bus_idx]  # In radians
        elif bus_types[bus_idx] == 1:  # PV bus: unknowns = [Q, θ]
            targets[bus_idx, 0] = q_inj_pu[bus_idx]  # Reactive power in per-unit
            targets[bus_idx, 1] = va_rad[bus_idx]  # Voltage angle in radians
        else:  # Slack bus: unknowns = [P, Q]
            targets[bus_idx, 0] = p_inj_pu[bus_idx]  # Active power in per-unit
            targets[bus_idx, 1] = q_inj_pu[bus_idx]  # Reactive power in per-unit
    
    return targets


# SECTION 2.5: VALIDATION AND CURTAILMENT FUNCTIONS

def validate_power_flow_inputs(net: pp.pandapowerNet) -> tuple[bool, str]:
    """
    PRE-POWER-FLOW VALIDATION: Check inputs before running power flow.
    Catches physically impossible inputs early (avoids wasted computation).
    
    Returns:
        (is_valid, reason): True if valid, False with reason if invalid
    """
    # 1. Generator capacity check (hard limit - P_gen must be <= P_rated)
    if not net.gen.empty:
        gen_p_mw = net.gen.p_mw.values
        gen_max_mw = net.gen.max_p_mw.values
        # Vectorized check: any generator exceeding capacity?
        if np.any(gen_p_mw > gen_max_mw * 1.01):  # 1% tolerance for floating point
            violating_idx = np.where(gen_p_mw > gen_max_mw * 1.01)[0]
            max_violation = ((gen_p_mw - gen_max_mw) / gen_max_mw * 100)[violating_idx].max()
            return False, f"Generator capacity violation: {max_violation:.1f}% over limit at gen {violating_idx[0]}"
    
    # 2. Inverter capability check (hard limit - P² + Q² must be <= S_rated)
    # Standard Pandapower field: sn_mva = Rated Apparent Power (S_rated)
    if not net.sgen.empty and 'sn_mva' in net.sgen.columns:
        sgen_p_mw = net.sgen.p_mw.values
        sgen_q_mvar = net.sgen.q_mvar.values
        sgen_sn_mva = net.sgen.sn_mva.values
        
        # Only check generators where sn_mva is defined (> 0)
        valid_rating = sgen_sn_mva > 0
        
        if np.any(valid_rating):
            # Calculate apparent power (vectorized)
            sgen_s_mva = np.sqrt(sgen_p_mw[valid_rating]**2 + sgen_q_mvar[valid_rating]**2)
            ratings = sgen_sn_mva[valid_rating]
            
            # Check limit: P² + Q² ≤ S_rated (with 1% tolerance for floating point)
            if np.any(sgen_s_mva > ratings * 1.01):
                violating_idx = np.where(sgen_s_mva > ratings * 1.01)[0]
                max_violation = ((sgen_s_mva - ratings) / ratings * 100)[violating_idx].max()
                return False, f"Inverter capability violation: {max_violation:.1f}% over rated S (sn_mva)"
    
    # 3. Load sanity check (loads should be positive and reasonable)
    if not net.load.empty:
        load_p_mw = net.load.p_mw.values
        # Vectorized check: any negative loads?
        if np.any(load_p_mw < -1e-6):  # Allow tiny negative for numerical precision
            return False, f"Negative load detected: min={load_p_mw.min():.3f} MW"
        
        # Check for unreasonably large loads (more than 10x base load)
        if len(load_p_mw) > 0:
            base_load_avg = np.mean(np.abs(load_p_mw))
            if np.any(np.abs(load_p_mw) > base_load_avg * 10):
                return False, f"Unreasonably large load detected: max={load_p_mw.max():.1f} MW (avg: {base_load_avg:.1f} MW)"
    
    return True, "Input validation passed"


def validate_power_flow_outputs(net: pp.pandapowerNet, convergence_stats: dict) -> tuple[bool, str, dict]:
    """
    POST-POWER-FLOW VALIDATION: Check outputs after power flow.
    Separates "Valid Stressed States" (keep) from "Numerical Garbage" (discard).
    
    Returns:
        (is_valid, reason, violation_flags): 
            - is_valid: True if physically possible (even if stressed), False if garbage
            - reason: Description of why invalid (if garbage) or what violations exist (if valid but stressed)
            - violation_flags: Dict with flags for each type of violation (for statistics)
    """
    violation_flags = {
        'voltage_violation': False,
        'angle_violation': False,
        'line_loading_violation': False,
        'slack_power_violation': False,
        'generator_capacity_violation': False,
        'inverter_capability_violation': False,
    }
    
    # 1. Convergence check (already done, but double-check)
    if not net.converged:
        return False, "Power flow did not converge", violation_flags
    
    # 2. Voltage sanity check (vectorized) - HARD LIMIT for garbage detection
    vm_pu = net.res_bus.vm_pu.values
    if np.any(vm_pu < 0.5) or np.any(vm_pu > 1.5):
        min_v = vm_pu.min()
        max_v = vm_pu.max()
        return False, f"Voltage out of physical bounds: min={min_v:.3f}, max={max_v:.3f} p.u. (garbage)", violation_flags
    
    # Check for operational violations (0.85-1.15 p.u. is stressed but valid)
    if np.any(vm_pu < 0.85) or np.any(vm_pu > 1.15):
        violation_flags['voltage_violation'] = True
    
    # 3. Angle difference check (check across all lines) - HARD LIMIT for garbage detection
    va_rad = np.deg2rad(net.res_bus.va_degree.values)
    max_angle_diff = 0.0
    
    # Vectorized approach: compute all angle differences at once
    if not net.line.empty and net.line[net.line.in_service].shape[0] > 0:
        active_lines = net.line[net.line.in_service]
        from_buses = active_lines.from_bus.values.astype(int)
        to_buses = active_lines.to_bus.values.astype(int)
        
        # Vectorized angle difference calculation
        angle_diffs = np.abs(va_rad[from_buses] - va_rad[to_buses])
        max_angle_diff = np.max(angle_diffs)
        
        if max_angle_diff > np.pi / 2:  # 90 degrees - HARD LIMIT (garbage)
            return False, f"Angle difference exceeds stability limit: {np.rad2deg(max_angle_diff):.1f}° (garbage)", violation_flags
        
        # Check for operational violations (45° is stressed but valid)
        if max_angle_diff > np.deg2rad(45):  # 45 degrees
            violation_flags['angle_violation'] = True
    
    # 4. Slack bus power sanity check (system-dependent) - HARD LIMIT for garbage detection
    if not net.res_ext_grid.empty:
        slack_p_mw = net.res_ext_grid.p_mw.values
        max_load_mw = net.load.p_mw.sum() if not net.load.empty else 0.0
        
        # System-dependent threshold: 5x max load is unrealistic
        threshold_mw = 5 * max_load_mw if max_load_mw > 0 else 10000.0
        
        # Vectorized check
        if np.any(np.abs(slack_p_mw) > threshold_mw):
            max_slack = np.abs(slack_p_mw).max()
            return False, f"Slack bus power unrealistic: {max_slack:.1f} MW (threshold: {threshold_mw:.1f} MW, max load: {max_load_mw:.1f} MW) (garbage)", violation_flags
        
        # Check for operational violations (2x max load is stressed but valid)
        if np.any(np.abs(slack_p_mw) > 2 * max_load_mw):
            violation_flags['slack_power_violation'] = True
    
    # 5. Line loading sanity check (catch numerical singularities) - HARD LIMIT for garbage detection
    if not net.res_line.empty:
        # Vectorized calculation of line loading percentage
        line_current_ka = net.res_line.i_ka.values
        line_max_i_ka = net.line.max_i_ka.values
        
        # Avoid division by zero
        valid_lines = line_max_i_ka > 1e-6
        if np.any(valid_lines):
            line_loading_pct = np.zeros_like(line_current_ka)
            line_loading_pct[valid_lines] = (line_current_ka[valid_lines] / line_max_i_ka[valid_lines]) * 100
            
            # Vectorized check for garbage
            if np.any(line_loading_pct > 1000):
                max_loading = line_loading_pct.max()
                return False, f"Line loading indicates numerical singularity: max={max_loading:.1f}% (garbage)", violation_flags
            
            # Check for operational violations (>100% is stressed but valid)
            if np.any(line_loading_pct > 100):
                violation_flags['line_loading_violation'] = True
    
    # 6. Generator capacity check (hard limit - should never be violated after power flow)
    if not net.res_gen.empty:
        gen_p_mw = net.res_gen.p_mw.values
        gen_max_mw = net.gen.max_p_mw.values
        
        # Vectorized check
        if np.any(gen_p_mw > gen_max_mw * 1.01):  # 1% tolerance for floating point
            violation_flags['generator_capacity_violation'] = True
            # This is a hard limit violation - log but don't discard (pandapower should enforce this)
            max_violation = ((gen_p_mw - gen_max_mw) / gen_max_mw * 100).max()
            return False, f"Generator exceeds rated capacity: {max_violation:.1f}% over limit (hard limit violation)", violation_flags
    
    # 7. Inverter capability check (P² + Q² ≤ S_rated) - HARD LIMIT
    # Standard Pandapower field: sn_mva = Rated Apparent Power (S_rated)
    if not net.res_sgen.empty and not net.sgen.empty and 'sn_mva' in net.sgen.columns:
        sgen_p_mw = net.res_sgen.p_mw.values
        sgen_q_mvar = net.res_sgen.q_mvar.values
        sgen_sn_mva = net.sgen.sn_mva.values
        
        # Only check generators where sn_mva is defined (> 0)
        valid_rating = sgen_sn_mva > 0
        
        if np.any(valid_rating):
            # Vectorized calculation
            sgen_s_mva = np.sqrt(sgen_p_mw[valid_rating]**2 + sgen_q_mvar[valid_rating]**2)
            ratings = sgen_sn_mva[valid_rating]
            
            # Vectorized check: P² + Q² ≤ S_rated (with 1% tolerance for floating point)
            if np.any(sgen_s_mva > ratings * 1.01):
                violation_flags['inverter_capability_violation'] = True
                # This is a hard limit violation - log but don't discard (should be caught in pre-check)
                max_violation = ((sgen_s_mva - ratings) / ratings * 100).max()
                return False, f"Inverter capability exceeded: {max_violation:.1f}% over rated S (sn_mva) (hard limit violation)", violation_flags
    
    # Build reason string for valid but stressed states
    violations = []
    if violation_flags['voltage_violation']:
        violations.append("voltage")
    if violation_flags['angle_violation']:
        violations.append("angle")
    if violation_flags['line_loading_violation']:
        violations.append("line_loading")
    if violation_flags['slack_power_violation']:
        violations.append("slack_power")
    
    if violations:
        reason = f"Valid but stressed state (operational violations: {', '.join(violations)})"
    else:
        reason = "Valid (no operational violations)"
    
    return True, reason, violation_flags


def apply_curtailment_with_retry(net: pp.pandapowerNet, base_renewable_p_mw: dict, 
                                  max_attempts: int = 10, convergence_stats: dict = None,
                                  has_contingency: bool = False) -> tuple[bool, float, dict]:
    """
    UNIFIED CURTAILMENT AND RETRY LOOP: Try to fix invalid states by reducing renewable generation.
    
    This is the PRIMARY strategy - preserves physics while maintaining time-series continuity.
    Works for both normal and contingency scenarios.
    
    Args:
        net: Pandapower network
        base_renewable_p_mw: Dict mapping sgen index -> original p_mw value
        max_attempts: Maximum curtailment retry attempts
        convergence_stats: Statistics dict to update
        has_contingency: Whether this is a contingency scenario
        
    Returns:
        (success, final_scaling_factor, violation_flags):
            - success: True if valid state found, False if all attempts failed
            - final_scaling_factor: Final scaling applied (1.0 = no curtailment, 0.0 = tripped)
            - violation_flags: Dict with violation flags (if successful)
    """
    violation_flags = {}
    curtailment_scaling = 1.0
    
    for attempt in range(max_attempts):
        # Apply curtailment scaling if this is a retry (attempt > 0)
        if attempt > 0:
            # Reduce renewable generation by 10% per attempt
            curtailment_scaling = 0.90 ** attempt
            
            # Apply scaling to all renewable generators
            for sgen_idx, base_p_mw in base_renewable_p_mw.items():
                net.sgen.at[sgen_idx, 'p_mw'] = base_p_mw * curtailment_scaling
            
            # Recalculate reactive power for curtailed active power
            for i, sgen in net.sgen.iterrows():
                q_gen = calculate_renewable_reactive_power(
                    net.sgen.at[i, 'p_mw'], sgen.bus, net, True  # Always use voltage control
                )
                net.sgen.at[i, 'q_mvar'] = q_gen
            
            if convergence_stats:
                convergence_stats['validation_stats']['curtailment_attempts'] += 1
            print(f"  [Curtailment] Attempt {attempt + 1}: Reducing renewable generation to {curtailment_scaling*100:.1f}%")
        
        # Try power flow
        try:
            with SuppressPrints():
                pp.runpp(net, numba=False, enforce_q_lims=True, algorithm='nr', tolerance_mva=1e-8)
            
            # POST-POWER-FLOW VALIDATION: Check outputs after power flow
            output_valid, output_reason, violation_flags = validate_power_flow_outputs(net, convergence_stats or {})
            
            if output_valid:
                # Success! Validation passed (may have operational violations, but physically valid)
                if attempt > 0 and convergence_stats:
                    convergence_stats['validation_stats']['curtailment_events'] += 1
                    convergence_stats['validation_stats']['curtailment_successful'] += 1
                    print(f"  [Curtailment] Successfully recovered with {curtailment_scaling*100:.1f}% renewable generation")
                return True, curtailment_scaling, violation_flags
            else:
                # Still invalid - check if it's "garbage" (hard limit) or can be fixed
                if "garbage" in output_reason.lower() or "hard limit" in output_reason.lower():
                    # Numerical garbage or hard limit - can't fix with curtailment
                    if convergence_stats:
                        convergence_stats['validation_stats']['post_validation_failed'] += 1
                        convergence_stats['validation_stats']['garbage_discarded'] += 1
                    return False, curtailment_scaling, violation_flags
                # Otherwise, continue curtailment loop (reduce more)
                
        except pp.LoadflowNotConverged:
            # Power flow didn't converge - continue curtailment loop
            continue
    
    # All curtailment attempts failed
    return False, curtailment_scaling, violation_flags


def hard_reset_system(net: pp.pandapowerNet, base_load_p: np.ndarray, base_load_q: np.ndarray,
                      base_renewable_p_mw: dict, convergence_stats: dict = None,
                      dropped_line_idx: int = None) -> tuple[bool, int]:
    """
    HARD RESET: Reset system to safe baseline state after 3+ consecutive failures.
    
    This prevents "flatline" data leakage where multiple consecutive failures create
    duplicate rows. Resets to a conservative state (50% renewable, base load) and
    clears any contingency state.
    
    Args:
        net: Pandapower network
        base_load_p: Base active load values (from start of simulation)
        base_load_q: Base reactive load values (from start of simulation)
        base_renewable_p_mw: Dict mapping sgen index -> original p_mw value
        convergence_stats: Statistics dict to update
        dropped_line_idx: Current contingency line index (if any)
    
    Returns:
        (success, new_dropped_line_idx):
            - success: True if reset succeeded, False otherwise
            - new_dropped_line_idx: None (contingency cleared) or original if reset failed
    """
    if convergence_stats:
        convergence_stats['validation_stats']['hard_resets'] = convergence_stats['validation_stats'].get('hard_resets', 0) + 1
    
    print(f"  [Hard Reset] Triggered after 3+ consecutive failures - resetting to safe baseline state")
    
    # 1. Clear any contingency state (restore all lines)
    if dropped_line_idx is not None:
        restore_contingency(net, dropped_line_idx)
        print(f"  [Hard Reset] Restored contingency line {dropped_line_idx}")
    
    # 2. Reset load to base values (conservative - no time-series variation)
    net.load.p_mw = base_load_p.copy()
    net.load.q_mvar = base_load_q.copy()
    print(f"  [Hard Reset] Reset load to base values")
    
    # 3. Reset renewable generation to 50% of original (safe baseline)
    # This is more conservative than full generation but maintains some renewable presence
    reset_scaling = 0.5
    if not net.sgen.empty and base_renewable_p_mw:
        for sgen_idx, base_p_mw in base_renewable_p_mw.items():
            reset_p_mw = base_p_mw * reset_scaling
            net.sgen.at[sgen_idx, 'p_mw'] = reset_p_mw
            
            # Recalculate reactive power for reset active power
            if sgen_idx in net.sgen.index:
                sgen = net.sgen.loc[sgen_idx]
                q_gen = calculate_renewable_reactive_power(
                    reset_p_mw, sgen.bus, net, False  # Don't use voltage control on reset
                )
                net.sgen.at[sgen_idx, 'q_mvar'] = q_gen
        
        print(f"  [Hard Reset] Reset renewable generation to {reset_scaling*100:.0f}% of original")
    else:
        # No renewable generators - just ensure they're at 0
        if not net.sgen.empty:
            net.sgen.p_mw = 0.0
            net.sgen.q_mvar = 0.0
        print(f"  [Hard Reset] No renewable generators to reset")
    
    # 4. Force power flow with reset state
    try:
        with SuppressPrints():
            pp.runpp(net, numba=False, enforce_q_lims=True, algorithm='nr', tolerance_mva=1e-8)
        
        # Validate the reset state
        output_valid, output_reason, violation_flags = validate_power_flow_outputs(net, convergence_stats or {})
        
        if output_valid:
            print(f"  [Hard Reset] Successfully reset to safe baseline state - grid stable")
            return True, None  # Success, no contingency
        else:
            print(f"  [Hard Reset] Reset state validation failed: {output_reason}")
            # Even reset failed - try with generators tripped
            if trip_renewable_generators(net, convergence_stats):
                print(f"  [Hard Reset] Recovered by tripping generators after reset")
                return True, None
            else:
                print(f"  [Hard Reset] Complete failure - even trip after reset failed")
                return False, None
            
    except pp.LoadflowNotConverged:
        print(f"  [Hard Reset] Power flow failed even after reset - attempting generator trip")
        # Last resort: trip generators
        if trip_renewable_generators(net, convergence_stats):
            return True, None
        else:
            return False, None


def trip_renewable_generators(net: pp.pandapowerNet, convergence_stats: dict = None) -> bool:
    """
    FINAL FALLBACK: Trip all renewable generators (set to 0.0) and run power flow.
    This is more physically realistic than forward-filling and avoids data leakage.
    
    Returns:
        True if power flow succeeded, False otherwise
    """
    # Set all renewable generation to 0.0 (generator trip)
    if not net.sgen.empty:
        net.sgen.p_mw = 0.0
        net.sgen.q_mvar = 0.0
    
    try:
        with SuppressPrints():
            pp.runpp(net, numba=False, enforce_q_lims=True, algorithm='nr', tolerance_mva=1e-8)
        
        # Validate the tripped state
        output_valid, output_reason, violation_flags = validate_power_flow_outputs(net, convergence_stats or {})
        
        if output_valid:
            if convergence_stats:
                convergence_stats['validation_stats']['generator_trips'] += 1
            print(f"  [Trip] Renewable generators tripped offline - grid stable")
            return True
        else:
            print(f"  [Trip] Even with generators tripped, validation failed: {output_reason}")
            return False
            
    except pp.LoadflowNotConverged:
        print(f"  [Trip] Power flow failed even with generators tripped")
        return False


# SECTION 3: SIMULATION AND SAVING

def simulate_time_series(net: pp.pandapowerNet, config: dict, output_dir: str = None, 
                         case_name: str = None, renewable_fraction: float = None, 
                         timestamp: str = None) -> dict:
    """
    Runs the main time-series power flow simulation with convergence tracking.
    
    MEMORY OPTIMIZATION: Supports chunked writing to disk to reduce RAM usage.
    If output_dir, case_name, renewable_fraction, and timestamp are provided,
    data is written incrementally in chunks. Otherwise, data is accumulated in RAM
    (backward compatible mode).
    
    Args:
        net: Pandapower network
        config: Configuration dictionary
        output_dir: Optional output directory for chunked writing
        case_name: Optional case name for chunked writing
        renewable_fraction: Optional renewable fraction for chunked writing
        timestamp: Optional timestamp for chunked writing
    
    Returns:
        Dictionary containing simulation data and convergence statistics
    """
    num_buses = len(net.bus)
    time_steps = config['time_steps']
    
    # MEMORY OPTIMIZATION: Chunked writing mode
    use_chunked_writing = config.get('use_chunked_writing', True)
    chunk_size = config.get('chunk_size', 1000)
    
    # Check if we have all parameters for chunked writing
    chunked_mode = (use_chunked_writing and output_dir is not None and 
                    case_name is not None and renewable_fraction is not None and 
                    timestamp is not None)
    
    if chunked_mode:
        print(f"[Memory Optimization] Using chunked writing (chunk_size={chunk_size})")
        # Create temporary files for incremental writing
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix='gen_data_chunks_')
        
        # Create memory-mapped files for incremental writing
        feature_file = os.path.join(temp_dir, 'features_temp.npy')
        target_file = os.path.join(temp_dir, 'targets_temp.npy')
        bus_types_file = os.path.join(temp_dir, 'bus_types_temp.npy')
        topology_ids_file = os.path.join(temp_dir, 'topology_ids_temp.npy')
        energy_coeffs_file = os.path.join(temp_dir, 'energy_coeffs_temp.npy')
        carbon_coeffs_file = os.path.join(temp_dir, 'carbon_coeffs_temp.npy')
        
        # Initialize memory-mapped arrays (write mode)
        # Use np.memmap for efficient disk-backed arrays
        feature_matrix = np.memmap(
            feature_file, mode='w+', dtype=np.float32, 
            shape=(time_steps, num_buses, 10)
        )
        target_matrix = np.memmap(
            target_file, mode='w+', dtype=np.float32,
            shape=(time_steps, num_buses, 2)
        )
        bus_types_array = np.memmap(
            bus_types_file, mode='w+', dtype=np.int32,
            shape=(time_steps, num_buses)
        )
        topology_ids = np.memmap(
            topology_ids_file, mode='w+', dtype=np.int32,
            shape=(time_steps,)
        )
        time_energy_coeffs = np.memmap(
            energy_coeffs_file, mode='w+', dtype=np.float32,
            shape=(time_steps,)
        )
        time_carbon_coeffs = np.memmap(
            carbon_coeffs_file, mode='w+', dtype=np.float32,
            shape=(time_steps,)
        )
        
        # Track written chunks for verification
        chunks_written = 0
        total_written = 0
    else:
        # Backward compatible: accumulate in RAM
        feature_matrix = np.zeros((time_steps, num_buses, 10), dtype=np.float32)
        target_matrix = np.zeros((time_steps, num_buses, 2), dtype=np.float32)
        bus_types_array = np.zeros((time_steps, num_buses), dtype=np.int32)
        topology_ids = np.zeros(time_steps, dtype=np.int32)
        time_energy_coeffs = np.zeros(time_steps, dtype=np.float32)
        time_carbon_coeffs = np.zeros(time_steps, dtype=np.float32)
        temp_dir = None
    
    # Calculate base adjacency matrix once (before any contingencies)
    base_adjacency_matrix = calculate_adjacency_matrix(net)
    
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
        
        # NEW: Validation and recovery statistics
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
    
    # Calculate realistic renewable capacity based on system load
    total_system_load_mw = base_load_p.sum()
    print(f"Total system load: {total_system_load_mw:.2f} MW")
    
    solar_gens = net.sgen[net.sgen.type == 'solar'] if 'type' in net.sgen.columns else pd.DataFrame()
    wind_gens = net.sgen[net.sgen.type == 'wind'] if 'type' in net.sgen.columns else pd.DataFrame()
    
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
    
    # SET RATED CAPACITY (sn_mva) FOR VALIDATION
    # Standard Pandapower field: sn_mva = Rated Apparent Power (S_rated)
    # Inverter rating must be slightly higher than max Active Power to allow for Reactive Power
    if num_total_renewable > 0 and 'type' in net.sgen.columns:
        for i, sgen in net.sgen.iterrows():
            if sgen.type == 'solar' and max_individual_solar_mw > 0:
                # Set inverter rating (MVA) 10% higher than max Active Power (MW) to allow for Q
                net.sgen.at[i, 'sn_mva'] = max_individual_solar_mw * 1.1
            elif sgen.type == 'wind' and max_individual_wind_mw > 0:
                # Set inverter rating (MVA) 10% higher than max Active Power (MW) to allow for Q
                net.sgen.at[i, 'sn_mva'] = max_individual_wind_mw * 1.1
    
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
    
    # Track consecutive failures for hard reset mechanism
    consecutive_failures = 0
    max_consecutive_failures = 0
    
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

        # PERFORMANCE FIX: Store topology_id instead of full adjacency matrix
        # The adjacency matrix will be reconstructed in data loader from base + topology_id
        if has_contingency and dropped_line_idx is not None:
            # Contingency: store line index + 1 (0 is reserved for base topology)
            topology_ids[t] = dropped_line_idx + 1
        else:
            # Base topology: store 0
            topology_ids[t] = 0

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
        
        # Initialize convergence tracking for this timestep
        convergence_successful = False
        resolution_method = None
        violation_flags = {}
        
        # === HARD RESET CHECK: Prevent flatline data leakage ===
        # If 3+ consecutive failures, reset to safe baseline state
        # This prevents RNN/LSTM gradient issues from duplicate rows
        if consecutive_failures >= 3:
            print(f"  [Hard Reset] {consecutive_failures} consecutive failures detected - triggering hard reset")
            
            # Store base renewable generation before reset (needed for reset function)
            base_renewable_p_mw_for_reset = {}
            if not net.sgen.empty:
                for i, sgen in net.sgen.iterrows():
                    # Get original generation from time-series profile (before any curtailment)
                    if config.get('use_time_series', False):
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
                    else:
                        # Monte Carlo mode - use current value as base
                        base_renewable_p_mw_for_reset[i] = net.sgen.at[i, 'p_mw']
            
            # Perform hard reset
            reset_success, new_dropped_line_idx = hard_reset_system(
                net, base_load_p, base_load_q, base_renewable_p_mw_for_reset,
                convergence_stats, dropped_line_idx
            )
            
            if reset_success:
                # Hard reset succeeded - update state
                dropped_line_idx = new_dropped_line_idx
                has_contingency = (dropped_line_idx is not None)
                consecutive_failures = 0  # Reset counter on successful hard reset
                convergence_successful = True
                resolution_method = 'hard_reset'
                _, _, violation_flags = validate_power_flow_outputs(net, convergence_stats)
                convergence_stats['successful'] += 1
                convergence_stats['successful_timesteps'].append(t)
                convergence_stats['resolution_methods']['hard_reset'] = convergence_stats['resolution_methods'].get('hard_reset', 0) + 1
                print(f"  [Hard Reset] Successfully recovered - continuing with reset state")
            else:
                # Hard reset failed - this is catastrophic
                consecutive_failures += 1
                max_consecutive_failures = max(max_consecutive_failures, consecutive_failures)
                convergence_stats['validation_stats']['consecutive_failures'] = consecutive_failures
                convergence_stats['validation_stats']['max_consecutive_failures'] = max_consecutive_failures
                convergence_stats['failed'] += 1
                if has_contingency:
                    convergence_stats['failed_with_contingency'].append(t)
                else:
                    convergence_stats['failed_no_contingency'].append(t)
                print(f"  ERROR: Hard reset failed - timestep {t} cannot be recovered, skipping")
                continue  # Skip this timestep entirely
        
        # === UNIFIED CURTAILMENT AND RETRY STRATEGY ===
        # PRIMARY: Curtailment loop (reduces renewable generation if validation fails)
        # FALLBACK: Trip generators (set to 0.0) if curtailment fails
        # This preserves physics and avoids data leakage
        
        if not convergence_successful:  # Only proceed if hard reset didn't already succeed
            resolution_method = None
            violation_flags = {}
            
            # PRE-POWER-FLOW VALIDATION: Check inputs before running power flow
            input_valid, input_reason = validate_power_flow_inputs(net)
            if not input_valid:
                # Pre-validation failures are usually hard limits (generator capacity, etc.)
                # These can't be fixed by curtailment - trip generators as final fallback
                convergence_stats['validation_stats']['pre_validation_failed'] += 1
                print(f"  WARNING: Timestep {t} failed pre-validation: {input_reason}, attempting generator trip")
                
                # FINAL FALLBACK: Trip generators
                if trip_renewable_generators(net, convergence_stats):
                    convergence_successful = True
                    resolution_method = 'trip_generators'
                    # Get violation flags for tripped state
                    _, _, violation_flags = validate_power_flow_outputs(net, convergence_stats)
                else:
                    consecutive_failures += 1
                    max_consecutive_failures = max(max_consecutive_failures, consecutive_failures)
                    convergence_stats['validation_stats']['consecutive_failures'] = consecutive_failures
                    convergence_stats['validation_stats']['max_consecutive_failures'] = max_consecutive_failures
                    convergence_stats['failed'] += 1
                    convergence_stats['failed_no_contingency'].append(t)
                    print(f"  ERROR: Timestep {t} failed even after generator trip - skipping")
                    continue
        
        # CRITICAL: Store original renewable generation for curtailment retry logic
        base_renewable_p_mw = {}
        if not net.sgen.empty and not convergence_successful:  # Only if we haven't already tripped
            for i, sgen in net.sgen.iterrows():
                base_renewable_p_mw[i] = net.sgen.at[i, 'p_mw']
        
        # UNIFIED CURTAILMENT LOOP: Works for both normal and contingency scenarios
        if not convergence_successful:  # Only if pre-validation passed but we haven't succeeded yet
            curtailment_success, curtailment_scaling, violation_flags = apply_curtailment_with_retry(
                net, base_renewable_p_mw, max_attempts=10, 
                convergence_stats=convergence_stats, has_contingency=has_contingency
            )
            
            if curtailment_success:
                # Success! Curtailment worked (may have required reduction)
                convergence_successful = True
                convergence_stats['successful'] += 1
                convergence_stats['successful_timesteps'].append(t)
                consecutive_failures = 0  # Reset consecutive failures on success
                
                # Track resolution method
                if has_contingency:
                    resolution_method = 'curtailment_contingency' if curtailment_scaling < 1.0 else 'strict_contingency'
                    convergence_stats['resolution_methods']['strict_contingency'] += 1
                    convergence_stats['contingencies_successful'] += 1
                    convergence_stats['contingencies_resolved_strict'] += 1
                else:
                    resolution_method = 'curtailment_normal' if curtailment_scaling < 1.0 else 'strict_normal'
                    convergence_stats['resolution_methods']['strict_normal'] += 1
            else:
                # Curtailment failed - try contingency fallback strategies WITH CURTAILMENT
                if has_contingency and dropped_line_idx is not None:
                    # Track critical line
                    line_key = f"line_{dropped_line_idx}"
                    if line_key not in convergence_stats['critical_lines']:
                        convergence_stats['critical_lines'][line_key] = {
                            'line_id': int(dropped_line_idx),
                            'failure_count': 0,
                            'resolution_methods': {'relaxed_curtailment': 0, 'restored_curtailment': 0, 'trip': 0}
                        }
                    convergence_stats['critical_lines'][line_key]['failure_count'] += 1
                    
                    # CONTINGENCY FALLBACK 1: Try relaxed settings WITH CURTAILMENT
                    # Restore original generation for curtailment retry
                    for i, base_p_mw in base_renewable_p_mw.items():
                        net.sgen.at[i, 'p_mw'] = base_p_mw
                    
                    # Try curtailment with relaxed power flow settings
                    relaxed_success, relaxed_scaling, relaxed_violations = apply_curtailment_with_retry(
                        net, base_renewable_p_mw, max_attempts=10,
                        convergence_stats=convergence_stats, has_contingency=True
                    )
                    
                    if relaxed_success:
                        # Modify power flow to use relaxed settings (curtailment already applied)
                        try:
                            with SuppressPrints():
                                pp.runpp(net, numba=False, enforce_q_lims=False, algorithm='nr', 
                                        tolerance_mva=1e-6, max_iteration=20)
                            # Re-validate with relaxed settings
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
                        # CONTINGENCY FALLBACK 2: Restore line and try curtailment with normal topology
                        restore_contingency(net, dropped_line_idx)
                        dropped_line_idx = None
                        has_contingency = False
                        convergence_stats['critical_lines'][line_key]['resolution_methods']['restored_curtailment'] += 1
                        
                        # Restore original generation
                        for i, base_p_mw in base_renewable_p_mw.items():
                            net.sgen.at[i, 'p_mw'] = base_p_mw
                        
                        # Try curtailment with restored topology
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
                            # All curtailment attempts failed - trip generators
                            if trip_renewable_generators(net, convergence_stats):
                                convergence_successful = True
                                _, _, violation_flags = validate_power_flow_outputs(net, convergence_stats)
                                convergence_stats['successful'] += 1
                                convergence_stats['successful_timesteps'].append(t)
                                consecutive_failures = 0
                                resolution_method = 'trip_after_restore'
                                convergence_stats['critical_lines'][line_key]['resolution_methods']['trip'] += 1
                            else:
                                # Complete failure
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
                    # Failed without contingency - trip generators as final fallback
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
        
        # If we reach here, we have a successful power flow (via curtailment, trip, or direct)
        if not convergence_successful:
            # This should never happen, but safety check
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
        
        # Store resolution method for this timestep
        if resolution_method:
            convergence_stats['timestep_resolution'][str(t)] = resolution_method
        
        # Track violations for valid but stressed states
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
            
            # Count valid stressed states (any operational violation but still valid)
            if any([violation_flags.get('voltage_violation', False), 
                   violation_flags.get('angle_violation', False),
                   violation_flags.get('line_loading_violation', False), 
                   violation_flags.get('slack_power_violation', False)]):
                convergence_stats['validation_stats']['valid_stressed_states'] += 1
        
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
        
        # ============================================================================
        # OPTIMAL POWER FLOW (OPF) APPROACH: Predict only unknowns based on bus type
        # ============================================================================
        # Identify bus types from pandapower network (determined by power flow solution)
        # Bus types are determined AFTER power flow - pandapower decides based on network state
        bus_types = identify_bus_types(net)
        bus_types_array[t] = bus_types  # Store for later use (e.g., evaluation, loss calculation)
        
        # Create OPF-style targets: only unknowns for each bus type
        # PQ bus: [V, θ], PV bus: [Q, θ], Slack: [P, Q]
        opf_targets = create_opf_targets(net, bus_types)
        target_matrix[t] = opf_targets  # [num_buses, 2]

        # ============================================================================
        # FEATURES: Power measurements + partial voltage measurements (what sensors provide)
        # Shape: [num_buses, 10] = [p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren, vm_meas, va_meas]
        # ============================================================================
        
        # Generate noise for measurements
        positive_noise_vm = np.abs(np.random.normal(0, config['voltage_error_std'], num_buses))
        positive_noise_angle = np.abs(np.random.normal(0, config['angle_error_std'], num_buses))
        positive_noise_power = np.abs(np.random.normal(0, config['power_error_std'], num_buses))
        
        # CRITICAL: Get system base power for per-unit conversion
        # All power measurements MUST be in per-unit for unit consistency
        s_base_mva = net.sn_mva  # Base power in MVA
        
        # Power measurements (from smart meters, SCADA, etc.)
        # These are the "known" quantities with sensor noise
        # CRITICAL FIX: Convert to per-unit BEFORE applying noise (unit consistency)
        meas_pl = (p_load / s_base_mva) * (1 + positive_noise_power)  # Per-unit
        meas_ql = (q_load / s_base_mva) * (1 + positive_noise_power)  # Per-unit
        meas_p_ext = (ext_grid_p_by_bus.values / s_base_mva) * (1 + positive_noise_power)  # Per-unit
        meas_q_ext = (ext_grid_q_by_bus.values / s_base_mva) * (1 + positive_noise_power)  # Per-unit
        meas_p_conv = (gen_p_by_bus.values / s_base_mva) * (1 + positive_noise_power)  # Per-unit
        meas_q_conv = (gen_q_by_bus.values / s_base_mva) * (1 + positive_noise_power)  # Per-unit
        meas_p_ren = (sgen_p_by_bus.values / s_base_mva) * (1 + positive_noise_power)  # Per-unit
        meas_q_ren = (sgen_q_by_bus.values / s_base_mva) * (1 + positive_noise_power)  # Per-unit
        
        # Partial voltage measurements (from PMUs at SOME buses only - ETH Zurich style)
        # PMUs are expensive, so only some buses have them (typically 20-40% coverage)
        # Select buses with PMUs (random selection each timestep for realism)
        num_pmu_buses = max(1, int(num_buses * config.get('pmu_coverage', 0.3)))
        pmu_buses = np.random.choice(num_buses, size=num_pmu_buses, replace=False)
        
        # Initialize with NaN (no measurement)
        meas_vm = np.full(num_buses, np.nan)
        meas_va = np.full(num_buses, np.nan)
        
        # Add noisy PMU measurements only at selected buses
        # Voltage is already in per-unit (vm_pu), angle is in radians
        meas_vm[pmu_buses] = vm_pu[pmu_buses] * (1 + positive_noise_vm[pmu_buses])  # Per-unit
        meas_va[pmu_buses] = va_rad[pmu_buses] * (1 + positive_noise_angle[pmu_buses])  # Radians
        
        # Replace NaN with 0 for missing measurements (model will learn to ignore these)
        meas_vm = np.nan_to_num(meas_vm, nan=0.0)
        meas_va = np.nan_to_num(meas_va, nan=0.0)
        
        # Feature matrix: Power measurements FIRST (indices 0-7), then partial voltages (indices 8-9)
        # This ordering is important for physics calculations
        # ALL VALUES ARE NOW IN CONSISTENT UNITS: per-unit for power, per-unit for voltage, radians for angle
        feature_matrix[t] = np.stack([
            meas_pl, meas_ql,           # [0:2] Load measurements (per-unit)
            meas_p_ext, meas_q_ext,     # [2:4] External grid (per-unit)
            meas_p_conv, meas_q_conv,   # [4:6] Conventional generation (per-unit)
            meas_p_ren, meas_q_ren,     # [6:8] Renewable generation (per-unit)
            meas_vm, meas_va            # [8:10] Partial voltage measurements (per-unit, radians)
        ], axis=1)
        
        # Adjacency matrix already stored above
        
        # MEMORY OPTIMIZATION: Flush chunk to disk periodically
        if chunked_mode and (t + 1) % chunk_size == 0:
            # Flush memory-mapped arrays to ensure data is written
            feature_matrix.flush()
            target_matrix.flush()
            bus_types_array.flush()
            topology_ids.flush()
            time_energy_coeffs.flush()
            time_carbon_coeffs.flush()
            chunks_written += 1
            total_written = t + 1
            print(f"  [Chunked Writing] Flushed chunk {chunks_written} ({total_written}/{time_steps} timesteps written)")
    
    # Final flush for any remaining data
    if chunked_mode:
        feature_matrix.flush()
        target_matrix.flush()
        bus_types_array.flush()
        topology_ids.flush()
        time_energy_coeffs.flush()
        time_carbon_coeffs.flush()
        
        # VERIFICATION: Ensure all data was written
        total_written = time_steps
        print(f"  [Chunked Writing] All data flushed to disk ({total_written}/{time_steps} timesteps)")
        
        # Verify file sizes match expected
        expected_size = time_steps * num_buses * 10 * 4  # float32 = 4 bytes
        actual_size = os.path.getsize(feature_file)
        if actual_size < expected_size * 0.99:  # Allow 1% tolerance for file system
            raise RuntimeError(
                f"Data verification failed: feature file size {actual_size} bytes < expected {expected_size} bytes. "
                f"This indicates incomplete write. Data may be corrupted."
            )
        print(f"  [Chunked Writing] Verification passed: {actual_size} bytes written")
    
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
    
    # MEMORY OPTIMIZATION: If using chunked writing, load arrays from disk for return
    # (backward compatibility - save_data expects arrays, but they're memory-mapped)
    if chunked_mode:
        # Arrays are already memory-mapped - copy to regular arrays to close file handles
        # This allows temp directory cleanup on Windows (file handles must be closed)
        # Small memory cost but ensures proper cleanup
        print(f"  [Chunked Writing] Copying memmap arrays to regular arrays for cleanup...")
        features_return = np.array(feature_matrix)
        targets_return = np.array(target_matrix)
        bus_types_return = np.array(bus_types_array)
        topology_ids_return = np.array(topology_ids)
        energy_coeffs_return = np.array(time_energy_coeffs)
        carbon_coeffs_return = np.array(time_carbon_coeffs)
        
        # Close memmap files by deleting references (allows Windows to delete temp dir)
        del feature_matrix, target_matrix, bus_types_array, topology_ids
        del time_energy_coeffs, time_carbon_coeffs
        import gc
        gc.collect()  # Force garbage collection to close file handles
        
        # Store temp directory info for cleanup after save_data
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
        # Normal mode: arrays are already in memory
        features_return = feature_matrix
        targets_return = target_matrix
        bus_types_return = bus_types_array
        topology_ids_return = topology_ids
        energy_coeffs_return = time_energy_coeffs
        carbon_coeffs_return = time_carbon_coeffs
    
    return {
        "features": features_return,  # [timesteps, buses, 10] = measurements
        "targets": targets_return,     # [timesteps, buses, 2] = OPF unknowns (bus-type dependent)
        "bus_types": bus_types_return, # [timesteps, buses] = bus type codes [0=PQ, 1=PV, 2=Slack]
        "base_adjacency": base_adjacency_matrix,  # [num_buses, num_buses] - single base matrix
        "topology_ids": topology_ids_return,  # [timesteps] - topology ID for each timestep (0=base, line_idx+1=contingency)
        "ybus_data": ybus_data,  # Sparse format
        "time_energy_coeffs": energy_coeffs_return, 
        "time_carbon_coeffs": carbon_coeffs_return,
        "convergence_stats": convergence_stats  # Detailed convergence report
    }

    

def save_data(data_dict: dict, case_name: str, renewable_fraction: float, output_dir: str, timestamp: str = None):
    """
    Saves generated data arrays with support for sparse Ybus format and convergence reports.
    
    MEMORY OPTIMIZATION: If data comes from chunked writing (memory-mapped arrays),
    this function efficiently copies them to final location. Otherwise, saves normally.
    
    Args:
        data_dict: Dictionary containing data arrays (may be memory-mapped from chunked writing)
        case_name: Name of the test case (e.g., 'case33')
        renewable_fraction: Renewable energy fraction
        output_dir: Directory to save files
        timestamp: Optional timestamp string to ensure data consistency
    """
    import json
    import shutil
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # MEMORY OPTIMIZATION: Check if data came from chunked writing
    # If so, we can efficiently copy the temp files instead of re-saving
    convergence_stats = data_dict.get('convergence_stats', {})
    temp_files = convergence_stats.get('_temp_files', None)
    use_temp_files = (temp_files is not None and os.path.exists(temp_files.get('features', '')))
    
    # MEMORY OPTIMIZATION: If using temp files from chunked writing, copy them efficiently
    if use_temp_files:
        print(f"[Memory Optimization] Copying chunked data files to final location...")
        
        # Map temp keys to final filenames
        file_mappings = {
            'features': f"{case_name}_features_frac{renewable_fraction:.1f}_{timestamp}.npy",
            'targets': f"{case_name}_targets_frac{renewable_fraction:.1f}_{timestamp}.npy",
            'bus_types': f"{case_name}_bus_types_frac{renewable_fraction:.1f}_{timestamp}.npy",
            'topology_ids': f"{case_name}_topology_ids_frac{renewable_fraction:.1f}_{timestamp}.npy",
            'energy_coeffs': f"{case_name}_time_energy_coeffs_frac{renewable_fraction:.1f}_{timestamp}.txt",
            'carbon_coeffs': f"{case_name}_time_carbon_coeffs_frac{renewable_fraction:.1f}_{timestamp}.txt"
        }
        
        # Use data from data_dict directly (already converted from memmap to regular arrays)
        # This is more reliable than trying to reload from temp files
        for temp_key, final_filename in file_mappings.items():
                final_path = os.path.join(output_dir, final_filename)
                
                if temp_key in ['energy_coeffs', 'carbon_coeffs']:
                    # Convert .npy to .txt for coefficient files
                    coeff_key = 'time_energy_coeffs' if 'energy' in temp_key else 'time_carbon_coeffs'
                    if coeff_key in data_dict:
                        data = data_dict[coeff_key]
                        # Convert to regular array if needed (for np.savetxt)
                        if isinstance(data, np.memmap):
                            data = np.array(data)
                        np.savetxt(final_path, data)
                        print(f"  Copied {temp_key} -> {final_filename} (converted to .txt)")
                    else:
                        raise RuntimeError(f"Cannot convert {temp_key}: data not found in data_dict")
                else:
                    # Map temp_key to data_dict key
                    data_key_map = {
                        'features': 'features',
                        'targets': 'targets',
                        'bus_types': 'bus_types',
                        'topology_ids': 'topology_ids'
                    }
                    
                    data_key = data_key_map.get(temp_key)
                    if data_key and data_key in data_dict:
                        # Get data from data_dict (already converted from memmap)
                        data = data_dict[data_key]
                        
                        # Convert to regular array if still memmap (shouldn't happen, but safety)
                        if isinstance(data, np.memmap):
                            data = np.array(data)
                        
                        # Save with explicit dtype and no pickle
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
                        print(f"  Copied {temp_key} -> {final_filename}")
                    else:
                        raise RuntimeError(f"Cannot copy {temp_key}: data not found in data_dict")
        
        # Verify all files were copied correctly
        for temp_key, final_filename in file_mappings.items():
            final_path = os.path.join(output_dir, final_filename)
            if not os.path.exists(final_path):
                raise RuntimeError(f"Failed to copy {temp_key}: {final_path} does not exist")
        
        print(f"[Memory Optimization] All chunked data files copied successfully")
    
    # Process remaining data (ybus, adjacency, convergence stats)
    for key, data in data_dict.items():
        # Skip keys that were already handled by chunked writing
        if use_temp_files and key in ['features', 'targets', 'bus_types', 'topology_ids', 
                                      'time_energy_coeffs', 'time_carbon_coeffs']:
            continue
        # Handle sparse Ybus data specially
        if key == "ybus_data":
            # Save each component of the sparse Ybus separately
            for sub_key, sub_data in data.items():
                sub_filename = f"{case_name}_ybus_{sub_key}_frac{renewable_fraction:.1f}_{timestamp}.npy"
                filepath = os.path.join(output_dir, sub_filename)
                print(f"Saving Ybus component '{sub_key}' to '{filepath}'...")
                np.save(filepath, sub_data, allow_pickle=False)
            continue
        
        # Handle convergence statistics specially (remove temp file references before saving)
        if key == "convergence_stats":
            # Create a copy without temp file references
            stats_to_save = {k: v for k, v in data.items() 
                           if not k.startswith('_temp')}
            
            # Transform to professional data quality audit format
            from utils.data_auditor import transform_convergence_to_audit
            audit_data = transform_convergence_to_audit(
                stats_to_save, case_name, renewable_fraction, timestamp
            )
            
            # Save as data_quality_audit.json (professional naming)
            # Note: raw_convergence_stats is included inside audit_data for backward compatibility
            stats_filename = f"{case_name}_data_quality_audit_frac{renewable_fraction:.1f}_{timestamp}.json"
            filepath = os.path.join(output_dir, stats_filename)
            print(f"Saving data quality audit to '{filepath}'...")
            with open(filepath, 'w') as f:
                json.dump(audit_data, f, indent=2)
            continue
        
        # Create a base filename that includes the case, renewable fraction, and timestamp
        base_filename = f"{case_name}_{key}_frac{renewable_fraction:.1f}_{timestamp}"
        
        # Check if the key indicates a coefficient file
        if "coeffs" in key:
            # Save these 1D arrays as .txt files
            filename = os.path.join(output_dir, base_filename + ".txt")
            print(f"Saving coefficient data to '{filename}'...")
            # Handle memory-mapped arrays (convert to regular array if needed)
            if isinstance(data, np.memmap):
                np.savetxt(filename, np.array(data))
            else:
                np.savetxt(filename, data)
        elif key == "topology_ids":
            # Save topology_ids as .npy file (1D array of integers)
            filename = os.path.join(output_dir, base_filename + ".npy")
            print(f"Saving topology IDs to '{filename}'...")
            # Handle memory-mapped arrays
            if isinstance(data, np.memmap):
                np.save(filename, np.array(data), allow_pickle=False)
            else:
                np.save(filename, data, allow_pickle=False)
        elif key == "base_adjacency":
            # Save base adjacency as .npy file (will be converted to edge_index format for compatibility)
            # Convert to edge_index format for backward compatibility with data loader
            from scipy import sparse
            # Handle memory-mapped arrays
            if isinstance(data, np.memmap):
                adj_data = np.array(data)
            else:
                adj_data = data
            adj_sparse = sparse.coo_matrix(adj_data)
            edge_index = np.array([adj_sparse.row, adj_sparse.col])
            # Save as object array (same format as old adjacency files)
            filename = os.path.join(output_dir, base_filename + ".npy")
            print(f"Saving base adjacency matrix to '{filename}'...")
            np.save(filename, np.array([edge_index], dtype=object), allow_pickle=True)
        else:
            # Save all other multi-dimensional arrays as .npy files
            filename = os.path.join(output_dir, base_filename + ".npy")
            print(f"Saving array data to '{filename}'...")
            # Handle memory-mapped arrays
            if isinstance(data, np.memmap):
                np.save(filename, np.array(data), allow_pickle=True)
            else:
                np.save(filename, data, allow_pickle=True)

# SECTION 4: MAIN EXECUTION BLOCK
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
                
                # FIXED: Save directly to data/[train|test] (removed time_series subfolder)
                # Structure: data/train or data/test
                script_dir = os.path.dirname(os.path.abspath(__file__))
                if "data" in script_dir:
                    # Script is being run from data/ subdirectory
                    output_path = os.path.join(script_dir, data_mode)
                else:
                    # Script is being run from main directory
                    output_path = os.path.join(script_dir, "data", data_mode)
                
                # Create directory if it doesn't exist
                os.makedirs(output_path, exist_ok=True)
                
                # MEMORY OPTIMIZATION: Pass output parameters for chunked writing
                generated_data = simulate_time_series(
                    net_with_renewables, CONFIG,
                    output_dir=output_path,
                    case_name=save_case_name,
                    renewable_fraction=frac,
                    timestamp=generation_timestamp
                )
                    
                # Pass the generation timestamp to ensure all files have the same timestamp
                save_data(generated_data, save_case_name, frac, output_path, generation_timestamp)
                
                # MEMORY OPTIMIZATION: Clean up temporary files after saving
                if '_temp_dir' in generated_data.get('convergence_stats', {}):
                    import shutil
                    import time
                    temp_dir = generated_data['convergence_stats']['_temp_dir']
                    if os.path.exists(temp_dir):
                        # On Windows, file handles may still be open - retry with delay
                        max_retries = 3
                        for retry in range(max_retries):
                            try:
                                shutil.rmtree(temp_dir)
                                print(f"  [Chunked Writing] Cleaned up temporary files")
                                break
                            except PermissionError as e:
                                if retry < max_retries - 1:
                                    # Wait a bit and retry (Windows file handle release delay)
                                    time.sleep(0.1)
                                    # Force garbage collection to close any remaining handles
                                    import gc
                                    gc.collect()
                                else:
                                    # Last retry failed - log warning but continue
                                    print(f"  [Warning] Could not delete temp directory {temp_dir}: {e}")
                                    print(f"  [Warning] Temp files will be cleaned up on next run or manually")

        except DataGenerationError as e:
            # SEVERE ERROR: Stop execution immediately
            print(f"\n{'='*80}")
            print(f"SEVERE ERROR: Data generation cannot continue!")
            print(f"{'='*80}")
            print(f"Error details:")
            traceback.print_exc()
            print(f"\n{'='*80}")
            print(f"Data generation STOPPED due to severe error.")
            print(f"This error indicates fundamental problems that make the data invalid.")
            print(f"Fix the issue before attempting to generate data again.")
            print(f"{'='*80}\n")
            sys.exit(1)  # Exit with error code
        except Exception as e:
            # Other recoverable errors: log and continue to next case
            print(f"\nWARNING: An error occurred while processing {case}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            traceback.print_exc()
            print(f"\nSkipping to the next test case.")
            continue
    
    # Write metadata file for smart data detection
    try:
        # FIXED: Save directly to data/[train|test] (removed time_series subfolder)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if "data" in script_dir:
            output_path = os.path.join(script_dir, data_mode)
        else:
            output_path = os.path.join(script_dir, "data", data_mode)
        
        os.makedirs(output_path, exist_ok=True)
        
        # Keep generation_mode in metadata for backward compatibility, but path structure is simplified
        generation_mode = 'time_series' if CONFIG.get('use_time_series', False) else 'monte_carlo'
        metadata = {
            'generation_mode': generation_mode,
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