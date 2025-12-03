import numpy as np
import warnings
import pandapower as pp
from utils.contingency_ybus import DataGenerationError

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
        warnings.warn(
            f"Inverter reactive power clipped: requested {q_mvar:.6f} Mvar, "
            f"limited to {q_mvar_clipped:.6f} Mvar (capability: ±{max_q_capability:.6f} Mvar). "
            f"Bus {bus_idx}, P={p_mw:.6f} MW.",
            UserWarning
        )
    
    return q_mvar_clipped
