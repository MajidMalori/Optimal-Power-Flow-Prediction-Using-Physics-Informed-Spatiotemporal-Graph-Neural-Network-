import numpy as np
import warnings
import pandapower as pp
from constants import HOURLY_LOAD_PATTERN
try:
    from utils.contingency_ybus import DataGenerationError
except ImportError:
    class DataGenerationError(Exception):
        """Fallback exception if utils is missing."""
        pass

def get_daily_load_profile(hour: int, season: str = 'summer') -> float:
    return HOURLY_LOAD_PATTERN.get(hour, 0.5) * np.random.uniform(0.95, 1.05)

def get_solar_generation_profile(hour: int, day_of_year: int = 180, weather_state: str = None) -> float:
    if hour < 5 or hour > 19: return 0.0
    
    hour_from_noon = abs(hour - 12)
    if hour_from_noon > 7: return 0.0
    
    solar_angle = (hour - 12) * (np.pi / 14)
    base_solar = max(0, np.cos(solar_angle))
    season_factor = 0.85 + 0.15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    if weather_state is None:
        weather_state = np.random.choice(['clear', 'partly_cloudy', 'cloudy', 'storm'], 
                                        p=[0.3, 0.4, 0.25, 0.05])
    
    cloud_factors = {
        'clear': (0.90, 1.0),
        'partly_cloudy': (0.35, 0.85),
        'cloudy': (0.08, 0.35),
        'storm': (0.0, 0.08)
    }
    low, high = cloud_factors.get(weather_state, (0.0, 0.08))
    cloud_factor = np.random.uniform(low, high)
    
    return base_solar * cloud_factor * season_factor

def get_wind_generation_profile(hour: int, day: int = 0, weather_state: str = None) -> float:
    if weather_state is None:
        day_seed = np.random.RandomState(day)
        weather_state = day_seed.choice(['calm', 'breezy', 'windy', 'storm'], 
                                       p=[0.15, 0.45, 0.30, 0.10])
    
    wind_ranges = {
        'calm': (0.0, 0.20),
        'breezy': (0.20, 0.55),
        'windy': (0.55, 0.90),
        'storm': (0.85, 1.0)
    }
    low, high = wind_ranges.get(weather_state, (0.85, 1.0))
    base_wind = np.random.uniform(low, high)
    
    thermal_factor = 1.0 + 0.08 * np.sin(2 * np.pi * (hour - 6) / 24)
    micro_variation = np.random.uniform(0.85, 1.15)
    
    return np.clip(base_wind * thermal_factor * micro_variation, 0.0, 1.0)

def simulate_weather_sequence(timesteps: int, hours_per_day: int = 24, seed: int = None) -> list:
    if seed is not None: np.random.seed(seed)
    
    solar_transitions = {
        'clear':         {'clear': 0.65, 'partly_cloudy': 0.30, 'cloudy': 0.05, 'storm': 0.0},
        'partly_cloudy': {'clear': 0.25, 'partly_cloudy': 0.45, 'cloudy': 0.25, 'storm': 0.05},
        'cloudy':        {'clear': 0.10, 'partly_cloudy': 0.30, 'cloudy': 0.50, 'storm': 0.10},
        'storm':         {'clear': 0.0,  'partly_cloudy': 0.10, 'cloudy': 0.40, 'storm': 0.50}
    }
    
    wind_transitions = {
        'calm':   {'calm': 0.60, 'breezy': 0.30, 'windy': 0.08, 'storm': 0.02},
        'breezy': {'calm': 0.20, 'breezy': 0.50, 'windy': 0.25, 'storm': 0.05},
        'windy':  {'calm': 0.05, 'breezy': 0.30, 'windy': 0.50, 'storm': 0.15},
        'storm':  {'calm': 0.02, 'breezy': 0.10, 'windy': 0.40, 'storm': 0.48}
    }
    
    solar_seq, wind_seq = [], []
    curr_solar, curr_wind = 'partly_cloudy', 'breezy'
    
    for _ in range(timesteps):
        solar_seq.append(curr_solar)
        wind_seq.append(curr_wind)
        
        curr_solar = np.random.choice(list(solar_transitions[curr_solar].keys()), p=list(solar_transitions[curr_solar].values()))
        curr_wind = np.random.choice(list(wind_transitions[curr_wind].keys()), p=list(wind_transitions[curr_wind].values()))
    
    return list(zip(solar_seq, wind_seq))

def calculate_renewable_reactive_power(p_mw: float, bus_idx: int, net: pp.pandapowerNet, use_voltage_control: bool = True) -> float:
    if p_mw < 1e-5: return 0.0
    
    max_q = 0.33 * p_mw
    q_mvar = 0.0
    
    if use_voltage_control and hasattr(net, 'res_bus') and not net.res_bus.empty:
        try:
            v_pu = net.res_bus.loc[bus_idx, 'vm_pu']
            if v_pu < 0.98:
                q_mvar = min(1.0, (0.98 - v_pu) / 0.03) * max_q
            elif v_pu > 1.02:
                q_mvar = -min(1.0, (v_pu - 1.02) / 0.03) * max_q
        except (KeyError, AttributeError):
            q_mvar = p_mw * np.tan(np.arccos(0.98))
    else:
        q_mvar = p_mw * np.tan(np.arccos(0.98))
    
    if abs(q_mvar) > 1e-6:
        max_mult = max_q / abs(q_mvar)
        q_mvar *= np.random.uniform(0.95, min(1.05, max_mult * 0.99))
    else:
        q_mvar *= np.random.uniform(0.95, 1.05)
    
    q_clipped = np.clip(q_mvar, -max_q, max_q)
    
    if max_q >= 1e-5 and abs(q_mvar) - abs(q_clipped) > max_q * 0.001:
        raise DataGenerationError(f"Inverter Q limit violation: {q_mvar:.6f} vs ±{max_q:.6f}")
        
    return q_clipped
