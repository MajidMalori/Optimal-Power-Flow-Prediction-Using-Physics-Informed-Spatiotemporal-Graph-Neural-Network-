import numpy as np
import pandapower as pp
import sys
import os
from data.profiles import calculate_renewable_reactive_power
from data.topology import restore_contingency

class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._devnull = open(os.devnull, 'w')
        sys.stdout = self._devnull
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        self._devnull.close()

def validate_power_flow_inputs(net: pp.pandapowerNet) -> tuple[bool, str]:
    if not net.gen.empty:
        gen_p, gen_max = net.gen.p_mw.values, net.gen.max_p_mw.values
        if np.any(gen_p > gen_max * 1.01):
            idx = np.where(gen_p > gen_max * 1.01)[0][0]
            return False, f"Generator capacity violation at gen {idx}"
    
    if not net.sgen.empty and 'sn_mva' in net.sgen.columns:
        sgen_p, sgen_q = net.sgen.p_mw.values, net.sgen.q_mvar.values
        sgen_s_rated = net.sgen.sn_mva.values
        valid = sgen_s_rated > 0
        if np.any(valid):
            sgen_s = np.sqrt(sgen_p[valid]**2 + sgen_q[valid]**2)
            if np.any(sgen_s > sgen_s_rated[valid] * 1.01):
                return False, "Inverter capability violation"
    
    if not net.load.empty and np.any(net.load.p_mw.values < -1e-6):
        return False, "Negative load detected"
    
    return True, "Input validation passed"

from constants import (
    V_GARBAGE_LOW, V_GARBAGE_HIGH, 
    ANGLE_GARBAGE_THRESHOLD, ANGLE_WARNING_THRESHOLD,
    SYSTEM_PHYSICS
)

def validate_power_flow_outputs(net: pp.pandapowerNet, convergence_stats: dict, case_name: str = None, config: dict = None) -> tuple[bool, str, dict]:
    flags = {k: False for k in ['voltage_violation', 'angle_violation', 'line_loading_violation', 
                                'slack_power_violation', 'generator_capacity_violation', 'inverter_capability_violation']}
    
    if not net.converged: return False, "Power flow did not converge", flags
    
    vm_pu = net.res_bus.vm_pu.values
    if np.any(vm_pu < V_GARBAGE_LOW) or np.any(vm_pu > V_GARBAGE_HIGH):
        return False, "Voltage out of physical bounds (garbage)", flags
    
    # Use strict physical limits based on topology
    physics = SYSTEM_PHYSICS.get(case_name, SYSTEM_PHYSICS['default'])
    v_min, v_max = physics['v_min'], physics['v_max']
        
    if np.any(vm_pu < v_min) or np.any(vm_pu > v_max): flags['voltage_violation'] = True
    
    va_rad = np.deg2rad(net.res_bus.va_degree.values)
    if not net.line.empty:
        active = net.line[net.line.in_service]
        diffs = np.abs(va_rad[active.from_bus.values.astype(int)] - va_rad[active.to_bus.values.astype(int)])
        if np.max(diffs) > ANGLE_GARBAGE_THRESHOLD: return False, "Angle difference > 90 deg (garbage)", flags
        if np.max(diffs) > ANGLE_WARNING_THRESHOLD: flags['angle_violation'] = True
    
    if not net.res_ext_grid.empty:
        slack_p = net.res_ext_grid.p_mw.values
        max_load = net.load.p_mw.sum() if not net.load.empty else 0
        limit = 5 * max_load if max_load > 0 else 10000
        if np.any(np.abs(slack_p) > limit): return False, "Slack power unrealistic (garbage)", flags
        if np.any(np.abs(slack_p) > 2 * max_load): flags['slack_power_violation'] = True
    
    if not net.res_line.empty:
        loading = net.res_line.loading_percent.values
        if np.any(loading > 1000): return False, "Line loading > 1000% (garbage)", flags
        if np.any(loading > 100): flags['line_loading_violation'] = True
        
    return True, "Valid", flags

def apply_curtailment_with_retry(net: pp.pandapowerNet, base_renewable_p_mw: dict, 
                                  max_attempts: int = 10, convergence_stats: dict = None,
                                  has_contingency: bool = False, case_name: str = None, config: dict = None) -> tuple[bool, float, dict]:
    scaling = 1.0
    for attempt in range(max_attempts):
        if attempt > 0:
            scaling = 0.90 ** attempt
            for idx, base_p in base_renewable_p_mw.items():
                net.sgen.at[idx, 'p_mw'] = base_p * scaling
            for i, sgen in net.sgen.iterrows():
                net.sgen.at[i, 'q_mvar'] = calculate_renewable_reactive_power(net.sgen.at[i, 'p_mw'], sgen.bus, net, True)
            
            if convergence_stats: convergence_stats['validation_stats']['curtailment_attempts'] += 1
        
        try:
            with SuppressPrints():
                pp.runpp(net, numba=True, enforce_q_lims=True, algorithm='nr', tolerance_mva=1e-8)
            
            valid, reason, flags = validate_power_flow_outputs(net, convergence_stats or {}, case_name, config)
            if valid:
                if attempt > 0 and convergence_stats:
                    convergence_stats['validation_stats']['curtailment_successful'] += 1
                return True, scaling, flags
            elif "garbage" in reason or "hard limit" in reason:
                return False, scaling, flags
                
        except pp.LoadflowNotConverged:
            continue
            
    return False, scaling, {}

def hard_reset_system(net: pp.pandapowerNet, base_load_p: np.ndarray, base_load_q: np.ndarray,
                      base_renewable_p_mw: dict, convergence_stats: dict = None,
                      dropped_line_idx: int = None, case_name: str = None, config: dict = None) -> tuple[bool, int]:
    if convergence_stats: convergence_stats['validation_stats']['hard_resets'] += 1
    
    if dropped_line_idx is not None: restore_contingency(net, dropped_line_idx)
    net.load.p_mw = base_load_p.copy()
    net.load.q_mvar = base_load_q.copy()
    
    reset_scaling = 0.5
    if not net.sgen.empty:
        if base_renewable_p_mw:
            for idx, base_p in base_renewable_p_mw.items():
                p_new = base_p * reset_scaling
                net.sgen.at[idx, 'p_mw'] = p_new
                if idx in net.sgen.index:
                    net.sgen.at[idx, 'q_mvar'] = calculate_renewable_reactive_power(p_new, net.sgen.at[idx, 'bus'], net, False)
        else:
            net.sgen.p_mw = 0.0
            net.sgen.q_mvar = 0.0

    try:
        with SuppressPrints():
            pp.runpp(net, numba=True, enforce_q_lims=True, algorithm='nr', tolerance_mva=1e-8)
        
        valid, _, _ = validate_power_flow_outputs(net, convergence_stats or {}, case_name, config)
        if valid: return True, None
        
        return trip_renewable_generators(net, convergence_stats, case_name, config), None
            
    except pp.LoadflowNotConverged:
        return trip_renewable_generators(net, convergence_stats, case_name, config), None

def trip_renewable_generators(net: pp.pandapowerNet, convergence_stats: dict = None, case_name: str = None, config: dict = None) -> bool:
    if not net.sgen.empty:
        net.sgen.p_mw = 0.0
        net.sgen.q_mvar = 0.0
    
    try:
        with SuppressPrints():
            pp.runpp(net, numba=True, enforce_q_lims=True, algorithm='nr', tolerance_mva=1e-8)
        
        valid, _, _ = validate_power_flow_outputs(net, convergence_stats or {}, case_name, config)
        if valid:
            if convergence_stats: convergence_stats['validation_stats']['generator_trips'] += 1
            return True
        return False
            
    except pp.LoadflowNotConverged:
        return False
