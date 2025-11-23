import numpy as np
import pandapower as pp
import sys
import os

# Import from sibling modules
from data.profiles import calculate_renewable_reactive_power
from data.topology import restore_contingency

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
    
    return True, "Input validation passed"


def validate_power_flow_outputs(net: pp.pandapowerNet, convergence_stats: dict, case_name: str = None) -> tuple[bool, str, dict]:
    """
    POST-POWER-FLOW VALIDATION: Check outputs after power flow.
    Separates "Valid Stressed States" (keep) from "Numerical Garbage" (discard).
    
    Args:
        net: pandapower network
        convergence_stats: convergence statistics dict
        case_name: Name of the test case (e.g., "case57") for case-specific limits
    
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
    
    # CASE-SPECIFIC VOLTAGE LIMITS
    # Case 57 is a transmission system with known voltage stability challenges
    # Transmission systems typically operate at lower voltage ranges than distribution
    if case_name and "57" in case_name:
        v_min_limit = 0.70  # Relaxed for transmission system
        v_max_limit = 1.20
        limit_desc = "transmission"
    else:
        v_min_limit = 0.85  # Standard for distribution systems
        v_max_limit = 1.15
        limit_desc = "distribution"
    
    # Check for operational violations
    if np.any(vm_pu < v_min_limit) or np.any(vm_pu > v_max_limit):
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
                                  has_contingency: bool = False, case_name: str = None) -> tuple[bool, float, dict]:
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
        case_name: Name of test case for case-specific validation
        
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
            # print(f"  [Curtailment] Attempt {attempt + 1}: Reducing renewable generation to {curtailment_scaling*100:.1f}%")
        
        # Try power flow
        try:
            with SuppressPrints():
                pp.runpp(net, numba=False, enforce_q_lims=True, algorithm='nr', tolerance_mva=1e-8)
            
            # POST-POWER-FLOW VALIDATION: Check outputs after power flow
            output_valid, output_reason, violation_flags = validate_power_flow_outputs(net, convergence_stats or {}, case_name)
            
            if output_valid:
                # Success! Validation passed (may have operational violations, but physically valid)
                if attempt > 0 and convergence_stats:
                    convergence_stats['validation_stats']['curtailment_events'] += 1
                    convergence_stats['validation_stats']['curtailment_successful'] += 1
                    # print(f"  [Curtailment] Successfully recovered with {curtailment_scaling*100:.1f}% renewable generation")
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
                      dropped_line_idx: int = None, case_name: str = None) -> tuple[bool, int]:
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
    
    # print(f"  [Hard Reset] Triggered after 3+ consecutive failures - resetting to safe baseline state")
    
    # 1. Clear any contingency state (restore all lines)
    if dropped_line_idx is not None:
        restore_contingency(net, dropped_line_idx)
        # print(f"  [Hard Reset] Restored contingency line {dropped_line_idx}")
    
    # 2. Reset load to base values (conservative - no time-series variation)
    net.load.p_mw = base_load_p.copy()
    net.load.q_mvar = base_load_q.copy()
    # print(f"  [Hard Reset] Reset load to base values")
    
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
        
        # print(f"  [Hard Reset] Reset renewable generation to {reset_scaling*100:.0f}% of original")
    else:
        # No renewable generators - just ensure they're at 0
        if not net.sgen.empty:
            net.sgen.p_mw = 0.0
            net.sgen.q_mvar = 0.0
        # print(f"  [Hard Reset] No renewable generators to reset")
    
    # 4. Force power flow with reset state
    try:
        with SuppressPrints():
            pp.runpp(net, numba=False, enforce_q_lims=True, algorithm='nr', tolerance_mva=1e-8)
        
        # Validate the reset state
        output_valid, output_reason, violation_flags = validate_power_flow_outputs(net, convergence_stats or {}, case_name)
        
        if output_valid:
            # print(f"  [Hard Reset] Successfully reset to safe baseline state - grid stable")
            return True, None  # Success, no contingency
        else:
            # print(f"  [Hard Reset] Reset state validation failed: {output_reason}")
            # Even reset failed - try with generators tripped
            if trip_renewable_generators(net, convergence_stats, case_name):
                # print(f"  [Hard Reset] Recovered by tripping generators after reset")
                return True, None
            else:
                # print(f"  [Hard Reset] Complete failure - even trip after reset failed")
                return False, None
            
    except pp.LoadflowNotConverged:
        # print(f"  [Hard Reset] Power flow failed even after reset - attempting generator trip")
        # Last resort: trip generators
        if trip_renewable_generators(net, convergence_stats, case_name):
            return True, None
        else:
            return False, None


def trip_renewable_generators(net: pp.pandapowerNet, convergence_stats: dict = None, case_name: str = None) -> bool:
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
        output_valid, output_reason, violation_flags = validate_power_flow_outputs(net, convergence_stats or {}, case_name)
        
        if output_valid:
            if convergence_stats:
                convergence_stats['validation_stats']['generator_trips'] += 1
            # print(f"  [Trip] Renewable generators tripped offline - grid stable")
            return True
        else:
            # print(f"  [Trip] Even with generators tripped, validation failed: {output_reason}")
            return False
            
    except pp.LoadflowNotConverged:
        # print(f"  [Trip] Power flow failed even with generators tripped")
        return False
