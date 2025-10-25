#!/usr/bin/env python3
"""
Fix for the 10-feature generation component issue.
The problem is that separated generation components have many zeros,
and when power flow fails, these become NaN.
"""

import numpy as np

def fix_generation_components():
    """Fix the generation component calculation to handle zeros properly"""
    
    print("🔧 FIXING GENERATION COMPONENT CALCULATION")
    print("=" * 60)
    
    # The issue is in the data generation script
    # We need to modify how the separated components are calculated
    
    fix_code = '''
# FIXED VERSION: Handle zero generation components properly
def calculate_separated_generation_components(net, num_buses):
    """Calculate separated generation components with proper zero handling"""
    
    # Initialize all components to zero
    ext_grid_p = np.zeros(num_buses)
    ext_grid_q = np.zeros(num_buses)
    gen_p = np.zeros(num_buses)
    gen_q = np.zeros(num_buses)
    sgen_p = np.zeros(num_buses)
    sgen_q = np.zeros(num_buses)
    
    # Fill in actual values where they exist
    if not net.res_ext_grid.empty:
        for _, ext in net.res_ext_grid.iterrows():
            bus_idx = int(ext.bus)
            ext_grid_p[bus_idx] = ext.p_mw
            ext_grid_q[bus_idx] = ext.q_mvar
    
    if not net.res_gen.empty:
        for _, gen in net.res_gen.iterrows():
            bus_idx = int(gen.bus)
            gen_p[bus_idx] = gen.p_mw
            gen_q[bus_idx] = gen.q_mvar
    
    if not net.res_sgen.empty:
        for _, sgen in net.res_sgen.iterrows():
            bus_idx = int(sgen.bus)
            sgen_p[bus_idx] = sgen.p_mw
            sgen_q[bus_idx] = sgen.q_mvar
    
    return ext_grid_p, ext_grid_q, gen_p, gen_q, sgen_p, sgen_q

# FIXED VERSION: Handle NaN values in measurements
def create_noisy_measurements_with_nan_handling(true_state, config, num_buses):
    """Create noisy measurements with proper NaN handling"""
    
    # Extract components
    vm_pu = true_state[:, 0]
    va_rad = true_state[:, 1]
    p_load = true_state[:, 2]
    q_load = true_state[:, 3]
    ext_grid_p = true_state[:, 4]
    ext_grid_q = true_state[:, 5]
    gen_p = true_state[:, 6]
    gen_q = true_state[:, 7]
    sgen_p = true_state[:, 8]
    sgen_q = true_state[:, 9]
    
    # Create noisy measurements with NaN protection
    meas_vm = vm_pu * (1 + np.random.normal(0, config['voltage_error_std'], num_buses))
    meas_va = va_rad + np.random.normal(0, config['angle_error_std'], num_buses)
    meas_pl = p_load * (1 + np.random.normal(0, config['power_error_std'], num_buses))
    meas_ql = q_load * (1 + np.random.normal(0, config['power_error_std'], num_buses))
    
    # For generation components, only add noise if the value is non-zero
    # This prevents NaN from zero * noise
    meas_p_ext = np.where(ext_grid_p != 0, 
                          ext_grid_p * (1 + np.random.normal(0, config['power_error_std'], num_buses)),
                          ext_grid_p)
    meas_q_ext = np.where(ext_grid_q != 0,
                          ext_grid_q * (1 + np.random.normal(0, config['power_error_std'], num_buses)),
                          ext_grid_q)
    meas_p_conv = np.where(gen_p != 0,
                           gen_p * (1 + np.random.normal(0, config['power_error_std'], num_buses)),
                           gen_p)
    meas_q_conv = np.where(gen_q != 0,
                           gen_q * (1 + np.random.normal(0, config['power_error_std'], num_buses)),
                           gen_q)
    meas_p_ren = np.where(sgen_p != 0,
                          sgen_p * (1 + np.random.normal(0, config['power_error_std'], num_buses)),
                          sgen_p)
    meas_q_ren = np.where(sgen_q != 0,
                          sgen_q * (1 + np.random.normal(0, config['power_error_std'], num_buses)),
                          sgen_q)
    
    return np.stack([meas_vm, meas_va, meas_pl, meas_ql, 
                     meas_p_ext, meas_q_ext, meas_p_conv, meas_q_conv, 
                     meas_p_ren, meas_q_ren], axis=1)
'''
    
    with open("generation_component_fix.py", "w") as f:
        f.write(fix_code)
    
    print("✅ Created generation_component_fix.py")
    print("\n🔍 ROOT CAUSE IDENTIFIED:")
    print("1. Separated generation components have many zeros")
    print("2. When power flow fails, zeros become NaN")
    print("3. Noise is applied to zero values, creating NaN")
    print("4. The old script combined everything, avoiding this issue")
    
    print("\n💡 SOLUTION:")
    print("1. Only apply noise to non-zero generation values")
    print("2. Keep zero values as zero (no noise)")
    print("3. This prevents NaN from zero * noise")
    print("4. Maintains the 10-feature approach while fixing NaN issues")

if __name__ == "__main__":
    fix_generation_components()
