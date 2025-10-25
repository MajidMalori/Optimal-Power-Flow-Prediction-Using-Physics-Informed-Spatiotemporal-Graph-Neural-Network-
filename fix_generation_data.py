#!/usr/bin/env python3
"""
Fix for Generation Data - Handle External Grid Negative Values
This script demonstrates the correct way to handle external grid results.
"""

import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd

def demonstrate_external_grid_issue():
    """Demonstrate why external grid can have negative values."""
    print("=" * 80)
    print("DEMONSTRATING EXTERNAL GRID NEGATIVE VALUES")
    print("=" * 80)
    
    # Load network
    net = pn.case33bw()
    print(f"Loaded network: {net.name}")
    
    # Show initial load
    total_load = net.load.p_mw.sum()
    print(f"Total system load: {total_load:.2f} MW")
    
    # Add high renewable generation (more than load)
    pp.create_sgen(net, bus=5, p_mw=total_load * 1.5, q_mvar=0, name="HighRenewable@5")
    print(f"Added renewable generation: {total_load * 1.5:.2f} MW (150% of load)")
    
    # Run power flow
    try:
        pp.runpp(net, numba=False, enforce_q_lims=True, algorithm='nr', tolerance_mva=1e-8)
        print("Power flow converged successfully!")
        
        # Check results
        print(f"\n--- RESULTS ---")
        print(f"External Grid P_mw: {net.res_ext_grid.p_mw.iloc[0]:.4f} MW")
        print(f"Renewable P_mw: {net.res_sgen.p_mw.iloc[0]:.4f} MW")
        print(f"Total Load: {net.res_load.p_mw.sum():.4f} MW")
        
        if net.res_ext_grid.p_mw.iloc[0] < 0:
            print("❌ NEGATIVE External Grid P_mw detected!")
            print("This happens when renewable generation exceeds load.")
            print("The external grid absorbs excess power (negative injection).")
        else:
            print("✅ External Grid P_mw is positive (injecting power)")
            
    except Exception as e:
        print(f"Power flow failed: {e}")

def show_correct_handling():
    """Show how to correctly handle external grid values."""
    print(f"\n" + "=" * 80)
    print("CORRECT HANDLING OF EXTERNAL GRID VALUES")
    print("=" * 80)
    
    # Load network
    net = pn.case33bw()
    
    # Add renewable generation
    pp.create_sgen(net, bus=5, p_mw=10, q_mvar=0, name="Renewable@5")
    
    # Run power flow
    pp.runpp(net, numba=False, enforce_q_lims=True, algorithm='nr', tolerance_mva=1e-8)
    
    # Get external grid results
    ext_grid_p = net.res_ext_grid.p_mw.iloc[0]
    ext_grid_q = net.res_ext_grid.q_mvar.iloc[0]
    
    print(f"Raw External Grid P_mw: {ext_grid_p:.4f} MW")
    print(f"Raw External Grid Q_mvar: {ext_grid_q:.4f} MW")
    
    # Method 1: Use absolute value (treats absorption as generation)
    ext_grid_p_abs = abs(ext_grid_p)
    print(f"Absolute External Grid P_mw: {ext_grid_p_abs:.4f} MW")
    
    # Method 2: Clamp to zero (ignore absorption)
    ext_grid_p_clamped = max(0, ext_grid_p)
    print(f"Clamped External Grid P_mw: {ext_grid_p_clamped:.4f} MW")
    
    # Method 3: Separate injection and absorption
    if ext_grid_p >= 0:
        print(f"External Grid is INJECTING: {ext_grid_p:.4f} MW")
    else:
        print(f"External Grid is ABSORBING: {abs(ext_grid_p):.4f} MW")
        print("For carbon emissions, this should be treated as zero grid power")

def show_fix_for_data_generation():
    """Show the fix needed in data generation code."""
    print(f"\n" + "=" * 80)
    print("FIX FOR DATA GENERATION CODE")
    print("=" * 80)
    
    print("CURRENT CODE (PROBLEMATIC):")
    print("```python")
    print("# 3. Aggregate slack bus (external grid) generation")
    print("ext_grid_p_by_bus = net.res_ext_grid.groupby(net.ext_grid.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)")
    print("```")
    
    print("\nFIXED CODE (CORRECT):")
    print("```python")
    print("# 3. Aggregate slack bus (external grid) generation")
    print("# Handle negative values: negative means grid is absorbing, not generating")
    print("ext_grid_p_raw = net.res_ext_grid.groupby(net.ext_grid.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)")
    print("ext_grid_p_by_bus = np.maximum(0, ext_grid_p_raw)  # Only count positive injection")
    print("```")
    
    print("\nALTERNATIVE FIX (USE ABSOLUTE VALUES):")
    print("```python")
    print("# 3. Aggregate slack bus (external grid) generation")
    print("# Use absolute values to treat absorption as generation")
    print("ext_grid_p_by_bus = np.abs(net.res_ext_grid.groupby(net.ext_grid.bus).p_mw.sum().reindex(net.bus.index, fill_value=0))")
    print("```")
    
    print("\nRECOMMENDED APPROACH:")
    print("1. Use np.maximum(0, ext_grid_p) to only count positive injection")
    print("2. This ensures external grid generation is never negative")
    print("3. For carbon emissions, negative grid power means zero grid emissions")
    print("4. This is physically correct: absorption doesn't cause emissions")

if __name__ == "__main__":
    demonstrate_external_grid_issue()
    show_correct_handling()
    show_fix_for_data_generation()
