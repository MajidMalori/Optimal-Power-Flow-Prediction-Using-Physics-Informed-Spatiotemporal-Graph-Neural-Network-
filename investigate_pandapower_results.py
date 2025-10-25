#!/usr/bin/env python3
"""
Investigate Pandapower Results to Understand Generation Values
This script examines what net.res_ext_grid, net.res_gen, and net.res_sgen actually contain.
"""

import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd

def investigate_pandapower_results():
    """Investigate what pandapower results actually contain."""
    print("=" * 80)
    print("INVESTIGATING PANDAPOWER RESULTS")
    print("=" * 80)
    
    # Load a simple test case
    net = pn.case33bw()
    print(f"Loaded network: {net.name}")
    print(f"Number of buses: {len(net.bus)}")
    
    # Show initial state
    print(f"\n--- INITIAL STATE ---")
    print("External Grid:")
    print("Columns:", net.ext_grid.columns.tolist())
    print(net.ext_grid)
    print("\nGenerators:")
    print("Columns:", net.gen.columns.tolist())
    print(net.gen)
    print("\nStatic Generators:")
    print("Columns:", net.sgen.columns.tolist())
    print(net.sgen)
    print("\nLoads:")
    print("Columns:", net.load.columns.tolist())
    print(net.load.head())
    
    # Run power flow
    print(f"\n--- RUNNING POWER FLOW ---")
    try:
        pp.runpp(net, numba=True, enforce_q_lims=True, algorithm='nr', tolerance_mva=1e-8)
        print("Power flow converged successfully!")
    except Exception as e:
        print(f"Power flow failed: {e}")
        return
    
    # Examine results
    print(f"\n--- POWER FLOW RESULTS ---")
    
    print("\n1. External Grid Results (net.res_ext_grid):")
    print("Columns:", net.res_ext_grid.columns.tolist())
    print("Data:")
    print(net.res_ext_grid)
    print(f"P_mw range: {net.res_ext_grid.p_mw.min():.4f} to {net.res_ext_grid.p_mw.max():.4f}")
    print(f"Q_mvar range: {net.res_ext_grid.q_mvar.min():.4f} to {net.res_ext_grid.q_mvar.max():.4f}")
    
    print("\n2. Generator Results (net.res_gen):")
    if not net.res_gen.empty:
        print("Columns:", net.res_gen.columns.tolist())
        print("Data:")
        print(net.res_gen)
        print(f"P_mw range: {net.res_gen.p_mw.min():.4f} to {net.res_gen.p_mw.max():.4f}")
        print(f"Q_mvar range: {net.res_gen.q_mvar.min():.4f} to {net.res_gen.q_mvar.max():.4f}")
    else:
        print("No generators in this network")
    
    print("\n3. Static Generator Results (net.res_sgen):")
    if not net.res_sgen.empty:
        print("Columns:", net.res_sgen.columns.tolist())
        print("Data:")
        print(net.res_sgen)
        print(f"P_mw range: {net.res_sgen.p_mw.min():.4f} to {net.res_sgen.p_mw.max():.4f}")
        print(f"Q_mvar range: {net.res_sgen.q_mvar.min():.4f} to {net.res_sgen.q_mvar.max():.4f}")
    else:
        print("No static generators in this network")
    
    print("\n4. Load Results (net.res_load):")
    print("Columns:", net.res_load.columns.tolist())
    print("Data (first 5 rows):")
    print(net.res_load.head())
    print(f"P_mw range: {net.res_load.p_mw.min():.4f} to {net.res_load.p_mw.max():.4f}")
    print(f"Q_mvar range: {net.res_load.q_mvar.min():.4f} to {net.res_load.q_mvar.max():.4f}")
    
    # Check for negative values
    print(f"\n--- CHECKING FOR NEGATIVE VALUES ---")
    
    ext_grid_negative = net.res_ext_grid.p_mw < 0
    if ext_grid_negative.any():
        print(f"❌ NEGATIVE External Grid P_mw found: {ext_grid_negative.sum()} values")
        print(f"Negative values: {net.res_ext_grid.p_mw[ext_grid_negative].values}")
    else:
        print("✅ No negative External Grid P_mw values")
    
    if not net.res_gen.empty:
        gen_negative = net.res_gen.p_mw < 0
        if gen_negative.any():
            print(f"❌ NEGATIVE Generator P_mw found: {gen_negative.sum()} values")
            print(f"Negative values: {net.res_gen.p_mw[gen_negative].values}")
        else:
            print("✅ No negative Generator P_mw values")
    
    if not net.res_sgen.empty:
        sgen_negative = net.res_sgen.p_mw < 0
        if sgen_negative.any():
            print(f"❌ NEGATIVE Static Generator P_mw found: {sgen_negative.sum()} values")
            print(f"Negative values: {net.res_sgen.p_mw[sgen_negative].values}")
        else:
            print("✅ No negative Static Generator P_mw values")
    
    load_negative = net.res_load.p_mw < 0
    if load_negative.any():
        print(f"❌ NEGATIVE Load P_mw found: {load_negative.sum()} values")
        print(f"Negative values: {net.res_load.p_mw[load_negative].values}")
    else:
        print("✅ No negative Load P_mw values")
    
    # Test with renewable generators
    print(f"\n--- TESTING WITH RENEWABLE GENERATORS ---")
    
    # Add some renewable generators
    net2 = pn.case33bw()
    
    # Add solar generator
    pp.create_sgen(net2, bus=5, p_mw=10, q_mvar=0, name="Solar@5", type='solar')
    pp.create_sgen(net2, bus=10, p_mw=15, q_mvar=0, name="Wind@10", type='wind')
    
    print("Added renewable generators:")
    print(net2.sgen[['bus', 'p_mw', 'q_mvar', 'name', 'type']])
    
    # Run power flow
    try:
        pp.runpp(net2, numba=True, enforce_q_lims=True, algorithm='nr', tolerance_mva=1e-8)
        print("Power flow with renewables converged successfully!")
        
        print("\nRenewable generator results:")
        print("Columns:", net2.res_sgen.columns.tolist())
        print(net2.res_sgen)
        print(f"P_mw range: {net2.res_sgen.p_mw.min():.4f} to {net2.res_sgen.p_mw.max():.4f}")
        
        # Check for negative values
        sgen_negative = net2.res_sgen.p_mw < 0
        if sgen_negative.any():
            print(f"❌ NEGATIVE Renewable P_mw found: {sgen_negative.sum()} values")
            print(f"Negative values: {net2.res_sgen.p_mw[sgen_negative].values}")
        else:
            print("✅ No negative Renewable P_mw values")
            
    except Exception as e:
        print(f"Power flow with renewables failed: {e}")
    
    print(f"\n--- PHYSICS INTERPRETATION ---")
    print("In power systems:")
    print("- P_mw > 0 means power is being INJECTED into the bus (generation)")
    print("- P_mw < 0 means power is being ABSORBED from the bus (consumption)")
    print("- For generators: P_mw should be positive (injecting power)")
    print("- For loads: P_mw should be positive (consuming power)")
    print("- For external grid: P_mw can be negative (absorbing excess) or positive (injecting)")

if __name__ == "__main__":
    investigate_pandapower_results()
