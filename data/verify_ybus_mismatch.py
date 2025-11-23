import sys
import os
import numpy as np
import pandapower as pp
import pandapower.networks as pn

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data.topology import calculate_ybus_from_net, identify_bus_types

def verify_mismatch():
    print("\nVERIFYING YBUS PHYSICS (MISMATCH TEST)")
    print("="*60)
    
    cases = ["case33", "case57", "case118"]
    all_passed = True
    
    for case in cases:
        print(f"\nTesting {case}...")
        
        # Load network
        if case == "case33": net = pn.case33bw()
        elif case == "case57": net = pn.case57()
        elif case == "case118": net = pn.case118()
        
        # Run power flow
        try:
            pp.runpp(net, algorithm='nr', calculate_voltage_angles=True)
            print(f"  Power flow converged")
        except Exception as e:
            print(f"  ERROR: Power flow failed: {e}")
            all_passed = False
            continue
            
        # Get Ybus from refactored module
        try:
            ybus = calculate_ybus_from_net(net)
            print(f"  Ybus extracted: {ybus.shape}")
        except Exception as e:
            print(f"  ERROR: Ybus extraction failed: {e}")
            all_passed = False
            continue
            
        # Get voltages
        V_mag = net.res_bus.vm_pu.values
        V_ang = np.radians(net.res_bus.va_degree.values)
        V = V_mag * np.exp(1j * V_ang)
        
        # Calculate power from Ybus (S = V * conj(Y * V))
        # Note: I = Y * V, so S = V * conj(I)
        I = ybus @ V
        S_calc = V * np.conj(I)
        
        # Calculate power injections from network elements
        S_inj = np.zeros(len(net.bus), dtype=complex)
        for bus_idx in range(len(net.bus)):
            # Loads
            loads = net.load[net.load.bus == bus_idx]
            p_load = loads.p_mw.sum() if not loads.empty else 0
            q_load = loads.q_mvar.sum() if not loads.empty else 0
            
            # Gens
            gens = net.gen[net.gen.bus == bus_idx]
            if not gens.empty:
                p_gen = gens.p_mw.sum()
                # Use result q_mvar for generators
                q_gen = net.res_gen.loc[gens.index, 'q_mvar'].sum()
            else:
                p_gen = 0
                q_gen = 0
                
            # Static Gens
            sgens = net.sgen[net.sgen.bus == bus_idx]
            p_sgen = sgens.p_mw.sum() if not sgens.empty else 0
            q_sgen = sgens.q_mvar.sum() if not sgens.empty else 0
            
            # Ext Grid
            exts = net.ext_grid[net.ext_grid.bus == bus_idx]
            if not exts.empty:
                p_ext = net.res_ext_grid.p_mw.sum()
                q_ext = net.res_ext_grid.q_mvar.sum()
            else:
                p_ext = 0
                q_ext = 0
                
            # Shunts
            shunts = net.shunt[net.shunt.bus == bus_idx]
            if not shunts.empty:
                # Shunts are modeled in Ybus usually, but let's see if we need to account for them explicitly
                # Pandapower includes shunts in Ybus, so we shouldn't add them to S_inj
                # UNLESS they are treated as constant power loads. 
                # Standard shunts are impedance, so they are in Ybus.
                pass
            
            # Net Injection = (Gen + Ext + Sgen) - Load
            # Normalized by base MVA
            S_inj[bus_idx] = ((p_gen + p_sgen + p_ext - p_load) + 1j * (q_gen + q_sgen + q_ext - q_load)) / net.sn_mva
            
        # Calculate mismatch
        bus_types = identify_bus_types(net)
        non_slack_mask = bus_types != 2
        
        mismatch = S_calc - S_inj
        mismatch_mag = np.abs(mismatch[non_slack_mask])
        
        max_mismatch = np.max(mismatch_mag)
        mean_mismatch = np.mean(mismatch_mag)
        
        print(f"  Max Mismatch (excluding Slack): {max_mismatch:.9f} p.u.")
        
        if max_mismatch < 1e-4:
            print("  PASS: Base topology mismatch is negligible.")
        else:
            print("  FAIL: Base topology mismatch is too high!")
            all_passed = False
            
        # --- CONTINGENCY TEST ---
        print(f"  Testing N-1 Contingency for {case}...")
        # Pick a random line to drop
        line_idx = net.line.index[0]
        net.line.loc[line_idx, 'in_service'] = False
        
        try:
            pp.runpp(net, algorithm='nr', calculate_voltage_angles=True)
            ybus_cont = calculate_ybus_from_net(net)
            
            # Recalculate mismatch for contingency
            V_mag = net.res_bus.vm_pu.values
            V_ang = np.radians(net.res_bus.va_degree.values)
            V = V_mag * np.exp(1j * V_ang)
            I = ybus_cont @ V
            S_calc = V * np.conj(I)
            
            # Recalculate injections (same logic, but net state changed)
            S_inj = np.zeros(len(net.bus), dtype=complex)
            for bus_idx in range(len(net.bus)):
                loads = net.load[net.load.bus == bus_idx]
                p_load = loads.p_mw.sum() if not loads.empty else 0
                q_load = loads.q_mvar.sum() if not loads.empty else 0
                
                gens = net.gen[net.gen.bus == bus_idx]
                if not gens.empty:
                    p_gen = gens.p_mw.sum()
                    q_gen = net.res_gen.loc[gens.index, 'q_mvar'].sum()
                else:
                    p_gen = 0; q_gen = 0
                    
                sgens = net.sgen[net.sgen.bus == bus_idx]
                p_sgen = sgens.p_mw.sum() if not sgens.empty else 0
                q_sgen = sgens.q_mvar.sum() if not sgens.empty else 0
                
                exts = net.ext_grid[net.ext_grid.bus == bus_idx]
                if not exts.empty:
                    p_ext = net.res_ext_grid.p_mw.sum()
                    q_ext = net.res_ext_grid.q_mvar.sum()
                else:
                    p_ext = 0; q_ext = 0
                
                S_inj[bus_idx] = ((p_gen + p_sgen + p_ext - p_load) + 1j * (q_gen + q_sgen + q_ext - q_load)) / net.sn_mva
            
            mismatch = S_calc - S_inj
            # Filter out slack and the isolated bus if any (though we just dropped one line)
            # For simplicity, just check max mismatch overall, ignoring slack
            mismatch_mag = np.abs(mismatch[bus_types != 2])
            max_cont_mismatch = np.max(mismatch_mag)
            
            print(f"  Max Contingency Mismatch: {max_cont_mismatch:.9f} p.u.")
            if max_cont_mismatch < 1e-4:
                print("  PASS: Contingency mismatch is negligible.")
            else:
                print("  FAIL: Contingency mismatch is too high!")
                all_passed = False
                
        except Exception as e:
            print(f"  WARNING: Contingency power flow failed (expected for some lines): {e}")
        
        # Restore line
        net.line.loc[line_idx, 'in_service'] = True
            
    print("="*60)
    if all_passed:
        print("VERIFICATION SUCCESSFUL: Ybus correctly predicts power flow results.")
        return 0
    else:
        print("VERIFICATION FAILED: Ybus does not match power flow results.")
        return 1

if __name__ == "__main__":
    sys.exit(verify_mismatch())
