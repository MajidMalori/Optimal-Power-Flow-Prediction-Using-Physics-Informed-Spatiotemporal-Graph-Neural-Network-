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

# Import from original and refactored
from data.topology import calculate_ybus_from_net as calc_ybus_refactored
from data.topology import calculate_ybus_from_net as calc_ybus_new

def verify_ybus_consistency():
    print("\nVERIFYING YBUS CONSISTENCY")
    print("="*60)
    
    cases = ["case33", "case57", "case118"]
    all_passed = True
    
    for case in cases:
        print(f"\nTesting {case}...")
        
        # Load network
        if case == "case33": net = pn.case33bw()
        elif case == "case57": net = pn.case57()
        elif case == "case118": net = pn.case118()
        
        # Run power flow to ensure internal state
        pp.runpp(net)
        
        # Calculate Ybus using both methods
        try:
            ybus_refactored = calc_ybus_refactored(net)
            ybus_new = calc_ybus_new(net)
            
            # Compare
            diff = np.abs(ybus_refactored - ybus_new)
            max_diff = np.max(diff)
            
            if max_diff < 1e-10:
                print(f"  PASS: Max difference = {max_diff:.12f}")
            else:
                print(f"  FAIL: Max difference = {max_diff:.12f}")
                all_passed = False
                
        except Exception as e:
            print(f"  ERROR: {e}")
            all_passed = False
            
    print("\n" + "="*60)
    if all_passed:
        print("VERIFICATION SUCCESSFUL: Refactored Ybus matches original exactly.")
        return 0
    else:
        print("VERIFICATION FAILED: Mismatches detected.")
        return 1

if __name__ == "__main__":
    sys.exit(verify_ybus_consistency())
