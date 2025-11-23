import unittest
import sys
import os
import subprocess

def run_script(script_name, args=[]):
    print(f"\n{'='*80}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*80}")
    cmd = [sys.executable, script_name] + args
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"FAILED: {script_name}")
        return False
    print(f"PASSED: {script_name}")
    return True

def main():
    # All tests should use test data mode
    data_mode = "test"
    
    # Test scripts that accept --mode argument
    test_scripts = [
        ("tests/test_feature_units.py", ["--mode", data_mode]),
        ("tests/test_power_injection.py", ["--mode", data_mode]),
        ("tests/test_feature_target_alignment.py", ["--mode", data_mode]),
        ("tests/test_renewable_penetration.py", ["--mode", data_mode]),
        ("tests/test_load_distribution.py", ["--mode", data_mode]),
    ]
    
    # Validation scripts (no arguments needed)
    validation_scripts = [
        ("validate_simple.py", ["all"]),
    ]
    
    failed = []
    
    # Run test scripts
    for script, args in test_scripts:
        if not run_script(script, args):
            failed.append(script)
    
    # Run validation scripts
    for script, args in validation_scripts:
        if not run_script(script, args):
            failed.append(script)
            
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    if failed:
        print(f"FAILURES ({len(failed)}):")
        for s in failed:
            print(f"  - {s}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED!")
        sys.exit(0)

if __name__ == "__main__":
    main()
