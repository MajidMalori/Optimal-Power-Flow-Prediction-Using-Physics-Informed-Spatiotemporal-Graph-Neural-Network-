import sys
import pytest
from unittest.mock import patch

# Mock standard terminal arguments to trick benchmark_warmstart.py into doing a tiny run
from scripts import benchmark_warmstart

def test_warmstart_benchmark_end_to_end():
    """
    Simulates calling `python scripts/benchmark_warmstart.py --samples 2`
    This provides an automated end-to-end test the user can click to run.
    """
    
    # We patch sys.argv to simulate the command line arguments
    test_args = [
        "benchmark_warmstart.py",
        "--case", "case33",
        "--samples", "2"
    ]
    
    with patch.object(sys, 'argv', test_args):
        try:
            benchmark_warmstart.main()
            success = True
        except Exception as e:
            pytest.fail(f"Warmstart benchmark failed to run end-to-end: {e}")
            success = False
            
    assert success, "Benchmark pipeline crashed during execution."
