import os
import pytest

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'prep')

def pytest_addoption(parser):
    parser.addoption(
        "--case", 
        action="store", 
        default=None, 
        help="Comma separated list of cases to test (e.g., '33,57' or 'all')"
    )

def pytest_generate_tests(metafunc):
    """Dynamically parametrize the 'case_name' fixture if it's used in a test."""
    if "case_name" in metafunc.fixturenames:
        requested = metafunc.config.getoption("case")
        
        # Determine available cases in data/prep
        available = []
        if os.path.isdir(PROCESSED_DIR):
            for d in os.listdir(PROCESSED_DIR):
                case_path = os.path.join(PROCESSED_DIR, d)
                if os.path.isdir(case_path) and os.path.exists(os.path.join(case_path, 'normalization.json')):
                    available.append(d)
        
        if requested:
            if requested.lower() == 'all':
                cases = available
            else:
                cases = [c.strip() if c.strip().startswith('case') else f"case{c.strip()}" 
                         for c in requested.split(',')]
                # Filter to only those that actually exist
                cases = [c for c in cases if c in available]
        else:
            cases = available

        metafunc.parametrize("case_name", cases)

@pytest.fixture
def requested_cases(request):
    """Fixture to get the --case argument from the CLI for non-parametrized tests."""
    val = request.config.getoption("--case")
    if val is None:
        return None
    if val.lower() == 'all':
        return 'all'
    
    cases = []
    for c in val.split(','):
        c = c.strip()
        if not c.startswith('case'):
            c = f"case{c}"
        cases.append(c)
    return cases
