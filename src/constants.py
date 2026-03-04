# --- Indices ---
class FeatureIndices:
    P_LOAD=0; Q_LOAD=1; P_EXT_GRID=2; Q_EXT_GRID=3; P_CONV=4; Q_CONV=5; P_REN=6; Q_REN=7; VM=8; VA=9; DEGREE=10
    FEATURE_NAMES = ['p_load', 'q_load', 'p_ext_grid', 'q_ext_grid', 'p_conv', 'q_conv', 'p_ren', 'q_ren', 'vm', 'va', 'degree']
    NUM_FEATURES = 11
    NUM_TARGETS = 10  # Added for direct import compatibility

class TargetIndices:
    P_LOAD=0; Q_LOAD=1; P_EXT_GRID=2; Q_EXT_GRID=3; P_CONV=4; Q_CONV=5; P_REN=6; Q_REN=7; VM=8; VA=9
    NUM_TARGETS = 10

class PredictionIndices:
    VM = 0
    VA = 1

# --- Training / Split Constants ---
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --- Physical & Validation Constants ---
V_GARBAGE_LOW = 0.5
V_GARBAGE_HIGH = 1.5
# Validation Margin Constants
GENERATOR_CAPACITY_MARGIN = 1.01
INVERTER_CAPACITY_MARGIN = 1.01
NEGATIVE_LOAD_CUTOFF = -1e-6
MAX_SLACK_MULTIPLIER = 5.0
MAX_SLACK_WARNING_MULTIPLIER = 2.0
MAX_LINE_LOADING_PERCENT = 1000
WARNING_LINE_LOADING_PERCENT = 100

# Physical Bounds Constants
ANGLE_GARBAGE_THRESHOLD = 1.57079632679  # pi/2 (90 deg)
ANGLE_WARNING_THRESHOLD = 0.78539816339  # 45 deg

# System Physics (Strict physical constants for specific grid topologies)
# These represent equipment ratings and normalization bases, not tunable parameters.
SYSTEM_PHYSICS = {
    'case33': {
        'base_mva': 10.0,
        'v_min': 0.85,  # ANSI C84.1 Range B or common distribution limits
        'v_max': 1.15
    },
    'case57': {
        'base_mva': 100.0,
        'v_min': 0.90,  # Transmission limits
        'v_max': 1.10
    },
    'case118': {
        'base_mva': 100.0,
        'v_min': 0.90,
        'v_max': 1.10
    },
    'default': {
        'base_mva': 100.0,
        'v_min': 0.90,
        'v_max': 1.10
    }
}

# Static Load Pattern (24-hour cycle)
HOURLY_LOAD_PATTERN = {
    0: 0.40, 1: 0.35, 2: 0.33, 3: 0.32, 4: 0.35, 5: 0.42,
    6: 0.55, 7: 0.70, 8: 0.85, 9: 0.90, 10: 0.92, 11: 0.95,
    12: 0.97, 13: 0.95, 14: 0.93, 15: 0.92, 16: 0.94, 17: 0.98,
    18: 1.00, 19: 0.98, 20: 0.90, 21: 0.80, 22: 0.65, 23: 0.50,
}
