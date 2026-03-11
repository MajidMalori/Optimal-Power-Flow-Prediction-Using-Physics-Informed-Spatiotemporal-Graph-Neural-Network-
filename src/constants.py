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

# Static Load Pattern (24-hour cycle) - Standard "Camel" Demand Shape
HOURLY_LOAD_PATTERN = {
    0: 0.40, 1: 0.35, 2: 0.32, 3: 0.30, 4: 0.35, 5: 0.45,
    6: 0.60, 7: 0.75, 8: 0.85, 9: 0.90, 10: 0.88, 11: 0.85,  # Morning Peak
    12: 0.82, 13: 0.80, 14: 0.82, 15: 0.85, 16: 0.88, 17: 0.92, # Midday Dip & Rise
    18: 1.00, 19: 0.98, 20: 0.90, 21: 0.75, 22: 0.60, 23: 0.50, # Evening Peak (Global High)
}

# Static Solar Pattern (24-hour cycle) - Standard Bell Curve
HOURLY_SOLAR_PATTERN = {
    0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.05,
    6: 0.15, 7: 0.30, 8: 0.50, 9: 0.70, 10: 0.85, 11: 0.95,
    12: 1.00, 13: 0.95, 14: 0.85, 15: 0.70, 16: 0.50, 17: 0.30, # Midday Peak
    18: 0.15, 19: 0.05, 20: 0.00, 21: 0.00, 22: 0.00, 23: 0.00,
}

# Static Wind Pattern (24-hour cycle) - Night-Peaking Coastal Profile
HOURLY_WIND_PATTERN = {
    0: 0.95, 1: 1.00, 2: 0.98, 3: 0.95, 4: 0.90, 5: 0.80,
    6: 0.70, 7: 0.65, 8: 0.60, 9: 0.55, 10: 0.50, 11: 0.45,
    12: 0.45, 13: 0.50, 14: 0.55, 15: 0.65, 16: 0.75, 17: 0.85, # Morning/Midday Dip
    18: 0.90, 19: 0.92, 20: 0.95, 21: 0.98, 22: 1.00, 23: 0.98, # Night Peaks
}
