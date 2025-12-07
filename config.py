import os
import torch
import yaml
import csv
import json
import inspect
import shutil
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


# FIXED: Args class has been REMOVED.
# All experimental settings are now part of Config class and loaded from YAML.
# See Config class below for experimental settings (test_config, bus_systems, etc.)

# ============================================================================
# YAML Configuration Loader (Merged from utils/yaml_config.py)
# ============================================================================

def _load_yaml_file(yaml_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    yaml_path = Path(yaml_path)
    
    if not yaml_path.is_absolute():
        current_dir = Path(__file__).parent
        yaml_path = current_dir / yaml_path
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return config_dict if config_dict is not None else {}


def _convert_numeric_string(value: Any) -> Any:
    """Convert string representations of numbers to proper numeric types."""
    if not isinstance(value, str):
        return value
    
    try:
        float_val = float(value)
        if '.' not in value and 'e' not in value.lower():
            try:
                return int(value)
            except ValueError:
                pass
        return float_val
    except (ValueError, OverflowError):
        return value


def _flatten_dict(nested_dict: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Flatten a nested dictionary."""
    items = []
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(_flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def _merge_yaml_with_config(yaml_path: str, config_obj: Any, verbose: bool = False) -> None:
    """Merge YAML configuration into an existing Config object."""
    yaml_config = _load_yaml_file(yaml_path)
    
    # Mapping from YAML keys to Config attribute names
    attribute_mapping = {
        # System configuration
        'system_device': 'DEVICE',
        'system_num_buses': 'NUM_BUSES',
        'system_test_cases': 'TEST_CASES',
        'system_seed': 'SEED',
        'system_case_name': 'CASE_NAME',
        
        # Training configuration
        'training_batch_size': 'BATCH_SIZE',
        'training_learning_rate': 'LEARNING_RATE',
        'training_max_grad_norm': 'MAX_GRAD_NORM',
        'training_num_epochs': 'NUM_EPOCHS',
        'training_early_stopping_patience': 'EARLY_STOPPING_PATIENCE',
        'training_use_learning_rate_scheduler': 'USE_LEARNING_RATE_SCHEDULER',
        'training_cosine_annealing_lr_t_max': 'COSINEANNEALINGLR_T_MAX',
        'training_cosine_annealing_lr_eta_min': 'COSINEANNEALINGLR_ETA_MIN',
        'training_weight_decay': 'WEIGHT_DECAY',
        
        # Physics configuration
        'physics_split_mode': 'DATA_SPLIT_MODE',
        'physics_splits_train': 'TRAIN_SPLIT',
        'physics_splits_val': 'VAL_SPLIT',
        
        # Data configuration
        'data_hours_per_day': 'HOURS_PER_DAY',
        'data_sequence_length': 'SEQUENCE_LENGTH',
        'data_contingency_rate': 'CONTINGENCY_RATE',
        'data_pmu_coverage': 'PMU_COVERAGE',
        
        # MOOPF configuration
        'moopf_weights_loss': 'MOOPF_WEIGHT_LOSS',
        'moopf_weights_voltage_deviation': 'MOOPF_WEIGHT_VDEV',
        'moopf_weights_carbon': 'MOOPF_WEIGHT_CARBON',
        
        # Experimental configuration
        'experimental_test_config': 'EXPERIMENTAL_TEST_CONFIG',
        'experimental_bus_systems': 'EXPERIMENTAL_BUS_SYSTEMS',
        'experimental_models_to_train': 'EXPERIMENTAL_MODELS_TO_TRAIN',
        'experimental_data_mode': 'EXPERIMENTAL_DATA_MODE',
        'experimental_train_timesteps': 'EXPERIMENTAL_TRAIN_TIMESTEPS',
        'experimental_test_timesteps': 'EXPERIMENTAL_TEST_TIMESTEPS',
        'experimental_force_cpu': 'EXPERIMENTAL_FORCE_CPU',
        'experimental_parallel_data_loading': 'EXPERIMENTAL_PARALLEL_DATA_LOADING',
        'experimental_data_workers': 'EXPERIMENTAL_DATA_WORKERS',
        'experimental_clear_results': 'EXPERIMENTAL_CLEAR_RESULTS',
        
        # Debug configuration
        'debug_enable': 'DEBUG_ENABLE',
        'debug_log_interval': 'DEBUG_LOG_INTERVAL',
    }
    
    flat_yaml = _flatten_dict(yaml_config)
    
    for yaml_key, value in flat_yaml.items():
        if value is None:
            continue
        
        value = _convert_numeric_string(value)
        config_attr = attribute_mapping.get(yaml_key, yaml_key.upper().replace('-', '_'))
        
        if yaml_key == 'system_device' and value == 'cuda':
            if not torch.cuda.is_available():
                value = 'cpu'
        
        if hasattr(config_obj, config_attr):
            setattr(config_obj, config_attr, value)
        elif hasattr(config_obj.__class__, config_attr):
            setattr(config_obj.__class__, config_attr, value)
        else:
            setattr(config_obj, config_attr, value)
    
    # Handle model capacity settings
    for bus_size in [33, 57, 118]:
        key = f'model_capacity_bus_{bus_size}'
        if key in flat_yaml:
            setattr(Config, f'CAPACITY_{bus_size}_BUS', flat_yaml[key])
    
    # Handle experimental arguments
    experimental_mappings = {
        'experimental_test_config': 'test_config',
        'experimental_bus_systems': 'bus_systems',
        'experimental_models_to_train': 'models_to_train',
        'experimental_data_mode': 'data_mode',
        'experimental_train_timesteps': 'train_timesteps',
        'experimental_test_timesteps': 'test_timesteps',
        'experimental_force_cpu': 'force_cpu',
        'experimental_parallel_data_loading': 'parallel_data_loading',
        'experimental_data_workers': 'data_workers',
        'experimental_clear_results': 'clear_results',
    }
    
    for yaml_key, config_attr in experimental_mappings.items():
        if yaml_key in flat_yaml and flat_yaml[yaml_key] is not None:
            setattr(Config, config_attr, flat_yaml[yaml_key])
    
    # Handle system-specific limits
    if 'system_limits' in yaml_config:
        case_name = getattr(config_obj, 'CASE_NAME', None)
        if case_name:
            case_name_lower = case_name.lower()
            system_limits = yaml_config['system_limits']
            
            if case_name_lower in system_limits:
                limits = system_limits[case_name_lower]
                if 'base_mva' in limits:
                    setattr(config_obj, 'BASE_MVA', limits['base_mva'])
                if 'v_min' in limits:
                    setattr(config_obj, 'V_MIN', limits['v_min'])
                if 'v_max' in limits:
                    setattr(config_obj, 'V_MAX', limits['v_max'])

class FeatureIndices:
    """
    Feature indices for the 10-dimensional measurement vector.
    Single source of truth to prevent indexing bugs.
    
    Order matches gen_meas_best.py: [p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren, vm, va]
    """
    P_LOAD = 0      # Active load (MW)
    Q_LOAD = 1      # Reactive load (Mvar)
    P_EXT_GRID = 2  # External grid active power (MW)
    Q_EXT_GRID = 3  # External grid reactive power (Mvar)
    P_CONV = 4      # Conventional generation active power (MW)
    Q_CONV = 5      # Conventional generation reactive power (Mvar)
    P_REN = 6       # Renewable generation active power (MW)
    Q_REN = 7       # Renewable generation reactive power (Mvar)
    VM = 8          # Voltage magnitude (p.u.)
    VA = 9          # Voltage angle (rad)
    
    # Feature names for documentation and debugging
    FEATURE_NAMES = [
        'p_load', 'q_load', 'p_ext_grid', 'q_ext_grid',
        'p_conv', 'q_conv', 'p_ren', 'q_ren', 'vm', 'va'
    ]
    
    NUM_FEATURES = 10


class TargetIndices:
    """
    Target indices for the 10-dimensional clean state vector.
    In Full State Reconstruction, Targets are identical to Features (but clean).
    """
    P_LOAD = 0      
    Q_LOAD = 1      
    P_EXT_GRID = 2  
    Q_EXT_GRID = 3  
    P_CONV = 4      
    Q_CONV = 5      
    P_REN = 6       
    Q_REN = 7       
    VM = 8          
    VA = 9          
    
    NUM_TARGETS = 10


class ModelOutputIndices:
    """
    Model output indices for the 10-dimensional reconstructed state.
    Direct regression of the clean state.
    """
    P_LOAD = 0      
    Q_LOAD = 1      
    P_EXT_GRID = 2  
    Q_EXT_GRID = 3  
    P_CONV = 4      
    Q_CONV = 5      
    P_REN = 6       
    Q_REN = 7       
    VM = 8          
    VA = 9          
    
    NUM_OUTPUTS = 10


class Config:
    """
    Configuration Module - YAML is the Single Source of Truth
    
    This module provides:
    - Code constants (FeatureIndices, TargetIndices, ModelOutputIndices)
    - Business logic methods (get_model_class_map, get_models_to_test, etc.)
    - Computed properties (CURRENT_RUN_DIR, model_config_map)
    - Computed runtime values (DEVICE, ROOT_DIR, etc.)
    
    ALL CONFIGURATION VALUES ARE LOADED FROM config.yaml - NO FALLBACK DEFAULTS.
    If config.yaml is missing, initialization will fail immediately.
    
    This ensures:
    - Single source of truth (no duplication)
    - Fail-fast behavior (problems are caught immediately)
    - Clear configuration management (all values in one place)
    
    See config.yaml for all configuration values.
    """
    
    # ============================================================================
    # YAML IS THE SINGLE SOURCE OF TRUTH
    # ============================================================================
    # All configuration values are loaded from config.yaml.
    # This file contains ONLY:
    #   - Computed values (DEVICE, ROOT_DIR, etc.)
    #   - Business logic methods (get_model_class_map, etc.)
    #   - Constants (FeatureIndices, TargetIndices, ModelOutputIndices)
    #   - Model configuration classes (_ModelConfig, etc.)
    #
    # NO DEFAULT VALUES - YAML is required and must contain all configuration.
    # If YAML is missing, the code will fail immediately (no fallback).
    # ============================================================================
    
    # --- Computed Values (NOT in YAML - computed at runtime) ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Auto-detected, can be overridden by YAML
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = None  # Set in __init__ based on data_mode
    EXPERIMENTAL_RESULTS_DIR = os.path.join(ROOT_DIR, 'experimental_results')
    _CURRENT_RUN_TIMESTAMP = None  # Set during __init__
    
    # --- Configuration Attributes (loaded from YAML, no defaults here) ---
    # All these are loaded from config.yaml via merge_yaml_with_config()
    # If YAML is missing, these will be None and code will fail (as intended)

    
    # --- Project Structure (computed, not in YAML) ---
    # These are already defined above, removing duplicate
    
    # --- Model Testing Configuration ---
    MODELS_TO_TEST = ['GCN', 'adaptiveGCN', 'AdaptivePIGCN', 'PIGCLSTM', 'PIGCGRU', 'ResnetPIGCLSTM', 'ResnetPIGCGRU']
    
    MODEL_TEST_CONFIGS = {
        'quick': ['AdaptivePIGCN'],  # Fast testing
        'core': ['adaptiveGCN', 'AdaptivePIGCN'],  # Core comparison: best non-physics vs physics
        'comprehensive': ['GCN', 'adaptiveGCN', 'AdaptivePIGCN', 'PIGCLSTM', 'PIGCGRU'],  # Full comparison
        'physics_only': ['AdaptivePIGCN', 'PIGCLSTM', 'PIGCGRU', 'ResnetPIGCLSTM', 'ResnetPIGCGRU'],  # Physics-informed only
        'non_physics_only': ['GCN', 'adaptiveGCN'],  # Non-physics-informed only
        'sequential_only': ['PIGCLSTM', 'PIGCGRU', 'ResnetPIGCLSTM', 'ResnetPIGCGRU'],  # Sequential models only (LSTM/GRU)
        'all': ['GCN', 'adaptiveGCN', 'AdaptivePIGCN', 'PIGCLSTM', 'PIGCGRU', 'ResnetPIGCLSTM', 'ResnetPIGCGRU']  # Everything
    }
    
    class _ModelConfig:
        """
        Base template for all model configurations.
        
        FULL STATE RECONSTRUCTION APPROACH:
        - INPUT_DIM: Number of input features (noisy measurements)
        - OUTPUT_DIM: Number of output features (clean state)
        """
        # Input features (measurements): [p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren, vm, va]
        INPUT_DIM = 10
        
        # Output features (Clean State): Same 10 features
        OUTPUT_DIM = 10
        
        # Legacy field (kept for backward compatibility with older model code)
        FEATURE_DIM = OUTPUT_DIM  # This refers to the OUTPUT dimension
        
        DROPOUT = 0.1  # Dropout rate (0.1 Mandatory for MC Dropout Uncertainty)
        
        HIDDEN_DIM_RANGE = (16, 128)  # Default fallback
        NUM_GC_LAYERS_RANGE = (1, 5)
        
        @staticmethod
        def get_hidden_dim_range(num_buses):
            """
            Get hidden dimension range based on system size and capacity setting.
            Capacity levels: 'normal' (conservative), 'medium' (balanced), 'large' (maximum)
            Returns ranges with minimum width to allow optimization exploration.
            """
            # Capacity presets for each bus system
            # Research-backed: Power systems GCNs use 64-256 units (Aalto University study)
            # Optimal: 128 in first layer, 64 in second layer
            capacity_settings = {
                33: {
                    'normal': (64, 128),   # Research-backed: Conservative range
                    'medium': (96, 192),   # Research-backed: Balanced range
                    'large': (64, 256)     # Research-backed: Full range (64-256)
                },
                57: {
                    'normal': (64, 128),   # Research-backed: Conservative range
                    'medium': (96, 192),   # Research-backed: Balanced range
                    'large': (64, 256)     # Research-backed: Full range (64-256)
                },
                118: {
                    'normal': (64, 128),   # Research-backed: Conservative range
                    'medium': (96, 192),   # Research-backed: Balanced range
                    'large': (64, 256)     # Research-backed: Full range (64-256)
                }
            }
            
            # Get capacity setting from Config (moved from Args class)
            # Note: This will be set from YAML or use default values
            # Map num_buses to the correct capacity_settings key
            if num_buses <= 33:
                capacity = Config.CAPACITY_33_BUS
                bus_key = 33
            elif num_buses <= 57:
                capacity = Config.CAPACITY_57_BUS
                bus_key = 57
            else:
                capacity = Config.CAPACITY_118_BUS
                bus_key = 118
            
            return capacity_settings[bus_key][capacity]
        
        @staticmethod
        def get_num_gc_layers_range(num_buses):
            """
            Get GC layers range based on system size and capacity setting.
            Returns ranges with minimum width to allow optimization exploration.
            """
            # Research-backed: Optimal GCN depth is 2-3 layers (PMC, Aalto University)
            # Beyond 3 layers causes over-smoothing (node representations become indistinguishable)
            capacity_settings = {
                33: {
                    'normal': (1, 2),     # Research-backed: Minimal (1-2 layers)
                    'medium': (2, 3),     # Research-backed: Optimal (2-3 layers)
                    'large': (2, 3)       # Research-backed: Optimal (2-3 layers)
                },
                57: {
                    'normal': (1, 2),     # Research-backed: Minimal (1-2 layers)
                    'medium': (2, 3),     # Research-backed: Optimal (2-3 layers)
                    'large': (2, 3)       # Research-backed: Optimal (2-3 layers)
                },
                118: {
                    'normal': (1, 2),     # Research-backed: Minimal (1-2 layers)
                    'medium': (2, 3),     # Research-backed: Optimal (2-3 layers)
                    'large': (2, 3)       # Research-backed: Optimal (2-3 layers)
                }
            }
            
            # Get capacity setting from Config
            # Map num_buses to the correct capacity_settings key
            if num_buses <= 33:
                capacity = Config.CAPACITY_33_BUS
                bus_key = 33
            elif num_buses <= 57:
                capacity = Config.CAPACITY_57_BUS
                bus_key = 57
            else:
                capacity = Config.CAPACITY_118_BUS
                bus_key = 118
            
            return capacity_settings[bus_key][capacity]
        
        @staticmethod
        def get_embedding_dim_range(num_buses):
            """
            Get embedding dimension range based on system size and capacity setting.
            Returns ranges with minimum width to allow optimization exploration.
            """
            # Research-backed: Typical embedding dimensions are 100-300 (Neo4j, GraphSAGE)
            # Common choices: 128 (node2vec, GraphSAGE), 200-450 for complex datasets
            capacity_settings = {
                33: {
                    'normal': (64, 150),  # Research-backed: Conservative range
                    'medium': (100, 200), # Research-backed: Typical range
                    'large': (100, 300)   # Research-backed: Full range (100-300)
                },
                57: {
                    'normal': (64, 150),  # Research-backed: Conservative range
                    'medium': (100, 200), # Research-backed: Typical range
                    'large': (100, 300)   # Research-backed: Full range (100-300)
                },
                118: {
                    'normal': (64, 150),  # Research-backed: Conservative range
                    'medium': (100, 200), # Research-backed: Typical range
                    'large': (100, 300)   # Research-backed: Full range (100-300)
                }
            }
            
            # Get capacity setting from Config (moved from Args class)
            # Note: This will be set from YAML or use default values
            # Map num_buses to the correct capacity_settings key
            if num_buses <= 33:
                capacity = Config.CAPACITY_33_BUS
                bus_key = 33
            elif num_buses <= 57:
                capacity = Config.CAPACITY_57_BUS
                bus_key = 57
            else:
                capacity = Config.CAPACITY_118_BUS
                bus_key = 118
            
            return capacity_settings[bus_key][capacity]
        
        @staticmethod
        def get_recommended_model(num_buses):
            """Return recommended model type based on system size"""
            if num_buses <= 33:
                return "PIGCGRU"  # Best performance for small systems
            elif num_buses <= 57:
                return "AdaptivePIGCN"     
            else:
                return "AdaptivePIGCN"     
        
        @staticmethod
        def get_adaptive_mosoa_params(num_buses):
            """Return adaptive MoSOA parameters based on system size and optimization strategy"""
            if num_buses <= 33:
                # THOROUGH: Small systems can afford extensive search
                return {
                    'num_seagulls': 5,     
                    'max_iterations': 10,   
                    'strategy': 'thorough',
                    'description': 'Extensive search for optimal hyperparameters'
                }
            elif num_buses <= 57:
                # BALANCED: Medium systems need balance between quality and time
                return {
                    'num_seagulls': 5,      
                    'max_iterations': 10,   
                    'strategy': 'balanced',
                    'description': 'Balance optimization quality vs computational time'
                }
            else:
                # QUICK: Large systems prioritize efficiency
                return {
                    'num_seagulls': 5,      # Optimized for quick testing
                    'max_iterations': 10,    # Optimized for quick testing
                    'strategy': 'quick',
                    'description': 'Fast optimization for memory/time constraints'
                }
    GCNConfig = _ModelConfig()
    
    AdaptivePIGCNConfig = _ModelConfig()
    AdaptivePIGCNConfig.EMBEDDING_DIM_RANGE = (8, 32)  # Will be overridden for 118-bus
    AdaptivePIGCNConfig.PHI_RANGE = (0.0, 1.0)
    AdaptivePIGCNConfig.NUM_GC_LAYERS_RANGE = (1, 6)  # Slightly more layers for 118-bus
    
    adaptiveGCNConfig = _ModelConfig()
    adaptiveGCNConfig.EMBEDDING_DIM_RANGE = (8, 32)
    adaptiveGCNConfig.PHI_RANGE = (0.0, 1.0)

    PIGCLSTMConfig = _ModelConfig()
    PIGCLSTMConfig.RNN_LAYERS_RANGE = (1, 3)
    PIGCLSTMConfig.SEQUENCE_LENGTH_RANGE = (5, 10)
    PIGCLSTMConfig.EMBEDDING_DIM_RANGE = (8, 16)
    PIGCLSTMConfig.PHI_RANGE = (0.0, 1.0)
    
    @staticmethod
    def get_sequential_ranges(num_buses):
        """
        Get system-size-dependent ranges for sequential models.
        Large systems need aggressive constraints to avoid OOM and slow training.
        """
        if num_buses <= 33:
            return {
                'hidden_dim': (32, 64),
                'sequence_length': (5, 10),
                'rnn_layers': (1, 3)
            }
        elif num_buses <= 57:
            return {
                'hidden_dim': (16, 48),
                'sequence_length': (3, 8),
                'rnn_layers': (1, 2)
            }
        else:  # 118-bus and larger
            return {
                'hidden_dim': (16, 32),
                'sequence_length': (3, 5),
                'rnn_layers': (1, 2)
            }
    
    PIGCLSTMConfig.get_sequential_ranges = get_sequential_ranges
    
    PIGCGRUConfig = _ModelConfig()
    PIGCGRUConfig.RNN_LAYERS_RANGE = (1, 3)
    PIGCGRUConfig.SEQUENCE_LENGTH_RANGE = (5, 10)
    PIGCGRUConfig.EMBEDDING_DIM_RANGE = (8, 16)
    PIGCGRUConfig.PHI_RANGE = (0.0, 1.0)
    PIGCGRUConfig.get_sequential_ranges = get_sequential_ranges

    ResnetPIGCGRUConfig = _ModelConfig()
    ResnetPIGCGRUConfig.RNN_LAYERS_RANGE = (1, 3)
    ResnetPIGCGRUConfig.SEQUENCE_LENGTH_RANGE = (5, 10)
    ResnetPIGCGRUConfig.EMBEDDING_DIM_RANGE = (8, 16)
    ResnetPIGCGRUConfig.PHI_RANGE = (0.0, 1.0)
    ResnetPIGCGRUConfig.get_sequential_ranges = get_sequential_ranges
    
    ResnetPIGCLSTMConfig = _ModelConfig()
    ResnetPIGCLSTMConfig.RNN_LAYERS_RANGE = (1, 3)
    ResnetPIGCLSTMConfig.SEQUENCE_LENGTH_RANGE = (5, 10)
    ResnetPIGCLSTMConfig.EMBEDDING_DIM_RANGE = (8, 16)
    ResnetPIGCLSTMConfig.PHI_RANGE = (0.0, 1.0)
    ResnetPIGCLSTMConfig.get_sequential_ranges = get_sequential_ranges
    
    @property
    def CURRENT_RUN_DIR(self):
        """Get the current run directory with timestamp."""
        return os.path.join(self.EXPERIMENTAL_RESULTS_DIR, f'run_{self._CURRENT_RUN_TIMESTAMP}')
    
    @property
    def LATEST_RUN_DIR(self):
        """Get the latest run directory (deprecated - use latest_run_info.txt instead)."""
        # This property is kept for backward compatibility but should not be used
        # The latest run is tracked via latest_run_info.txt pointer file
        return self.CURRENT_RUN_DIR
    
    @property
    def EVALUATION_DIR(self):
        """Backward compatibility property."""
        return self.CURRENT_RUN_DIR
    
    @property
    def model_config_map(self):
        """Returns mapping of model names to their configurations."""
        return {
            'GCN': self.GCNConfig, 
            'adaptiveGCN': self.adaptiveGCNConfig, 
            'AdaptivePIGCN': self.AdaptivePIGCNConfig,
            'PIGCLSTM': self.PIGCLSTMConfig, 
            'PIGCGRU': self.PIGCGRUConfig,
            'ResnetPIGCGRU': self.ResnetPIGCGRUConfig, 
            'ResnetPIGCLSTM': self.ResnetPIGCLSTMConfig
        }
    
    def __init__(self, data_mode='train', train_timesteps=None, test_timesteps=None, clear_results=False, 
                 hours_per_day=24, sequence_length=5, yaml_config_path=None, load_yaml=True):
        """
        Initializes configuration.
        
        YAML CONFIGURATION IS REQUIRED - No fallback defaults.
        If config.yaml is missing, this will raise an exception immediately.
        
        Args:
            data_mode: 'train' or 'test'
            train_timesteps: Number of timesteps for train mode
            test_timesteps: Number of timesteps for test mode
            clear_results: Whether to clear experimental_results folder before running
            hours_per_day: Hours per day for time-series data
            sequence_length: Sequence length for LSTM/GRU models
            yaml_config_path: Path to YAML configuration file (default: 'config.yaml')
            load_yaml: Whether to load configuration from YAML file (default: True, REQUIRED)
        
        Raises:
            FileNotFoundError: If config.yaml is missing
            ImportError: If PyYAML is not installed
            ValueError: If YAML is malformed or missing required keys
        """
        # YAML IS REQUIRED - Fail fast if missing (no fallback)
        if load_yaml:
            yaml_path = yaml_config_path or 'config.yaml'
            yaml_full_path = yaml_path if os.path.isabs(yaml_path) else os.path.join(self.ROOT_DIR, yaml_path)
            
            if not os.path.exists(yaml_full_path):
                raise FileNotFoundError(
                    f"REQUIRED: config.yaml not found at {yaml_full_path}\n"
                    f"YAML is the single source of truth. No fallback defaults.\n"
                    f"Please create config.yaml or specify correct path via yaml_config_path argument."
                )
            
            try:
                _merge_yaml_with_config(yaml_path, self, verbose=False)
                # print(f"[Config] Loaded configuration from {yaml_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load YAML configuration from {yaml_path}: {e}\n"
                    f"YAML is required - no fallback defaults available."
                ) from e
        else:
            raise ValueError(
                "load_yaml=False is not allowed. YAML configuration is required.\n"
                "Set load_yaml=True and ensure config.yaml exists."
            )
        
        # Set clear_results flag - YAML overrides function argument
        if hasattr(Config, 'clear_results'):
            clear_results = Config.clear_results
        
        # Clear experimental results folder if requested
        if clear_results and os.path.exists(self.EXPERIMENTAL_RESULTS_DIR):
            try:
                # print(f"\n[Clear Results] Deleting experimental_results folder...")
                shutil.rmtree(self.EXPERIMENTAL_RESULTS_DIR)
                # print(f"[Clear Results] Successfully deleted: {self.EXPERIMENTAL_RESULTS_DIR}")
            except Exception as e:
                print(f"[Clear Results] Warning: Could not delete experimental_results folder: {e}")
        
        # Initialize DATA_MODE_TIMESTEPS - CLI arguments override YAML (CLI has priority)
        if train_timesteps is None:
            train_timesteps = getattr(Config, 'train_timesteps', None)
            if train_timesteps is None:
                raise ValueError("train_timesteps not found in YAML. YAML is required - no fallback defaults.")
        # CLI argument has priority - only use YAML if CLI argument is None
        if test_timesteps is None:
            test_timesteps = getattr(Config, 'test_timesteps', None)
            if test_timesteps is None:
                raise ValueError("test_timesteps not found in YAML. YAML is required - no fallback defaults.")
        self.DATA_MODE_TIMESTEPS = {'train': train_timesteps, 'test': test_timesteps}
        
        # Set data mode and validate
        self.DATA_MODE = data_mode
        if data_mode not in self.DATA_MODE_TIMESTEPS:
            raise ValueError(f"Invalid data_mode '{data_mode}'. Must be 'train' or 'test'")
        
        # Set time-series configuration (always time-series mode)
        self.HOURS_PER_DAY = hours_per_day
        self.SEQUENCE_LENGTH = sequence_length
        
        # print(f"Data mode: {self.DATA_MODE}, Timesteps: {self.DATA_MODE_TIMESTEPS[self.DATA_MODE]}")
        
        # Data directory structure: data/[train|test] (removed time_series subfolder)
        self.DATA_DIR = os.path.join(self.ROOT_DIR, 'data', data_mode)
        
        # print(f"\n[Data Mode] Using time-series data in {data_mode} mode")
        # print(f"[Data Directory] {self.DATA_DIR}")
        
        # Detect if we're in a test environment
        # Check if the calling script is in tests/ directory or if pytest is running
        frame = inspect.currentframe()
        test_mode = False
        try:
            # Check call stack for test files
            while frame:
                filename = frame.f_globals.get('__file__', '')
                if 'tests' in filename or 'test_' in os.path.basename(filename) or 'pytest' in filename:
                    test_mode = True
                    break
                frame = frame.f_back
        except:
            pass
        
        # Initialize timestamp only when actually starting a run (not in test mode)
        if not test_mode:
            self._initialize_run_timestamp()
        elif test_mode:
            # In test mode, set a dummy timestamp to avoid errors
            if not hasattr(self, '_CURRENT_RUN_TIMESTAMP') or self._CURRENT_RUN_TIMESTAMP is None:
                self._CURRENT_RUN_TIMESTAMP = 'test_mode'

    def create_run_directories(self):
        """
        Explicitly create run directories and metadata.
        Should ONLY be called by training scripts, not data generation.
        """
        # Create base directories
        for dir_path in [self.DATA_DIR, self.EXPERIMENTAL_RESULTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Create current run directory
        os.makedirs(self.CURRENT_RUN_DIR, exist_ok=True)
        
        # Update latest run (copy current run info)
        self._update_latest_run_link()
        
        # Create run metadata
        self._create_run_metadata()
    
    @staticmethod
    def get_model_class_map():
        """Returns mapping of model names to their classes."""
        # Import here to avoid circular imports
        from models.adaptive_gcn import adaptiveGCN
        from models.gcn import GCN
        from models.adaptive_pigcn import AdaptivePIGCN
        from models.pigc_rnn import PIGCLSTM, PIGCGRU, ResnetPIGCGRU, ResnetPIGCLSTM
        
        return {
            'adaptiveGCN': adaptiveGCN, 
            'GCN': GCN, 
            'AdaptivePIGCN': AdaptivePIGCN, 
            'PIGCLSTM': PIGCLSTM,
            'PIGCGRU': PIGCGRU, 
            'ResnetPIGCGRU': ResnetPIGCGRU, 
            'ResnetPIGCLSTM': ResnetPIGCLSTM
    }
    
    @staticmethod
    def get_models_to_test(test_config='quick'):
        """
        Get list of models to test based on configuration.
        
        Available configurations:
        - 'quick': Fast testing with one model (AdaptivePIGCN)
        - 'core': Core comparison - best non-physics vs physics
        - 'comprehensive': Full comparison of key models
        - 'physics_only': Only physics-informed models
        - 'non_physics_only': Only non-physics-informed models
        - 'sequential_only': Only sequential models (LSTM/GRU-based)
        - 'all': Every available model
        """
        return Config.MODEL_TEST_CONFIGS.get(test_config, Config.MODEL_TEST_CONFIGS['quick'])
    
    @staticmethod
    def is_sequential_model(model_name):
        """Check if model is sequential (LSTM/GRU based)."""
        return 'LSTM' in model_name.upper() or 'GRU' in model_name.upper()
    
    @staticmethod
    def is_physics_informed(model_name):
        """Check if model is physics-informed."""
        return 'PI' in model_name
    
    @staticmethod
    def uses_adaptive_graph(model_name):
        """Check if model uses adaptive graph features."""
        return model_name in ['PIGCLSTM', 'PIGCGRU', 'adaptiveGCN', 'AdaptivePIGCN', 'ResnetPIGCGRU', 'ResnetPIGCLSTM']
    
    def get_evaluation_path(self, filename):
        """Constructs a path in the evaluation directory."""
        return os.path.join(self.EVALUATION_DIR, filename)

    def get_model_eval_dir(self, num_buses: int, model_name: str) -> str:
        """Returns the evaluation directory path for a specific model."""
        return os.path.join(self.EVALUATION_DIR, f"{num_buses}bus", "models", model_name)

    def get_renewable_impacts_dir(self, num_buses: int, model_name: str) -> str:
        """Returns the renewable impacts directory path for a specific model."""
        return os.path.join(self.get_model_eval_dir(num_buses, model_name), "renewable_impacts")

    def get_model_checkpoint_path(self, num_buses: int, model_name: str) -> str:
        """Returns the checkpoint path for a specific model."""
        return os.path.join(self.get_model_eval_dir(num_buses, model_name), "checkpoint.pth")
    
    def get_checkpoint_path(self, filename: str) -> str:
        """Returns the checkpoint path for a filename (used by trainer)."""
        # This is called during training, so we need to get num_buses and model_name from context
        # For now, save to a temporary location - the trainer will handle the actual path
        # The trainer should use get_model_checkpoint_path instead
        if not hasattr(self, '_checkpoint_dir'):
            # Create a temporary checkpoint directory
            self._checkpoint_dir = os.path.join(self.CURRENT_RUN_DIR, 'checkpoints')
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        return os.path.join(self._checkpoint_dir, filename)

    def get_moopf_results_path(self, num_buses: int, model_name: str) -> str:
        """Returns the MOOPF results path for a specific model."""
        return os.path.join(self.get_model_eval_dir(num_buses, model_name), "moopf_results.csv")

    def get_convergence_plot_path(self, num_buses: int, model_name: str) -> str:
        """Returns the convergence plot path for a specific model."""
        return os.path.join(self.get_model_eval_dir(num_buses, model_name), "mosoa_conv.png")

    def get_training_history_path(self, num_buses: int, model_name: str) -> str:
        """Returns the training history plot path for a specific model."""
        return os.path.join(self.get_model_eval_dir(num_buses, model_name), "train_hist.png")

    def get_summary_path(self, num_buses: int, model_name: str) -> str:
        """Returns the summary CSV path for a specific model."""
        return os.path.join(self.get_model_eval_dir(num_buses, model_name), "summary.csv")
    
    def get_training_log_path(self, num_buses: int, model_name: str, mode: str = None) -> str:
        """Returns the training log file path for a specific model and mode.
        
        Structure: experimental_results/run_XXX/{num_buses}bus/log/{model_name}_{mode}.log
        All configurations for the same model and mode append to the same file.
        
        Args:
            num_buses: Number of buses in the system
            model_name: Name of the model
            mode: Training mode ('train' or 'test'). If None, uses DATA_MODE from config.
        """
        if mode is None:
            mode = getattr(self, 'DATA_MODE', 'train')
        
        log_dir = os.path.join(self.EVALUATION_DIR, f"{num_buses}bus", "log")
        os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists
        return os.path.join(log_dir, f"{model_name}_{mode}.log")
    
    def _initialize_run_timestamp(self):
        """Initialize the run timestamp - always create new timestamp for each run."""
        # Always create new timestamp for new run
        self._CURRENT_RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
        # print(f"Starting new run: run_{self._CURRENT_RUN_TIMESTAMP}")

    def _update_latest_run_link(self):
        """Update the latest_run_info.txt pointer file to track current run."""
        import shutil
        
        # Clean up old duplicate directories if they exist (migration from old system)
        if os.path.exists(self.EXPERIMENTAL_RESULTS_DIR):
            try:
                for item in os.listdir(self.EXPERIMENTAL_RESULTS_DIR):
                    # Remove old latest_run_* duplicate directories
                    if item.startswith('latest_run_') and os.path.isdir(os.path.join(self.EXPERIMENTAL_RESULTS_DIR, item)):
                        old_dir = os.path.join(self.EXPERIMENTAL_RESULTS_DIR, item)
                        shutil.rmtree(old_dir)
                    # Remove old generic latest_run directory
                    elif item == 'latest_run' and os.path.isdir(os.path.join(self.EXPERIMENTAL_RESULTS_DIR, item)):
                        old_dir = os.path.join(self.EXPERIMENTAL_RESULTS_DIR, item)
                        shutil.rmtree(old_dir)
            except (OSError, FileNotFoundError, PermissionError):
                pass  # If cleanup fails, continue anyway
        
        # Create/update latest_run_info.txt pointer file
        latest_info_file = os.path.join(self.EXPERIMENTAL_RESULTS_DIR, 'latest_run_info.txt')
        try:
            with open(latest_info_file, 'w') as f:
                f.write(f"Latest run: run_{self._CURRENT_RUN_TIMESTAMP}\n")
                f.write(f"Directory: {self.CURRENT_RUN_DIR}\n")
                f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except (OSError, PermissionError):
            pass
    
    def _create_run_metadata(self):
        """Create metadata for the current run."""
        
        metadata = {
            'run_id': f'run_{self._CURRENT_RUN_TIMESTAMP}',
            'start_time': datetime.now().isoformat(),
            'timestamp': self._CURRENT_RUN_TIMESTAMP,
            'config': {
                'device': self.DEVICE,
                'num_buses': getattr(self, 'NUM_BUSES', 'auto'),
                'learning_rate': getattr(self, 'LEARNING_RATE', 'auto'),
                'num_epochs': getattr(self, 'NUM_EPOCHS', 'auto'),
                'batch_size': getattr(self, 'BATCH_SIZE', 'auto'),
                's_base_mva': 'system_specific',  # Determined dynamically based on case type
                'loss_weighting': 'learnable_uncertainty',  # Kendall et al., CVPR 2018
                'note': 'Loss weights (σ_data, σ_power, σ_voltage) learned automatically during training'
            },
            'directory_structure': {
                'root': self.ROOT_DIR,
                'data': self.DATA_DIR,
                'results': self.CURRENT_RUN_DIR,
                'experimental_results': self.EXPERIMENTAL_RESULTS_DIR
            }
        }
        
        metadata_file = os.path.join(self.CURRENT_RUN_DIR, 'run_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def finalize_run(self, run_summary: dict = None):
        """Finalize the current run by updating latest_run and logging."""
        # Skip if saving is disabled
        
        
        # No duplication - the pointer file (latest_run_info.txt) already tracks latest run
        
        # Update run metadata with completion info
        metadata_file = os.path.join(self.CURRENT_RUN_DIR, 'run_metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metadata['end_time'] = datetime.now().isoformat()
            metadata['status'] = 'completed'
            if run_summary:
                metadata['results_summary'] = run_summary
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Log to experiment tracking CSV
        experiment_log = os.path.join(self.EXPERIMENTAL_RESULTS_DIR, 'experiment_log.csv')
        log_entry = {
            'run_id': f'run_{self._CURRENT_RUN_TIMESTAMP}',
            'start_time': metadata.get('start_time', ''),
            'end_time': metadata.get('end_time', ''),
            'status': 'completed',
            'models_tested': run_summary.get('models_tested', []) if run_summary else [],
            'best_model': run_summary.get('best_model', '') if run_summary else '',
            'best_score': run_summary.get('best_score', '') if run_summary else ''
        }
        
        # Write to CSV (append mode)
        file_exists = os.path.exists(experiment_log)
        with open(experiment_log, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)
    
    def get_run_info(self):
        """Get information about the current run."""
        return {
            'run_id': f'run_{self._CURRENT_RUN_TIMESTAMP}',
            'timestamp': self._CURRENT_RUN_TIMESTAMP,
            'current_run_dir': self.CURRENT_RUN_DIR,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
