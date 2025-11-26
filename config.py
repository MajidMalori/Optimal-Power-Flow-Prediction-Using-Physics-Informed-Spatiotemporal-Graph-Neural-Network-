import os
import torch
from datetime import datetime

# FIXED: Args class has been REMOVED.
# All experimental settings are now part of Config class and loaded from YAML.
# See Config class below for experimental settings (test_config, bus_systems, etc.)

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
    Target indices for the 2-dimensional OPF unknown vector.
    
    In OPF mode, targets are bus-type dependent:
    - PQ bus (0): [V, θ] - both unknown
    - PV bus (1): [Q, θ] - V is known, Q and θ are predicted
    - Slack bus (2): [P, Q] - V and θ are known, P and Q are predicted
    """
    VAR1 = 0  # First unknown (bus-type dependent)
    VAR2 = 1  # Second unknown (bus-type dependent)
    
    NUM_TARGETS = 2


class ModelOutputIndices:
    """
    Model output indices for the 4-dimensional natural parameter vector.
    
    Heteroscedastic uncertainty with natural parametrization (Immer et al., NeurIPS 2023):
    [η1_var1, η1_var2, f2_var1, f2_var2]
    """
    ETA1_VAR1 = 0  # Natural parameter η1 for variable 1
    ETA1_VAR2 = 1  # Natural parameter η1 for variable 2
    F2_VAR1 = 2    # f2 for variable 1 (used to compute η2 via softplus)
    F2_VAR2 = 3    # f2 for variable 2 (used to compute η2 via softplus)
    
    NUM_OUTPUTS = 4


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
        
        OPF APPROACH:
        - INPUT_DIM: Number of input features (measurements from sensors)
        - OUTPUT_DIM: Number of output features (OPF unknowns: 2 per bus, bus-type dependent)
        - FEATURE_DIM: Legacy name for OUTPUT_DIM (kept for backward compatibility)
        """
        # Input features (measurements): [p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren, vm_partial, va_partial]
        INPUT_DIM = 10
        
        # Output features (OPF unknowns): 2 per bus (PQ: V,θ | PV: Q,θ | Slack: P,Q)
        OUTPUT_DIM = 2
        
        # Legacy field (kept for backward compatibility with older model code)
        FEATURE_DIM = OUTPUT_DIM  # This refers to the OUTPUT dimension
        
        DROPOUT = 0.2  # Dropout rate (0.0 = disabled, 0.1-0.3 = typical, 0.5+ = aggressive)
        # Higher dropout = stronger regularization = less overfitting but may underfit
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
            # Single values are converted to ranges with ±10% tolerance for optimization
            capacity_settings = {
                33: {
                    'normal': (32, 64),    # Reduced: 32-64 (was 32-96) - capped at 64
                    'medium': (48, 96),    # Reduced: 48-96 (was 64-128)
                    'large': (64, 128)     # Reduced: 64-128 (was 96-160) - capped at 128
                },
                57: {
                    'normal': (32, 64),    # Reduced: 32-64 (was 32-96) - capped at 64
                    'medium': (48, 96),    # Reduced: 48-96 (was 64-128)
                    'large': (64, 128)     # Reduced: 64-128 (was 96-160) - capped at 128
                },
                118: {
                    'normal': (64, 128),   # Kept: 64-128 (capped at 128 as requested)
                    'medium': (96, 160),   # Kept: 96-160
                    'large': (128, 256)    # Kept: 128-256
                }
            }
            
            # Get capacity setting from Config (moved from Args class)
            # Note: This will be set from YAML or use default values
            if num_buses <= 33:
                capacity = Config.CAPACITY_33_BUS
            elif num_buses <= 57:
                capacity = Config.CAPACITY_57_BUS
            else:
                capacity = Config.CAPACITY_118_BUS
            
            return capacity_settings[num_buses][capacity]
        
        @staticmethod
        def get_num_gc_layers_range(num_buses):
            """
            Get GC layers range based on system size and capacity setting.
            Returns ranges with minimum width to allow optimization exploration.
            """
            capacity_settings = {
                33: {
                    'normal': (1, 4),     # Reverted: Allow original range
                    'medium': (2, 5),     # Reverted: Allow original range
                    'large': (3, 7)       # Reverted: Allow original range
                },
                57: {
                    'normal': (1, 4),     # Reduced: 1-4 (was 1-6)
                    'medium': (2, 5),     # Reduced: 2-5 (was 3-7)
                    'large': (3, 7)       # Reduced: 3-7 (was 5-9)
                },
                118: {
                    'normal': (2, 6),     # Reduced: 2-6 (was 2-8)
                    'medium': (4, 9),     # Kept: 4-9
                    'large': (6, 12)      # Kept: 6-12
                }
            }
            
            # Get capacity setting from Config (moved from Args class)
            # Note: This will be set from YAML or use default values
            if num_buses <= 33:
                capacity = Config.CAPACITY_33_BUS
            elif num_buses <= 57:
                capacity = Config.CAPACITY_57_BUS
            else:
                capacity = Config.CAPACITY_118_BUS
            
            return capacity_settings[num_buses][capacity]
        
        @staticmethod
        def get_embedding_dim_range(num_buses):
            """
            Get embedding dimension range based on system size and capacity setting.
            Returns ranges with minimum width to allow optimization exploration.
            """
            capacity_settings = {
                33: {
                    'normal': (8, 24),    # Reduced: 8-24 (was 8-48)
                    'medium': (24, 48),   # Reduced: 24-48 (was 32-64)
                    'large': (32, 64)     # Reduced: 32-64 (was 48-96) - capped at 64
                },
                57: {
                    'normal': (8, 24),    # Reduced: 8-24 (was 8-48)
                    'medium': (24, 48),   # Reduced: 24-48 (was 32-64)
                    'large': (32, 64)     # Reduced: 32-64 (was 48-96) - capped at 64
                },
                118: {
                    'normal': (16, 48),   # Reduced: 16-48 (was 16-64)
                    'medium': (48, 96),   # Kept: 48-96
                    'large': (64, 128)    # Kept: 64-128
                }
            }
            
            # Get capacity setting from Config (moved from Args class)
            # Note: This will be set from YAML or use default values
            if num_buses <= 33:
                capacity = Config.CAPACITY_33_BUS
            elif num_buses <= 57:
                capacity = Config.CAPACITY_57_BUS
            else:
                capacity = Config.CAPACITY_118_BUS
            
            return capacity_settings[num_buses][capacity]
        
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
                    'num_seagulls': 2,     
                    'max_iterations': 2,   
                    'strategy': 'thorough',
                    'description': 'Extensive search for optimal hyperparameters'
                }
            elif num_buses <= 57:
                # BALANCED: Medium systems need balance between quality and time
                return {
                    'num_seagulls': 2,      
                    'max_iterations': 2,   
                    'strategy': 'balanced',
                    'description': 'Balance optimization quality vs computational time'
                }
            else:
                # QUICK: Large systems prioritize efficiency
                return {
                    'num_seagulls': 2,      # Temporarily set to 4 for quick testing
                    'max_iterations': 2,    # Temporarily set to 5 for quick testing
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
    
    def __init__(self, data_mode='train', save_results=True, train_timesteps=None, test_timesteps=100, clear_results=False, 
                 hours_per_day=24, sequence_length=5, yaml_config_path=None, load_yaml=True):
        """
        Initializes directories and sets up experimental run structure.
        
        YAML CONFIGURATION IS REQUIRED - No fallback defaults.
        If config.yaml is missing, this will raise an exception immediately.
        
        Args:
            data_mode: 'train' or 'test'
            save_results: Whether to save results to files
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
            from utils.yaml_config import merge_yaml_with_config
            yaml_path = yaml_config_path or 'config.yaml'
            yaml_full_path = yaml_path if os.path.isabs(yaml_path) else os.path.join(self.ROOT_DIR, yaml_path)
            
            if not os.path.exists(yaml_full_path):
                raise FileNotFoundError(
                    f"REQUIRED: config.yaml not found at {yaml_full_path}\n"
                    f"YAML is the single source of truth. No fallback defaults.\n"
                    f"Please create config.yaml or specify correct path via yaml_config_path argument."
                )
            
            try:
                merge_yaml_with_config(yaml_path, self, verbose=False)
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
        
        # Set save_results flag (from argument, can be overridden by YAML)
        self.SAVE_RESULTS = save_results
        
        # Set data plotting flag from YAML (required, no fallback)
        self.PLOT_DATA_INFO = getattr(Config, 'plot_data_info', None)
        if self.PLOT_DATA_INFO is None:
            raise ValueError("plot_data_info not found in YAML. YAML is required - no fallback defaults.")
        # Backward compatibility: also set GENERATE_DATA_PROFILE_STORY (deprecated, use PLOT_DATA_INFO)
        self.GENERATE_DATA_PROFILE_STORY = self.PLOT_DATA_INFO
        
        # Clear experimental results folder if requested
        if clear_results and os.path.exists(self.EXPERIMENTAL_RESULTS_DIR):
            import shutil
            try:
                # print(f"\n[Clear Results] Deleting experimental_results folder...")
                shutil.rmtree(self.EXPERIMENTAL_RESULTS_DIR)
                # print(f"[Clear Results] Successfully deleted: {self.EXPERIMENTAL_RESULTS_DIR}")
            except Exception as e:
                print(f"[Clear Results] Warning: Could not delete experimental_results folder: {e}")
        
        # Initialize DATA_MODE_TIMESTEPS - use argument if provided, otherwise from YAML (required)
        if train_timesteps is None:
            train_timesteps = getattr(Config, 'train_timesteps', None)
            if train_timesteps is None:
                raise ValueError("train_timesteps not found in YAML. YAML is required - no fallback defaults.")
        test_timesteps_yaml = getattr(Config, 'test_timesteps', None)
        if test_timesteps_yaml is not None:
            test_timesteps = test_timesteps_yaml  # Override argument with YAML value
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
        
        # Initialize timestamp only when actually starting a run
        self._initialize_run_timestamp()
        
        # Only create directories if saving results
        if self.SAVE_RESULTS:
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
        from models.pigclstm import PIGCLSTM
        from models.pigcgru import PIGCGRU
        from models.ResnetPIGCGRU import ResnetPIGCGRU
        from models.ResnetPIGCLSTM import ResnetPIGCLSTM
        
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
        import json
        
        metadata = {
            'run_id': f'run_{self._CURRENT_RUN_TIMESTAMP}',
            'start_time': datetime.now().isoformat(),
            'timestamp': self._CURRENT_RUN_TIMESTAMP,
            'config': {
                'device': self.DEVICE,
                'num_buses': self.NUM_BUSES,
                'learning_rate': self.LEARNING_RATE,
                'num_epochs': self.NUM_EPOCHS,
                'batch_size': self.BATCH_SIZE,
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
        if not self.SAVE_RESULTS:
            return
        
        import json
        import csv
        
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
        
        print(f"Run finalized: {self.CURRENT_RUN_DIR}")
        print(f"Experiment logged to: {experiment_log}")
    
    def get_run_info(self):
        """Get information about the current run."""
        return {
            'run_id': f'run_{self._CURRENT_RUN_TIMESTAMP}',
            'timestamp': self._CURRENT_RUN_TIMESTAMP,
            'current_run_dir': self.CURRENT_RUN_DIR,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
