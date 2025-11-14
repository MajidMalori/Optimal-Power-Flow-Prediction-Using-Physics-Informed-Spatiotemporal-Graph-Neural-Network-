import os
import torch
from datetime import datetime

class Args:
    """
    Training arguments and configuration.
    Centralized location for all training-related parameters.
    
    QUICK ACCESS - Modify these for your experiments
    """
    # === MODEL & SYSTEM CONFIGURATION ===
    test_config = 'all'  # Options: 'quick', 'core', 'comprehensive', 'physics_only', 'non_physics_only', 'sequential_only', 'all'
    bus_systems = 'all'  # Options: 'all', '33', '57', '118', or comma-separated like '33,57'
    models_to_train = 'all'  # Options: 'all', 'PIGCLSTM', 'PIGCGRU', 'ResnetPIGCLSTM', 'ResnetPIGCGRU', or comma-separated like 'PIGCLSTM,PIGCGRU'
    seed = 42
    
    # === DATA CONFIGURATION ===
    data_mode = 'train'  # Options: 'train' or 'test'
    train_timesteps = 12000  # Number of timesteps for train mode (500 days = 300+100+100 for 60/20/20 split)
    test_timesteps = 960  # Number of timesteps for test mode (45 complete days = 27+9+9 for 60/20/20 split)
    
    # Data plotting configuration
    plot_data_info = True  # True: Generate data validation and profile story plots in bus folders, False: Skip (faster)
    # When True, generates:
    # - Data validation plots (if any) in each bus folder
    # - Data profile story plots (load/generation profiles, integrity checks) in each bus folder
    
    # ┌──────────────────────────────────────────────────────────────────────────┐
    # │ TIMESTEP REFERENCE TABLE (60/20/20 split with complete 24-hour cycles)  │
    # ├────────────────┬───────────┬──────┬───────────┬──────────┬──────────────┤
    # │ Purpose        │ Timesteps │ Days │ Train     │ Val      │ Test         │
    # ├────────────────┼───────────┼──────┼───────────┼──────────┼──────────────┤
    # │ Quick test     │    960    │  40  │ 24 days   │  8 days  │  8 days      │
    # │ Recommended    │   1080    │  45  │ 27 days   │  9 days  │  9 days  ← ✓ │
    # │ Thorough test  │   1200    │  50  │ 30 days   │ 10 days  │ 10 days      │
    # │ Light train    │   9000    │ 375  │ 225 days  │ 75 days  │ 75 days      │
    # │ Recommended    │  10080    │ 420  │ 252 days  │ 84 days  │ 84 days  ← ✓ │
    # │ Heavy train    │  12000    │ 500  │ 300 days  │ 100 days │ 100 days     │
    # └────────────────┴───────────┴──────┴───────────┴──────────┴──────────────┘
    #       Timesteps must be multiples of 120 (5 days × 24 hours) for 60/20/20 split.
    #       Use calculate_timesteps.py to compute custom configurations.
    
    # Time-series data configuration
    hours_per_day = 24
    sequence_length = 5     # Sequence length for LSTM/GRU models (past N hours to predict current)
    
    # === MODEL CAPACITY CONFIGURATION ===
    # Control model size per bus system for experimentation
    # Options: 'normal' (conservative), 'medium' (balanced), 'large' (maximum capacity)
    #
    # Capacity Presets (REDUCED RANGES for smaller, faster models):
    # ┌─────────┬────────────────────┬──────────────────────┬─────────────────────┐
    # │ System  │ normal             │ medium               │ large               │
    # ├─────────┼────────────────────┼──────────────────────┼─────────────────────┤
    # │ 33-bus  │ H:32-64, GC:1-4    │ H:48-96, GC:2-5     │ H:64-128, GC:3-7    │
    # │ 57-bus  │ H:32-64, GC:1-4    │ H:48-96, GC:2-5     │ H:64-128, GC:3-7    │
    # │ 118-bus │ H:64-128, GC:2-6   │ H:96-160, GC:4-9    │ H:128-256, GC:6-12  │
    # └─────────┴────────────────────┴──────────────────────┴─────────────────────┘
    # H=Hidden_dim, GC=GC_layers, E=Embedding_dim
    # Reduced normal capacity: 33/57-bus capped at 64, 118-bus capped at 128
    #
    CAPACITY_33_BUS = 'normal'   # 33-bus: reduced to H:32-64, GC:1-4 (was 32-96, 1-6)
    CAPACITY_57_BUS = 'normal'   # 57-bus: reduced to H:32-64, GC:1-4 (was 32-96, 1-6)
    CAPACITY_118_BUS = 'large'  # 118-bus: reduced to H:64-128, GC:2-6 (was 64-128, 2-8
    
    # === RESULTS SAVING CONFIGURATION ===
    save_results = True  # False: No files saved (console output only), True: Save all results
    clear_results = True  # True: Delete experimental_results folder before running, False: Keep old results
    
    # === HYPERPARAMETER OPTIMIZATION CONFIGURATION ===
    use_mosoa = True  # True: Use MoSOA from paper, False: Use trial-based search (faster)
    num_trials = 20  # Only used if use_mosoa=False
    
    # === PARALLEL TRAINING CONFIGURATION ===
    # Device configuration
    force_cpu = False  # Set to True to force CPU training even if GPU is available
    
    # Parallel training mode
    parallel_data_loading = True   # DISABLED for low-RAM systems (< 2GB available)
    
    # Worker configuration (auto-configured based on device if set to 'auto')
    data_workers = 'auto'         # Number of data loading workers

class Config:
    """
    Main configuration class for the project.
    Contains global settings and nested classes for model-specific hyperparameters.
    """
    
    # --- Device & System Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_BUSES = [33, 57, 118]
    SEED = 42
    NUM_WORKERS = 2  # Conservative default - will be auto-configured based on system capabilities
    
    # --- Training Parameters ---
    BATCH_SIZE = 64  # Default, will be overridden by adaptive function
    LEARNING_RATE = 0.0005  # Best from LR testing: 0.0005 works well for both GCN and AdaptivePIGCN  # Increased from 0.001 to help escape local minima (model was stuck at plateau)
    NUM_EPOCHS = 10  # Testing medium vs large capacity (set to 200 for full training)
    EARLY_STOPPING_PATIENCE = 75  # Increased to prevent premature stopping on 118-bus
    
    # --- Gradient Accumulation Configuration ---
    # Gradient accumulation simulates larger batch sizes without storing all gradients at once
    # If False: Increases actual batch size (faster but uses more memory)
    # If True: Uses smaller batches with accumulation (slower but uses less memory)
    USE_GRADIENT_ACCUMULATION = False  # Set to False to disable and use larger batches (requires more memory)
    
    # --- Learning Rate Scheduler Configuration ---
    # TEST RESULTS (Nov 2025): OneCycleLR causes up-down-up MSE pattern (scores: -75 to -99)
    # Best configurations:
    #   - GCN: StepLR + EB_On_Conservative (score: 69.49, decreasing: True, no overfitting)
    #   - AdaptivePIGCN: CosineAnnealingLR + EB_On_Aggressive (score: 51.75, decreasing: True, no overfitting)
    # RECOMMENDATION: Disable OneCycleLR, use StepLR for GCN, CosineAnnealingLR for AdaptivePIGCN
    USE_LEARNING_RATE_SCHEDULER = False  # DISABLED: OneCycleLR causes instability. Use StepLR/CosineAnnealingLR instead.
    ONECYCLE_MAX_LR = None  # None: Auto-calculate as 10x LEARNING_RATE (recommended)
    ONECYCLE_PCT_START = 0.3  # 30% of cycle for warmup (increasing phase)
    ONECYCLE_DIV_FACTOR = 25.0  # Initial LR = max_lr / div_factor (smooth start)
    ONECYCLE_FINAL_DIV_FACTOR = 10000.0  # Final LR = max_lr / final_div_factor (very low at end)
    
    # Alternative scheduler settings (now implemented in trainer)
    # Options: 'StepLR', 'CosineAnnealingLR', 'ExponentialLR', 'ReduceLROnPlateau', 'OneCycleLR', None
    # The selected scheduler will be used for ALL models
    # Recommended: 'StepLR' for GCN, 'CosineAnnealingLR' for AdaptivePIGCN (based on test results)
    LR_SCHEDULER_TYPE = 'StepLR'  # Set to None to disable scheduler, or specify scheduler type
    STEPLR_STEP_SIZE = 7  # Decay LR every N epochs
    STEPLR_GAMMA = 0.5  # Multiply LR by gamma at each step
    COSINEANNEALINGLR_T_MAX = 20  # Maximum number of iterations (epochs)
    COSINEANNEALINGLR_ETA_MIN = 1e-6  # Minimum learning rate
    
    # --- Weight Decay (L2 Regularization) Configuration ---
    USE_WEIGHT_DECAY = False  # True: Enable L2 regularization, False: Disable
    WEIGHT_DECAY = 1e-5  # Weight decay coefficient (only used if USE_WEIGHT_DECAY=True)
    
    TRAIN_SPLIT = 0.6  
    VAL_SPLIT = 0.2    
    
    # Mixed precision training (speeds up training but may reduce precision slightly)
    USE_MIXED_PRECISION = False  # True: Use float16 (faster), False: Use float32 (more accurate)
    # Set to False for 33-bus or if you need maximum precision
    
    # --- Time-Series Configuration (Set during __init__ from Args) ---
    
    # --- Multi-Objective Optimization Weights (normalized to sum to 1.0) ---
    MOOPF_WEIGHT_LOSS = 1/3
    MOOPF_WEIGHT_VDEV = 1/3
    MOOPF_WEIGHT_CARBON = 1/3
    
    # --- Physical Constraints ---
    V_MIN = 0.90  # Minimum voltage limit
    V_MAX = 1.10  # Maximum voltage limit
    S_MAX = 1.2   # Maximum apparent power
    # S_BASE_MVA is determined dynamically based on system type in metrics.py
    # Case33: 10 MVA, Case57/118: 100 MVA
    
    # Loss weighting method: Learnable Uncertainty Weighting (Kendall et al., CVPR 2018)
    # Automatically learns optimal weights via backpropagation
    # Paper: "Multi-Task Learning Using Uncertainty to Weigh Losses"
    USE_HETEROSCEDASTIC_UNCERTAINTY = True  # True: Heteroscedastic (input-dependent), False: Homoscedastic (learnable params)
    
    # Homoscedastic: Single learnable sigma per loss term (simpler, faster)
    # Heteroscedastic: Model predicts uncertainty per sample (more expressive, requires model to output uncertainties)
    
    # Heteroscedastic loss formulation (only used if USE_HETEROSCEDASTIC_UNCERTAINTY=True)
    # Natural Parametrization: Immer et al. (NeurIPS 2023) - "Effective Bayesian Heteroscedastic Regression"
    # Paper: https://arxiv.org/abs/2306.17758, Citations: 27+ (as of Nov 2025)
    # 
    # Natural parameters: η1 = μ/σ², η2 = -1/(2σ²) < 0
    # Model outputs: [η1_var1, η1_var2, f2_var1, f2_var2] where η2 = -g+(f2)
    # g+ is exp or softplus (ensures η2 < 0, avoiding negative log variance issues)
    # 
    # Key advantages:
    # 1. Jointly concave objective (more stable optimization)
    # 2. Simpler gradients: ∇η1 = μ - y, ∇η2 = σ² - (y² - μ²) (separate residuals)
    # 3. No negative log variance issues (η2 < 0 by construction)
    # 4. Better generalization (empirical Bayes regularization)
    
    # Natural parametrization function type
    # Options: 'exp' (g+(x) = 0.5*exp(x)) or 'softplus' (g+(x) = (1/β)*log(1+exp(β*x)))
    # getattr() explanation: getattr(config, 'HETEROSCEDASTIC_NATURAL_FUNCTION', 'exp')
    #   - Gets HETEROSCEDASTIC_NATURAL_FUNCTION from config if it exists
    #   - If it doesn't exist, uses 'exp' as the default value
    #   - This allows optional config parameters with sensible defaults
    HETEROSCEDASTIC_NATURAL_FUNCTION = 'exp'  # 'exp' or 'softplus'
    HETEROSCEDASTIC_SOFTPLUS_BETA = 1.0  # Only used if HETEROSCEDASTIC_NATURAL_FUNCTION='softplus'
    
    # Clamping for numerical stability (NOT in paper)
    # Paper (Appendix E.2) does NOT use clamping - they compute natural parameters directly
    # We add clamping to prevent: division by zero (η2 → 0), overflow (η2 → -∞)
    # Set to False to match paper exactly (may have numerical issues)
    HETEROSCEDASTIC_USE_CLAMPING = False  # True: Use clamping (not in paper), False: Match paper exactly
    
    # Violation weighting (NOT in paper - our addition for physics-informed models)
    # Paper only does regression, not physics constraints
    # If True: Weight violations using predicted uncertainties (BIV-style: violation/variance)
    # If False: Use unweighted violations (match paper exactly - no physics constraints)
    HETEROSCEDASTIC_WEIGHT_VIOLATIONS = True  # True: Weight violations (our addition), False: Unweighted (match paper)
    
    # Empirical Bayes (Section 4.3 from paper) - Automatic regularization via marginal likelihood optimization
    # Paper: "Effective Bayesian Heteroscedastic Regression with Deep Neural Networks" (Immer et al., NeurIPS 2023)
    # Automatically learns layer-wise prior precisions (δl) by maximizing marginal likelihood
    # This provides automatic regularization and helps prevent overfitting
    # TEST RESULTS (Nov 2025): EB significantly improves performance. Best configs use EB.
    #   - GCN: EB_On_Conservative (burn_in=5, update_freq=5, steps=10, lr=0.001)
    #   - AdaptivePIGCN: EB_On_Aggressive (burn_in=3, update_freq=3, steps=20, lr=0.01)
    USE_EMPIRICAL_BAYES = True  # ENABLED: Test results show EB improves performance
    # Conservative settings (best for GCN):
    EB_BURN_IN_EPOCHS = 5  # Let model stabilize before starting EB optimization (reduced from 10)
    EB_UPDATE_FREQUENCY = 50  # Update hyperparameters every N epochs (reduced from 20 for more frequent updates)
    EB_HYPERPARAMETER_STEPS = 10  # Number of gradient steps for δ optimization (reduced from 20)
    EB_HYPERPARAMETER_LR = 0.001  # Lower learning rate for δ optimization (conservative)
    # Note: For AdaptivePIGCN, consider using aggressive settings (burn_in=3, update_freq=3, steps=20, lr=0.01)
    
    # HETEROSCEDASTIC_BETA removed - only BIV method is supported (no beta parameter needed)

    
    # --- Data Mode Configuration (Set during __init__ from Args) ---
    # DATA_MODE and DATA_MODE_TIMESTEPS are set dynamically in __init__()
    # Modify Args.data_mode, Args.train_timesteps, and Args.test_timesteps at the top of this file instead
    
    # --- Project Structure ---
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    EXPERIMENTAL_RESULTS_DIR = os.path.join(ROOT_DIR, 'experimental_results')
    _CURRENT_RUN_TIMESTAMP = None
    
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
            
            # Get capacity setting from Args
            if num_buses <= 33:
                capacity = Args.CAPACITY_33_BUS
            elif num_buses <= 57:
                capacity = Args.CAPACITY_57_BUS
            else:
                capacity = Args.CAPACITY_118_BUS
            
            return capacity_settings[num_buses][capacity]
        
        @staticmethod
        def get_num_gc_layers_range(num_buses):
            """
            Get GC layers range based on system size and capacity setting.
            Returns ranges with minimum width to allow optimization exploration.
            """
            capacity_settings = {
                33: {
                    'normal': (1, 4),     # Reduced: 1-4 (was 1-6)
                    'medium': (2, 5),     # Reduced: 2-5 (was 3-7)
                    'large': (3, 7)       # Reduced: 3-7 (was 5-9)
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
            
            # Get capacity setting from Args
            if num_buses <= 33:
                capacity = Args.CAPACITY_33_BUS
            elif num_buses <= 57:
                capacity = Args.CAPACITY_57_BUS
            else:
                capacity = Args.CAPACITY_118_BUS
            
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
            
            # Get capacity setting from Args
            if num_buses <= 33:
                capacity = Args.CAPACITY_33_BUS
            elif num_buses <= 57:
                capacity = Args.CAPACITY_57_BUS
            else:
                capacity = Args.CAPACITY_118_BUS
            
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
                    'num_seagulls': 1,     
                    'max_iterations': 1,   
                    'strategy': 'thorough',
                    'description': 'Extensive search for optimal hyperparameters'
                }
            elif num_buses <= 57:
                # BALANCED: Medium systems need balance between quality and time
                return {
                    'num_seagulls': 1,      
                    'max_iterations': 1,   
                    'strategy': 'balanced',
                    'description': 'Balance optimization quality vs computational time'
                }
            else:
                # QUICK: Large systems prioritize efficiency
                return {
                    'num_seagulls': 1,      # Temporarily set to 4 for quick testing
                    'max_iterations': 1,    # Temporarily set to 5 for quick testing
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
                 hours_per_day=24, sequence_length=5):
        """Initializes directories and sets up experimental run structure."""
        # Set save_results flag
        self.SAVE_RESULTS = save_results
        
        # Set data plotting flag from Args (controls both validation and profile story plots)
        self.PLOT_DATA_INFO = Args.plot_data_info
        # Backward compatibility: also set GENERATE_DATA_PROFILE_STORY (deprecated, use PLOT_DATA_INFO)
        self.GENERATE_DATA_PROFILE_STORY = Args.plot_data_info
        
        # Clear experimental results folder if requested
        if clear_results and os.path.exists(self.EXPERIMENTAL_RESULTS_DIR):
            import shutil
            try:
                print(f"\n[Clear Results] Deleting experimental_results folder...")
                shutil.rmtree(self.EXPERIMENTAL_RESULTS_DIR)
                print(f"[Clear Results] Successfully deleted: {self.EXPERIMENTAL_RESULTS_DIR}")
            except Exception as e:
                print(f"[Clear Results] Warning: Could not delete experimental_results folder: {e}")
        
        # Initialize DATA_MODE_TIMESTEPS - use Args values if not provided, fallback to defaults
        if train_timesteps is None:
            train_timesteps = Args.train_timesteps
        self.DATA_MODE_TIMESTEPS = {'train': train_timesteps, 'test': test_timesteps}
        
        # Set data mode and validate
        self.DATA_MODE = data_mode
        if data_mode not in self.DATA_MODE_TIMESTEPS:
            raise ValueError(f"Invalid data_mode '{data_mode}'. Must be 'train' or 'test'")
        
        # Set time-series configuration (always time-series mode)
        self.HOURS_PER_DAY = hours_per_day
        self.SEQUENCE_LENGTH = sequence_length
        
        print(f"Data mode: {self.DATA_MODE}, Timesteps: {self.DATA_MODE_TIMESTEPS[self.DATA_MODE]}")
        print(f"Generation mode: Time-Series")
        
        # Data directory structure: data/time_series/[train|test]
        self.DATA_DIR = os.path.join(self.ROOT_DIR, 'data', 'time_series', data_mode)
        
        print(f"\n[Data Mode] Using time_series data in {data_mode} mode")
        print(f"[Data Directory] {self.DATA_DIR}")
        
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
    
    def _initialize_run_timestamp(self):
        """Initialize the run timestamp - always create new timestamp for each run."""
        # Always create new timestamp for new run
        self._CURRENT_RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"Starting new run: run_{self._CURRENT_RUN_TIMESTAMP}")

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
