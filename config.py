import os
import torch
from datetime import datetime

class Config:
    """
    Main configuration class for the project.
    Contains global settings and nested classes for model-specific hyperparameters.
    """
    
    # =============================================================================
    # CONFIGURABLE PARAMETERS - Modify these for your experiments
    # =============================================================================
    
    # --- Device & System Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_BUSES = [33, 57, 118]
    SEED = 42
    NUM_WORKERS = 2  # Conservative default - will be auto-configured based on system capabilities
    
    # --- Training Parameters ---
    BATCH_SIZE = 64  # Default, will be overridden by adaptive function
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 5
    EARLY_STOPPING_PATIENCE = 25
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    
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
    
    # --- Loss Function Weights ---
    LAMBDA_P = 10.0  # Weight for power balance violation
    LAMBDA_V = 10.0  # Weight for voltage limit violation
    
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
    
    # =============================================================================
    # MODEL CONFIGURATION TEMPLATE
    # =============================================================================
    
    class _ModelConfig:
        """Base template for all model configurations"""
        FEATURE_DIM = 10  # Updated to include separated generation components: [vm, va, p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren]
        DROPOUT = 0.2
        HIDDEN_DIM_RANGE = (16, 128)  # Default fallback
        NUM_GC_LAYERS_RANGE = (1, 5)
        
        @staticmethod
        def get_hidden_dim_range(num_buses):
            """Scale hidden dimensions based on system size - more conservative to prevent OOM"""
            if num_buses <= 33:
                return (32, 64)      # Smaller systems - reduced max
            elif num_buses <= 57:
                return (32, 64)      # Medium systems - more conservative
            else:
                return (32, 64)      # Large systems - very conservative to prevent OOM
        
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
                    'max_iterations': 2,   
                    'strategy': 'thorough',
                    'description': 'Extensive search for optimal hyperparameters'
                }
            elif num_buses <= 57:
                # BALANCED: Medium systems need balance between quality and time
                return {
                    'num_seagulls': 1,      
                    'max_iterations': 2,   
                    'strategy': 'balanced',
                    'description': 'Balance optimization quality vs computational time'
                }
            else:
                # QUICK: Large systems prioritize efficiency
                return {
                    'num_seagulls': 1,      # Temporarily set to 4 for quick testing
                    'max_iterations': 2,    # Temporarily set to 5 for quick testing
                    'strategy': 'quick',
                    'description': 'Fast optimization for memory/time constraints'
                }

    # =============================================================================
    # MODEL-SPECIFIC CONFIGURATIONS
    # =============================================================================
    
    GCNConfig = _ModelConfig()
    
    AdaptivePIGCNConfig = _ModelConfig()
    AdaptivePIGCNConfig.EMBEDDING_DIM_RANGE = (8, 32)
    AdaptivePIGCNConfig.PHI_RANGE = (0.0, 1.0)
    
    adaptiveGCNConfig = _ModelConfig()
    adaptiveGCNConfig.EMBEDDING_DIM_RANGE = (8, 32)
    adaptiveGCNConfig.PHI_RANGE = (0.0, 1.0)

    PIGCLSTMConfig = _ModelConfig()
    PIGCLSTMConfig.RNN_LAYERS_RANGE = (1, 3)  # Reduced from (1, 5) to prevent OOM
    PIGCLSTMConfig.SEQUENCE_LENGTH_RANGE = (5, 10)  # Reduced from (5, 15) to prevent OOM
    PIGCLSTMConfig.EMBEDDING_DIM_RANGE = (8, 16)  # Reduced from (8, 32) to prevent OOM
    PIGCLSTMConfig.PHI_RANGE = (0.0, 1.0)
    
    PIGCGRUConfig = _ModelConfig()
    PIGCGRUConfig.RNN_LAYERS_RANGE = (1, 3)  # Reduced from (1, 5) to prevent OOM
    PIGCGRUConfig.SEQUENCE_LENGTH_RANGE = (5, 10)  # Reduced from (5, 15) to prevent OOM
    PIGCGRUConfig.EMBEDDING_DIM_RANGE = (8, 16)  # Reduced from (8, 32) to prevent OOM
    PIGCGRUConfig.PHI_RANGE = (0.0, 1.0)

    ResnetPIGCGRUConfig = _ModelConfig()
    ResnetPIGCGRUConfig.RNN_LAYERS_RANGE = (1, 3)  # Reduced from (1, 5) to prevent OOM
    ResnetPIGCGRUConfig.SEQUENCE_LENGTH_RANGE = (5, 10)  # Reduced from (5, 15) to prevent OOM
    ResnetPIGCGRUConfig.EMBEDDING_DIM_RANGE = (8, 16)  # Reduced from (8, 32) to prevent OOM
    ResnetPIGCGRUConfig.PHI_RANGE = (0.0, 1.0)
    
    ResnetPIGCLSTMConfig = _ModelConfig()
    ResnetPIGCLSTMConfig.RNN_LAYERS_RANGE = (1, 3)  # Reduced from (1, 5) to prevent OOM
    ResnetPIGCLSTMConfig.SEQUENCE_LENGTH_RANGE = (5, 10)  # Reduced from (5, 15) to prevent OOM
    ResnetPIGCLSTMConfig.EMBEDDING_DIM_RANGE = (8, 16)  # Reduced from (8, 32) to prevent OOM
    ResnetPIGCLSTMConfig.PHI_RANGE = (0.0, 1.0)

    # =============================================================================
    # PROPERTIES
    # =============================================================================
    
    @property
    def CURRENT_RUN_DIR(self):
        """Get the current run directory with timestamp."""
        return os.path.join(self.EXPERIMENTAL_RESULTS_DIR, f'run_{self._CURRENT_RUN_TIMESTAMP}')
    
    @property
    def LATEST_RUN_DIR(self):
        """Get the latest run directory (symlink/copy target)."""
        if self._CURRENT_RUN_TIMESTAMP:
            return os.path.join(self.EXPERIMENTAL_RESULTS_DIR, f'latest_run_{self._CURRENT_RUN_TIMESTAMP}')
        else:
            return os.path.join(self.EXPERIMENTAL_RESULTS_DIR, 'latest_run')
    
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
    
    # =============================================================================
    # INITIALIZATION
    # =============================================================================

    def __init__(self):
        """Initializes directories and sets up experimental run structure."""
        # Initialize timestamp only when actually starting a run
        self._initialize_run_timestamp()
        
        # Create base directories
        for dir_path in [self.DATA_DIR, self.EXPERIMENTAL_RESULTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Create current run directory
        os.makedirs(self.CURRENT_RUN_DIR, exist_ok=True)
        
        # Update latest run (copy current run info)
        self._update_latest_run_link()
        
        # Create run metadata
        self._create_run_metadata()
    
    # =============================================================================
    # STATIC METHODS - Configuration Helpers
    # =============================================================================
    
    @staticmethod
    def get_adaptive_batch_size(num_buses):
        """Return appropriate batch size based on system size to prevent OOM"""
        if num_buses <= 33:
            return 32  # Reduced from 64
        elif num_buses <= 57:
            return 16  # Reduced from 32
        else:
            return 8   # Reduced from 16
    
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
    
    # =============================================================================
    # INSTANCE METHODS - Path & Directory Helpers
    # =============================================================================
    
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
        return os.path.join(self.get_model_eval_dir(num_buses, model_name), "convergence.png")

    def get_training_history_path(self, num_buses: int, model_name: str) -> str:
        """Returns the training history plot path for a specific model."""
        return os.path.join(self.get_model_eval_dir(num_buses, model_name), "training_history.png")

    def get_summary_path(self, num_buses: int, model_name: str) -> str:
        """Returns the summary CSV path for a specific model."""
        return os.path.join(self.get_model_eval_dir(num_buses, model_name), "summary.csv")
    
    # =============================================================================
    # INSTANCE METHODS - Run Management (Private)
    # =============================================================================
    
    def _initialize_run_timestamp(self):
        """Initialize the run timestamp - always create new timestamp for each run."""
        # Always create new timestamp for new run
        self._CURRENT_RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"Starting new run: run_{self._CURRENT_RUN_TIMESTAMP}")

    def _update_latest_run_link(self):
        """Update the latest_run directory to point to current run."""
        import shutil
        
        # Only clean up if the experimental_results directory exists
        if os.path.exists(self.EXPERIMENTAL_RESULTS_DIR):
            # Clean up old latest_run_* directories (only directories, not files)
            try:
                for item in os.listdir(self.EXPERIMENTAL_RESULTS_DIR):
                    if item.startswith('latest_run_') and os.path.isdir(os.path.join(self.EXPERIMENTAL_RESULTS_DIR, item)):
                        old_dir = os.path.join(self.EXPERIMENTAL_RESULTS_DIR, item)
                        shutil.rmtree(old_dir)
            except (OSError, FileNotFoundError, PermissionError):
                # Directory might not exist or be accessible, that's okay
                pass
            
            # Remove generic latest_run if it exists (backward compatibility)
            generic_latest = os.path.join(self.EXPERIMENTAL_RESULTS_DIR, 'latest_run')
            if os.path.exists(generic_latest) and os.path.isdir(generic_latest):
                try:
                    shutil.rmtree(generic_latest)
                except (OSError, PermissionError):
                    # Might be in use, that's okay
                    pass
        
        # Create latest_run_info.txt with current run info
        latest_info_file = os.path.join(self.EXPERIMENTAL_RESULTS_DIR, 'latest_run_info.txt')
        try:
            with open(latest_info_file, 'w') as f:
                f.write(f"Latest run: run_{self._CURRENT_RUN_TIMESTAMP}\n")
                f.write(f"Latest run directory: latest_run_{self._CURRENT_RUN_TIMESTAMP}\n")
                f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except (OSError, PermissionError):
            # If we can't write the info file, that's okay - not critical
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
                'lambda_p': self.LAMBDA_P,
                'lambda_v': self.LAMBDA_V
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
    
    # =============================================================================
    # INSTANCE METHODS - Run Management (Public)
    # =============================================================================
    
    def finalize_run(self, run_summary: dict = None):
        """Finalize the current run by updating latest_run and logging."""
        import shutil
        import json
        import csv
        
        # Copy current run to latest_run
        if os.path.exists(self.LATEST_RUN_DIR):
            shutil.rmtree(self.LATEST_RUN_DIR)
        shutil.copytree(self.CURRENT_RUN_DIR, self.LATEST_RUN_DIR)
        
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
            
            # Also save to latest_run
            latest_metadata_file = os.path.join(self.LATEST_RUN_DIR, 'run_metadata.json')
            with open(latest_metadata_file, 'w') as f:
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
        print(f"Latest run updated: {self.LATEST_RUN_DIR}")
        print(f"Experiment logged to: {experiment_log}")
    
    def get_run_info(self):
        """Get information about the current run."""
        return {
            'run_id': f'run_{self._CURRENT_RUN_TIMESTAMP}',
            'timestamp': self._CURRENT_RUN_TIMESTAMP,
            'current_run_dir': self.CURRENT_RUN_DIR,
            'latest_run_dir': self.LATEST_RUN_DIR,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
