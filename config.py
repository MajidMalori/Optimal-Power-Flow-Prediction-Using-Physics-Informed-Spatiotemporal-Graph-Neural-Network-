import os
import torch

class Config:
    """
    Main configuration class for the project.
    Contains global settings and nested classes for model-specific hyperparameters.
    """
    
    # Base template for all model configurations
    class _ModelConfig:
        FEATURE_DIM = 6
        DROPOUT = 0.2
        
        # Adaptive scaling based on system size
        @staticmethod
        def get_hidden_dim_range(num_buses):
            """Scale hidden dimensions based on system size"""
            if num_buses <= 33:
                return (32, 96)      # Smaller systems
            elif num_buses <= 57:
                return (48, 96)      # Medium systems - more conservative
            else:
                return (64, 128)     # Large systems - conservative to prevent OOM
        
        @staticmethod
        def get_recommended_model(num_buses):
            """Return recommended model type based on system size"""
            if num_buses <= 33:
                return "PIGCGRU"  # Best performance for small systems
            elif num_buses <= 57:
                return "PIGCN"     # Most memory efficient for medium systems
            else:
                return "PIGCN"     # Only PIGCN for large systems (118-bus)
        
        @staticmethod
        def get_adaptive_mosoa_params(num_buses):
            """Return adaptive MoSOA parameters based on system size and optimization strategy"""
            if num_buses <= 33:
                # THOROUGH: Small systems can afford extensive search
                return {
                    'num_seagulls': 10,     # More agents for better exploration 
                    'max_iterations': 25,   # More iterations for convergence 
                    'strategy': 'thorough',
                    'description': 'Extensive search for optimal hyperparameters'
                }
            elif num_buses <= 57:
                # BALANCED: Medium systems need balance between quality and time
                return {
                    'num_seagulls': 6,      # Moderate number of agents 
                    'max_iterations': 15,   # Reasonable convergence time 
                    'strategy': 'balanced',
                    'description': 'Balance optimization quality vs computational time'
                }
            else:
                # QUICK: Large systems prioritize efficiency
                return {
                    'num_seagulls': 4,      # Fewer agents for speed 
                    'max_iterations': 8,    # Quick convergence 
                    'strategy': 'quick',
                    'description': 'Fast optimization for memory/time constraints'
                }
        
        # Static ranges for other parameters
        HIDDEN_DIM_RANGE = (16, 128)  # Default fallback
        NUM_GC_LAYERS_RANGE = (1, 5)

    # =============================================================================
    # Project Structure
    # =============================================================================
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    EVALUATION_DIR = os.path.join(ROOT_DIR, 'model_evaluation')

    # =============================================================================
    # Global Training & System Parameters
    # =============================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_BUSES = [33, 57, 118]
    SEED = 42
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    NUM_WORKERS = 0
    
    # Adaptive batch sizes based on system size
    @staticmethod
    def get_adaptive_batch_size(num_buses):
        """Return appropriate batch size based on system size to prevent OOM"""
        if num_buses <= 33:
            return 64
        elif num_buses <= 57:
            return 32
        else:
            return 16
    
    BATCH_SIZE = 64  # Default, will be overridden by adaptive function
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 25
    
    # Weights for the multi-objective score, normalized to sum to 1.0.
    MOOPF_WEIGHT_LOSS = 1/3
    MOOPF_WEIGHT_VDEV = 1/3
    MOOPF_WEIGHT_CARBON = 1/3

    # Physical constraints for voltage and power
    V_MIN = 0.90
    V_MAX = 1.10
    S_MAX = 1.2
    
    # System base power for per-unit calculations (matches pandapower test cases)
    S_BASE_MVA = 100.0
    
    LAMBDA_P = 10.0  # Weight for power balance violation in the loss function
    LAMBDA_V = 10.0  # Weight for voltage limit violation in the

    # =============================================================================
    # Model-Specific Hyperparameter Search Ranges
    # =============================================================================
    GCNConfig = _ModelConfig()
    
    PIGCNConfig = _ModelConfig()
    PIGCNConfig.EMBEDDING_DIM_RANGE = (8, 32)
    PIGCNConfig.PHI_RANGE = (0.0, 1.0)
    
    adaptiveGCNConfig = _ModelConfig()
    adaptiveGCNConfig.EMBEDDING_DIM_RANGE = (8, 32)
    adaptiveGCNConfig.PHI_RANGE = (0.0, 1.0)

    PIGCLSTMConfig = _ModelConfig()
    PIGCLSTMConfig.RNN_LAYERS_RANGE = (1, 5)
    PIGCLSTMConfig.SEQUENCE_LENGTH_RANGE = (5, 15)
    PIGCLSTMConfig.EMBEDDING_DIM_RANGE = (8, 32)
    PIGCLSTMConfig.PHI_RANGE = (0.0, 1.0)
    
    PIGCGRUConfig = _ModelConfig()
    PIGCGRUConfig.RNN_LAYERS_RANGE = (1, 5)
    PIGCGRUConfig.SEQUENCE_LENGTH_RANGE = (5, 15)
    PIGCGRUConfig.EMBEDDING_DIM_RANGE = (8, 32)
    PIGCGRUConfig.PHI_RANGE = (0.0, 1.0)

    # --- START CORRECTION: Add explicit configurations for the new residual models ---
    # Although they share parameters with their base versions, defining them explicitly
    # makes the configuration complete, clear, and easier to manage independently.

    ResnetPIGCGRUConfig = _ModelConfig()
    ResnetPIGCGRUConfig.RNN_LAYERS_RANGE = (1, 5)
    ResnetPIGCGRUConfig.SEQUENCE_LENGTH_RANGE = (5, 15)
    ResnetPIGCGRUConfig.EMBEDDING_DIM_RANGE = (8, 32)
    ResnetPIGCGRUConfig.PHI_RANGE = (0.0, 1.0)
    
    ResnetPIGCLSTMConfig = _ModelConfig()
    ResnetPIGCLSTMConfig.RNN_LAYERS_RANGE = (1, 5)
    ResnetPIGCLSTMConfig.SEQUENCE_LENGTH_RANGE = (5, 15)
    ResnetPIGCLSTMConfig.EMBEDDING_DIM_RANGE = (8, 32)
    ResnetPIGCLSTMConfig.PHI_RANGE = (0.0, 1.0)
    # --- END CORRECTION ---

    def __init__(self):
        """Initializes directories."""
        for dir_path in [self.DATA_DIR, self.EVALUATION_DIR]:
            os.makedirs(dir_path, exist_ok=True)

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