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
        HIDDEN_DIM_RANGE = (16, 128)
        NUM_GC_LAYERS_RANGE = (1, 5)

    # =============================================================================
    # Project Structure
    # =============================================================================
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
    EVALUATION_DIR = os.path.join(ROOT_DIR, 'model_evaluation')
    RUNS_DIR = os.path.join(ROOT_DIR, 'runs')

    # =============================================================================
    # Global Training & System Parameters
    # =============================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_BUSES = [33, 57, 118]
    SEED = 42
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    NUM_WORKERS = 0
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 500
    EARLY_STOPPING_PATIENCE = 25
    
    # Weights for the multi-objective score, normalized to sum to 1.0.
    MOOPF_WEIGHT_LOSS = 1/3
    MOOPF_WEIGHT_VDEV = 1/3
    MOOPF_WEIGHT_CARBON = 1/3

    # Physical constraints for voltage and power
    V_MIN = 0.95
    V_MAX = 1.05
    S_MAX = 1.2

    # =============================================================================
    # Model-Specific Hyperparameter Search Ranges
    # =============================================================================
    GCNConfig = _ModelConfig()
    PIGCNConfig = _ModelConfig()
    
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
        for dir_path in [self.CHECKPOINT_DIR, self.DATA_DIR, self.EVALUATION_DIR, self.RUNS_DIR]:
            os.makedirs(dir_path, exist_ok=True)

    def get_checkpoint_path(self, filename):
        """Constructs a path in the checkpoint directory."""
        return os.path.join(self.CHECKPOINT_DIR, filename)

    def get_evaluation_path(self, filename):
        """Constructs a path in the evaluation directory."""
        return os.path.join(self.EVALUATION_DIR, filename)

    def get_runs_path(self, filename):
        """Constructs a path in the TensorBoard runs directory."""
        return os.path.join(self.RUNS_DIR, filename)