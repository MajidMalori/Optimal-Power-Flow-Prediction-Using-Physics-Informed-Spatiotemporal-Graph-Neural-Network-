import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FeatureIndices

class PhysicsInformedOutput(nn.Module):
    """
    Physics-Informed Output Layer.
    
    Applies specific activation functions to different features based on their physical constraints.
    - Strictly Positive (> 0): Softplus (smooth ReLU)
      - Active Load (P_LOAD)
      - Conventional Generation (P_CONV)
      - Renewable Generation (P_REN)
      - Voltage Magnitude (VM)
    - Unconstrained (+/-): Linear (No activation)
      - Reactive Load (Q_LOAD)
      - External Grid (P_EXT, Q_EXT)
      - Reactive Generation (Q_CONV, Q_REN)
      - Voltage Angle (VA)
      
    This ensures the model cannot predict physically impossible negative values for 
    quantities that must be positive (like load or voltage magnitude).
    """
    def __init__(self, hidden_dim: int, output_dim: int = 10):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)
        
        # Define indices for positive-only features
        # P_LOAD (0), P_CONV (4), P_REN (6), VM (8)
        self.positive_indices = [
            FeatureIndices.P_LOAD,
            FeatureIndices.P_CONV,
            FeatureIndices.P_REN,
            FeatureIndices.VM
        ]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply linear transformation first
        out = self.linear(x)
        
        # Apply Softplus to positive-only features WITHOUT in-place modification
        # We split the tensor, apply activation, and concatenate back
        
        # Strategy: Iterate through features and build a list of processed columns
        processed_features = []
        for i in range(out.shape[-1]):
            feature_col = out[..., i:i+1]  # Keep dim for concatenation
            if i in self.positive_indices:
                processed_features.append(F.softplus(feature_col))
            else:
                processed_features.append(feature_col)
                
        # Concatenate along the feature dimension
        return torch.cat(processed_features, dim=-1)
