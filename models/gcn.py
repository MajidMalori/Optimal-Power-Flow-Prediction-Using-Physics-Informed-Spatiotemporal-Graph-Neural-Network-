import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .professional_gcn_layer import ProfessionalGCNLayer
from typing import Optional

class GCN(BaseModel):
    """
    Graph Convolutional Network with professional GCN layers.
    
    Uses ProfessionalGCNLayer which implements:
    - Self-loops (A_hat = A + I) to preserve node features
    - Symmetric normalization (D_hat^(-0.5) * A_hat * D_hat^(-0.5)) to prevent gradient explosion
    - Proper GCN operation (A_norm @ features @ weight)
    """
    def __init__(self,
                 feature_dim: int = 10,
                 hidden_dim: int = 64,
                 num_gc_layers: int = 3,
                 num_buses: int = 118,
                 dropout: float = 0.1,
                 config=None,
                 normalizer=None):
        """
        Simple GCN for OPF prediction.
        
        Args:
            feature_dim: Number of input features (10 measurements)
            hidden_dim: Hidden layer dimension
            num_gc_layers: Number of graph convolution layers
            num_buses: Number of buses in the system
            dropout: Dropout rate
            config: Configuration object (unused, kept for compatibility)
            normalizer: PowerSystemNormalizer (unused, kept for compatibility)
        """
        # Output: [batch, buses, 4] = [η1_var1, η1_var2, f2_var1, f2_var2]
        # Natural parameters: η1 = f1 (direct), η2 = -g+(f2) where g+ is softplus
        output_features_per_bus = 4
        output_dim = num_buses * output_features_per_bus
        
        super().__init__(
            feature_dim=feature_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            num_gc_layers=num_gc_layers, num_buses=num_buses, dropout=dropout
        )
        
        # Professional GCN layers with self-loops and symmetric normalization
        self.gc_layers = nn.ModuleList()
        for i in range(num_gc_layers):
            in_dim = feature_dim if i == 0 else hidden_dim
            # Use ReLU activation for all layers except potentially the last (but we apply it manually)
            self.gc_layers.append(ProfessionalGCNLayer(in_dim, hidden_dim, bias=True, activation='relu'))
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output Layer (no activation - raw outputs for natural parametrization)
        # Output: [batch, buses, 4] = [η1_var1, η1_var2, f2_var1, f2_var2]
        # Natural parameters: η1 = f1 (direct), η2 = -g+(f2) where g+ is softplus
        self.output_layer = nn.Linear(hidden_dim, output_features_per_bus)
        

    def forward(self, x: torch.Tensor, adj: torch.Tensor, bus_types: Optional[torch.Tensor] = None):
        """
        Forward pass.
        
        Args:
            x: Input measurements [batch_size, num_buses, 10]
            adj: Adjacency matrix [batch_size, num_buses, num_buses] or [num_buses, num_buses]
            bus_types: Optional bus type codes [batch_size, num_buses] with [0=PQ, 1=PV, 2=Slack] (unused, kept for compatibility)
            
        Returns:
            torch.Tensor: Predicted unknowns [batch_size, num_buses, 4]
                         OPF mode: [η1_var1, η1_var2, f2_var1, f2_var2] (natural parameters)
        """
        # Ensure adj has batch dimension for ProfessionalGCNLayer
        if adj.dim() == 2:
            batch_size = x.shape[0]
            num_nodes = adj.shape[0]
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_nodes, num_nodes]
        
        # Apply professional GCN layers (adjacency is pre-normalized in data loader)
        for i, gc_layer in enumerate(self.gc_layers):
            x = gc_layer(x, adj, is_pre_normalized=True)  # Adjacency is pre-normalized for performance
            if i < len(self.gc_layers) - 1:  # Apply dropout to all but last layer
                x = self.dropout_layer(x)
        
        # Output: [batch, buses, 4] = [η1_var1, η1_var2, f2_var1, f2_var2]
        # Natural parameters: η1 = f1 (direct), η2 = -g+(f2) where g+ is softplus
        out = self.output_layer(x)  # [batch_size, num_buses, 4]
        
        return out