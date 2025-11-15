import torch
import torch.nn as nn
from .base_model import BaseModel
from .layers import VoltageGraphLayer
from typing import Optional

class GCN(BaseModel):
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
            config: Configuration object (for generator constraints)
            normalizer: PowerSystemNormalizer (for generator constraints)
        """
        # Output: [batch, buses, 4] = [η1_var1, η1_var2, f2_var1, f2_var2]
        # Natural parameters: η1 = f1 (direct), η2 = -g+(f2) where g+ is exp or softplus
        output_features_per_bus = 4
        output_dim = num_buses * output_features_per_bus
        
        super().__init__(
            feature_dim=feature_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            num_gc_layers=num_gc_layers, num_buses=num_buses, dropout=dropout
        )
        
        self.gc_layers = nn.ModuleList([
            VoltageGraphLayer(feature_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_gc_layers)
        ])
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output Layer
        # Output: [batch, buses, 4] = [η1_var1, η1_var2, f2_var1, f2_var2]
        # Natural parameters: η1 = f1 (direct), η2 = -g+(f2) where g+ is exp or softplus
        self.output_layer = nn.Linear(hidden_dim, output_features_per_bus)
        

    def forward(self, x: torch.Tensor, adj: torch.Tensor, bus_types: Optional[torch.Tensor] = None):
        """
        Forward pass.
        
        Args:
            x: Input measurements [batch_size, num_buses, 10]
            adj: Adjacency matrix
            bus_types: Optional bus type codes [batch_size, num_buses] with [0=PQ, 1=PV, 2=Slack] (unused, kept for compatibility)
            
        Returns:
            torch.Tensor: Predicted unknowns [batch_size, num_buses, 4]
                         OPF mode: [η1_var1, η1_var2, f2_var1, f2_var2] (natural parameters)
        """
        batch_size, num_nodes, _ = x.shape
        
        for gc_layer in self.gc_layers:
            x = torch.relu(gc_layer(x, adj))
            x = self.dropout_layer(x)
        
        # Output: [batch, buses, 4] = [η1_var1, η1_var2, f2_var1, f2_var2]
        # Natural parameters: η1 = f1 (direct), η2 = -g+(f2) where g+ is exp or softplus
        out = self.output_layer(x)  # [batch_size, num_buses, 4]
        
        return out