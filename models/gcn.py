import torch
import torch.nn as nn
from .base_model import BaseModel
from .layers import VoltageGraphLayer

class GCN(BaseModel):
    def __init__(self,
                 feature_dim: int = 10,
                 hidden_dim: int = 64,
                 num_gc_layers: int = 3,
                 num_buses: int = 118,
                 dropout: float = 0.1):
        """
        Simple GCN for OPF prediction.
        
        Args:
            feature_dim: Number of input features (10 measurements)
            hidden_dim: Hidden layer dimension
            num_gc_layers: Number of graph convolution layers
            num_buses: Number of buses in the system
            dropout: Dropout rate
        """
        # Pure state estimation: only predict voltage [vm, va]
        output_features_per_bus = 2
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
        # Single head: Combined output [batch, buses, 2]
        # OPF mode: outputs vary by bus type (PQ: V,θ | PV: Q,θ | Slack: P,Q)
        self.output_layer = nn.Linear(hidden_dim, output_features_per_bus)  # 2 features per bus

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Forward pass.
        
        Args:
            x: Input measurements [batch_size, num_buses, 10]
            adj: Adjacency matrix
            
        Returns:
            torch.Tensor: Predicted unknowns [batch_size, num_buses, 2]
                         OPF mode: bus-type dependent (PQ: V,θ | PV: Q,θ | Slack: P,Q)
        """
        batch_size, num_nodes, _ = x.shape
        
        for gc_layer in self.gc_layers:
            x = torch.relu(gc_layer(x, adj))
            x = self.dropout_layer(x)
        
        # Single head: Combined output
        out = self.output_layer(x)  # [batch_size, num_buses, 2]
        # ROOT CAUSE DETECTION: NO CLIPPING - Let physics loss handle constraints
        return out