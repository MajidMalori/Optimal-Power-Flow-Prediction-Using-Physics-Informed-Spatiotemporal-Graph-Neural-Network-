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
                 dropout: float = 0.1,
                 use_twin_heads: bool = False):
        """
        Simple GCN for pure state estimation.
        
        Args:
            feature_dim: Number of input features (10 measurements)
            hidden_dim: Hidden layer dimension
            num_gc_layers: Number of graph convolution layers
            num_buses: Number of buses in the system
            dropout: Dropout rate
            use_twin_heads: If True, use separate networks for VM and VA (ETH Zurich style)
        """
        # Pure state estimation: only predict voltage [vm, va]
        output_features_per_bus = 2
        output_dim = num_buses * output_features_per_bus
        
        super().__init__(
            feature_dim=feature_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            num_gc_layers=num_gc_layers, num_buses=num_buses, dropout=dropout
        )
        
        self.use_twin_heads = use_twin_heads
        
        self.gc_layers = nn.ModuleList([
            VoltageGraphLayer(feature_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_gc_layers)
        ])
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output Layers
        if use_twin_heads:
            # ETH Zurich Twin Heads: Separate networks for magnitude and phase
            self.mag_output_layer = nn.Linear(hidden_dim, 1)  # Voltage magnitude head
            self.pha_output_layer = nn.Linear(hidden_dim, 1)  # Voltage angle head
        else:
            # Single head: Combined output [batch, buses, 2]
            self.output_layer = nn.Linear(hidden_dim, output_features_per_bus)  # Only 2 features: [vm_pu, va_rad]

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Forward pass.
        
        Args:
            x: Input measurements [batch_size, num_buses, 10]
            adj: Adjacency matrix
            
        Returns:
            If use_twin_heads=False:
                torch.Tensor: Predicted voltages [batch_size, num_buses, 2]
            If use_twin_heads=True:
                Tuple[torch.Tensor, torch.Tensor]: (x_mag, x_pha) where each is [batch, buses]
        """
        batch_size, num_nodes, _ = x.shape
        
        for gc_layer in self.gc_layers:
            x = torch.relu(gc_layer(x, adj))
            x = self.dropout_layer(x)
        
        if self.use_twin_heads:
            # ETH Zurich Twin Heads: Separate outputs for magnitude and phase
            x_mag = self.mag_output_layer(x).squeeze(-1)  # [batch, buses]
            x_pha = self.pha_output_layer(x).squeeze(-1)  # [batch, buses]
            return x_mag, x_pha
        else:
            # Single head: Combined output
            out = self.output_layer(x)  # [batch_size, num_buses, 2]
            # ROOT CAUSE DETECTION: NO CLIPPING - Let physics loss handle constraints
            return out