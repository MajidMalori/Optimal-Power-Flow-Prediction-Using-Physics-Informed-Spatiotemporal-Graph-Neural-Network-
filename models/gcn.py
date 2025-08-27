# models/gcn.py

import torch
import torch.nn as nn
# FIX: Changed to relative import
from .base_model import BaseModel
from .layers import VoltageGraphLayer

class GCN(BaseModel):
    def __init__(self,
                 feature_dim: int = 6,
                 hidden_dim: int = 64,
                 num_gc_layers: int = 3,
                 num_buses: int = 118,
                 dropout: float = 0.1):
        
        super().__init__(
            feature_dim=feature_dim, hidden_dim=hidden_dim,
            num_gc_layers=num_gc_layers, num_buses=num_buses, dropout=dropout
        )
        
        self.gc_layers = nn.ModuleList([
            VoltageGraphLayer(feature_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_gc_layers)
        ])
        
        self.dropout_layer = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 6)  # All 6 features: V_mag, V_angle, P_load, Q_load, P_gen, Q_gen

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        
        for gc_layer in self.gc_layers:
            x = torch.relu(gc_layer(x, adj))
            x = self.dropout_layer(x)
        
        out = self.output_layer(x)
        return out.reshape(batch_size, -1)  # Flatten to [batch_size, num_buses * 6]