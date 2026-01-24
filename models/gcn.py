import torch
import torch.nn as nn
from .base_model import BaseModel
from .gcn_layer import GCNLayer
from typing import Optional

class GCN(BaseModel):
    """
    Standard Graph Convolutional Network.
    Refactored for Full State Reconstruction (10 features).
    Originally GCN.
    """
    def __init__(self,
                 feature_dim: int = 10,
                 hidden_dim: int = 64,
                 num_gc_layers: int = 3,
                 num_buses: int = 118,
                 dropout: float = 0.1):
        
        output_features_per_bus = 10
        output_dim = num_buses * output_features_per_bus
        
        super().__init__(
            feature_dim=feature_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            num_gc_layers=num_gc_layers, num_buses=num_buses, dropout=dropout
        )
        
        self.gc_layers = nn.ModuleList()
        for i in range(num_gc_layers):
            in_dim = feature_dim if i == 0 else hidden_dim
            self.gc_layers.append(GCNLayer(in_dim, hidden_dim, bias=True, activation='relu'))
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Use Physics-Informed Output Layer to enforce sign conventions
        from .physics_layer import PhysicsInformedOutput
        self.output_layer = PhysicsInformedOutput(hidden_dim, output_features_per_bus)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor, bus_types: Optional[torch.Tensor] = None):
        if adj.dim() == 2:
            batch_size = x.shape[0]
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Standard GCN needs to normalize adjacency internally if not pre-normalized
        # In Suspect #2 fix, we disabled pre-normalization in loader.
        # So we need to normalize here.
        adj_norm = GCNLayer.normalize_adjacency(adj)

        for gc_layer in self.gc_layers:
            # Use normalized adjacency
            x = gc_layer(x, adj_norm, is_pre_normalized=True)
            x = self.dropout_layer(x)
        
        out = self.output_layer(x)
        return out
