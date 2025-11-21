# models/adaptive_gcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base_adaptive_gcn import BaseAdaptiveGCN
from .professional_gcn_layer import ProfessionalGCNLayer

class adaptiveGCN(BaseAdaptiveGCN):
    """
    Adaptive Graph Convolutional Network (non-physics-informed).
    Uses adaptive adjacency matrix combining physical and learned graphs.
    
    FIXED: Now uses ProfessionalGCNLayer instead of flawed aggregation-then-MLP architecture.
    The GCN operation is now: A_norm @ (features @ weight), not (A @ features) @ weight.
    """
    def __init__(self, feature_dim, hidden_dim, num_gc_layers, num_buses, dropout,
                 embedding_dim: int = 16, phi: float = 0.5, config=None, normalizer=None):
        # Initialize base class with adaptive adjacency components
        super().__init__(num_buses=num_buses, embedding_dim=embedding_dim, phi=phi)
        
        self.dropout = dropout

        # Professional GCN layers (proper GCN operation with self-loops and normalization)
        # FIXED: Now uses ProfessionalGCNLayer which applies weight transformation BEFORE aggregation
        self.gc_layers = nn.ModuleList()
        for i in range(num_gc_layers):
            in_dim = feature_dim if i == 0 else hidden_dim
            # ProfessionalGCNLayer handles: A_norm @ (features @ weight) + bias, then ReLU
            self.gc_layers.append(ProfessionalGCNLayer(in_dim, hidden_dim, bias=True, activation='relu'))

        # Output: 4 features per bus [η1_var1, η1_var2, f2_var1, f2_var2]
        # Natural parameters: η1 = f1 (direct), η2 = -g+(f2) where g+ is softplus
        num_output_features = 4
        self.output_layer = nn.Linear(hidden_dim, num_output_features)
        
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, static_adj, bus_types: Optional[torch.Tensor] = None):
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, num_buses, feature_dim]
            static_adj: Static adjacency matrix [batch_size, num_buses, num_buses] (consistent shape from data loader)
            bus_types: Optional bus type codes [batch_size, num_buses] (unused, kept for compatibility)
        """
        batch_size = x.size(0)
        
        # Use base class method to compute adaptive adjacency (normalizes the combined matrix)
        A_adp_batch = self.compute_adaptive_adjacency(static_adj, batch_size, normalize=True)  # [batch_size, num_buses, num_buses]

        # Apply professional GCN layers (adaptive adjacency is normalized in compute_adaptive_adjacency)
        h = x
        for i, gc_layer in enumerate(self.gc_layers):
            # ProfessionalGCNLayer: A_norm @ (h @ weight) + bias, then ReLU
            # A_adp_batch is already normalized by compute_adaptive_adjacency
            h = gc_layer(h, A_adp_batch, is_pre_normalized=True)  # [batch_size, num_buses, hidden_dim]
            if i < len(self.gc_layers) - 1:  # Apply dropout to all but last layer
                h = self.dropout_layer(h)

        output = self.output_layer(h)  # [batch_size, num_buses, 4]
        return output