import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base_adaptive_gcn import BaseAdaptiveGCN
from .professional_gcn_layer import ProfessionalGCNLayer
from utils.forensic_logger import get_logger

class adaptiveGCN(BaseAdaptiveGCN):
    """
    Adaptive Graph Convolutional Network (non-physics-informed).
    Refactored for Full State Reconstruction (10 features).
    Originally adaptiveGCN.
    """
    def __init__(self, feature_dim, hidden_dim, num_gc_layers, num_buses, dropout,
                 embedding_dim: int = 16, phi: float = 0.5):
        # Initialize base class with adaptive adjacency components
        super().__init__(num_buses=num_buses, embedding_dim=embedding_dim, phi=phi)
        
        self.dropout = dropout

        # Professional GCN layers
        self.gc_layers = nn.ModuleList()
        for i in range(num_gc_layers):
            in_dim = feature_dim if i == 0 else hidden_dim
            self.gc_layers.append(ProfessionalGCNLayer(in_dim, hidden_dim, bias=True, activation='relu'))

        # Output: 10 features per bus
        num_output_features = 10
        self.output_layer = nn.Linear(hidden_dim, num_output_features)
        
        self.dropout_layer = nn.Dropout(dropout)

        # Forensic logging state
        self.forensic_logger = None
        self.forward_count = 0
    
    def set_logger(self, logger):
        """Attach a forensic logger."""
        self.forensic_logger = logger

    def forward(self, x, static_adj, bus_types: Optional[torch.Tensor] = None):
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, num_buses, 10]
            static_adj: Static adjacency matrix [batch_size, num_buses, num_buses]
            
        Returns:
            torch.Tensor: Predicted full state [batch_size, num_buses, 10]
        """
        # FORENSIC: Log input
        self.forward_count += 1
        if self.forensic_logger and self.forensic_logger.log_interval > 0 and self.forward_count % self.forensic_logger.log_interval == 1:
            self.forensic_logger.log_model_forward(
                f"{self.__class__.__name__}_INPUT",
                {'features': x, 'adjacency': static_adj},
                None
            )

        batch_size = x.size(0)
        
        # Compute adaptive adjacency
        # SUSPECT #2 FIX: static_adj is now RAW (unnormalized).
        # compute_adaptive_adjacency mixes raw static + raw learned, then normalizes.
        A_adp_batch = self.compute_adaptive_adjacency(static_adj, batch_size, normalize=True)

        # Apply GCN layers
        h = x
        for i, gc_layer in enumerate(self.gc_layers):
            h = gc_layer(h, A_adp_batch, is_pre_normalized=True)
            h = self.dropout_layer(h)

        output = self.output_layer(h)
        
        return output
