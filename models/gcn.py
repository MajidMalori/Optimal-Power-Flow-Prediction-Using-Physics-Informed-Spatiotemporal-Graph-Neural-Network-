import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .professional_gcn_layer import ProfessionalGCNLayer
from typing import Optional
from utils.forensic_logger import get_logger

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
            self.gc_layers.append(ProfessionalGCNLayer(in_dim, hidden_dim, bias=True, activation='relu'))
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Use Physics-Informed Output Layer to enforce sign conventions
        from .physics_layer import PhysicsInformedOutput
        self.output_layer = PhysicsInformedOutput(hidden_dim, output_features_per_bus)
        
        self.forensic_logger = None
        self.forward_count = 0

    def set_logger(self, logger):
        self.forensic_logger = logger

    def forward(self, x: torch.Tensor, adj: torch.Tensor, bus_types: Optional[torch.Tensor] = None):
        if adj.dim() == 2:
            batch_size = x.shape[0]
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        
        self.forward_count += 1
        if self.forensic_logger and self.forensic_logger.log_interval > 0 and self.forward_count % self.forensic_logger.log_interval == 1:
            self.forensic_logger.log_model_forward(
                "GCN_INPUT",
                {'features': x, 'adjacency': adj},
                None
            )

        # Standard GCN needs to normalize adjacency internally if not pre-normalized
        # In Suspect #2 fix, we disabled pre-normalization in loader.
        # So we need to normalize here.
        adj_norm = self._normalize_adjacency_batch(adj)

        for gc_layer in self.gc_layers:
            # Use normalized adjacency
            x = gc_layer(x, adj_norm, is_pre_normalized=True)
            x = self.dropout_layer(x)
        
        out = self.output_layer(x)
        return out
        
    def _normalize_adjacency_batch(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Normalize a batch of adjacency matrices (Symmetric normalization).
        D^-0.5 * (A + I) * D^-0.5
        """
        batch_size, num_nodes, _ = adj.shape
        device = adj.device
        dtype = adj.dtype
        
        # Add self-loops (A_hat = A + I) - vectorized
        identity = torch.eye(num_nodes, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        adj_hat = adj + identity
        
        # Compute degree matrix
        degree = torch.sum(adj_hat, dim=-1)  # [batch_size, num_nodes]
        epsilon = 1e-8
        degree = degree + epsilon
        
        # Symmetric normalization: D_hat^(-0.5) * A_hat * D_hat^(-0.5)
        degree_inv_sqrt = torch.pow(degree, -0.5)  # [batch_size, num_nodes]
        degree_inv_sqrt = torch.clamp(degree_inv_sqrt, min=0.0, max=1e10)
        degree_matrix_inv_sqrt = torch.diag_embed(degree_inv_sqrt)  # [batch_size, num_nodes, num_nodes]
        
        adj_norm = torch.bmm(torch.bmm(degree_matrix_inv_sqrt, adj_hat), degree_matrix_inv_sqrt)
        return adj_norm
