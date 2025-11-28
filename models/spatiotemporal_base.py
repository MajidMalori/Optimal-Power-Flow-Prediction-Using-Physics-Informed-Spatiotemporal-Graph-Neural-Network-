import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from abc import abstractmethod
from .base_model import BaseModel
from .base_adaptive_gcn import BaseAdaptiveGCN


class SpatioTemporalBase(BaseModel, BaseAdaptiveGCN):
    """
    Base class for spatio-temporal models.
    Handles common initialization and forward pass logic:
    - Adaptive adjacency matrix calculation
    - Output transformation
    
    Refactored for Full State Reconstruction (10 features).
    Removed unused legacy GC layers.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int, num_gc_layers: int, 
                 num_buses: int, rnn_layers: int, dropout: float,
                 embedding_dim: int = 16, phi: float = 0.5, 
                 config=None, normalizer=None, rnn_type: str = 'GRU', **kwargs):
        """
        Initialize common components for spatio-temporal models.
        """
        # Output: [batch, buses, 10] - Full Clean State
        output_dim = 10
        
        # Initialize nn.Module first
        nn.Module.__init__(self)
        
        # Mark that we're handling initialization manually (for BaseModel.__init__)
        self._skip_super_init = True
        
        # Initialize BaseAdaptiveGCN first (it doesn't call super, so safe)
        BaseAdaptiveGCN.__init__(self, num_buses=num_buses, embedding_dim=embedding_dim, phi=phi)
        
        # Then initialize BaseModel
        BaseModel.__init__(
            self,
            feature_dim=feature_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            num_gc_layers=num_gc_layers, num_buses=num_buses, rnn_type=rnn_type,
            rnn_layers=rnn_layers, physics_informed=True, dropout=dropout
        )
        
        # Output Layer
        output_features = 10
        self.output_transform = nn.Linear(hidden_dim, output_features)
        self.dropout_layer = nn.Dropout(dropout)
        
    
    def compute_adaptive_adjacency_for_sequence(self, adj: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Compute adaptive adjacency matrix for temporal sequences.
        """
        # Use base class method to compute adaptive adjacency for one batch
        # Adjacency is guaranteed to be 3D [batch_size, num_nodes, num_nodes] from data loader
        # Use first batch element (all are identical)
        adj_single = adj[0:1]  # [1, num_nodes, num_nodes]
        # Normalize the combined adaptive adjacency
        A_adp_batch = self.compute_adaptive_adjacency(adj_single, batch_size=1, normalize=True)  # [1, num_nodes, num_nodes]
        A_adp_2d = A_adp_batch[0]  # [num_nodes, num_nodes]
        
        # Expand for batch and sequence processing: [batch_size * seq_len, num_nodes, num_nodes]
        A_adp_expanded = A_adp_2d.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        
        return A_adp_expanded
    
    def apply_output_transformation(self, last_step_per_node: torch.Tensor) -> torch.Tensor:
        """
        Apply output transformation.
        """
        # Output: [batch, buses, 10]
        final_output = self.output_transform(last_step_per_node)
        return final_output
    
    @abstractmethod
    def forward(self, x: torch.Tensor, adj: torch.Tensor, bus_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass
