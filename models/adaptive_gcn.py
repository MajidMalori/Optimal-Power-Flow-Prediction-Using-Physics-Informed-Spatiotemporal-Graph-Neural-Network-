import torch
import torch.nn as nn
from typing import Optional
from .base_model import BaseModel
from .adaptive_topology_learner import AdaptiveTopologyLearner
from .gcn_layer import GCNLayer
from .physics_layer import PhysicsInformedOutput

class AdaptiveGCN(BaseModel, AdaptiveTopologyLearner):
    """
    Adaptive Graph Convolutional Network.
    Supports both physics-informed and standard modes via configuration.
    Refactored for Full State Reconstruction (10 features).
    """
    def __init__(self, feature_dim: int = 10, hidden_dim: int = 64,
                 num_gc_layers: int = 3, num_buses: int = 118, dropout: float = 0.1,
                 embedding_dim: int = 16, phi: float = 0.5, 
                 physics_informed: bool = True, use_batch_norm: bool = True,
                 config=None, normalizer=None):
        """
        Args:
            feature_dim: Number of input features (10)
            hidden_dim: Hidden layer dimension
            num_gc_layers: Number of graph convolution layers
            num_buses: Number of buses
            dropout: Dropout rate
            embedding_dim: Embedding dimension for adaptive adjacency
            phi: Mixing coefficient
            physics_informed: Whether to use physics-informed loss/constraints (passed to BaseModel)
            use_batch_norm: Whether to use Batch Normalization
        """
        self.output_features_per_bus = 10
        total_output_dim = num_buses * self.output_features_per_bus
        
        # Initialize nn.Module first
        nn.Module.__init__(self)
        
        # Mark that we're handling initialization manually (for AdaptiveTopologyLearner)
        self._skip_super_init = True
        
        # Initialize AdaptiveTopologyLearner first (it doesn't call super, so safe)
        AdaptiveTopologyLearner.__init__(self, num_buses=num_buses, embedding_dim=embedding_dim, phi=phi)
        
        # Then initialize BaseModel
        BaseModel.__init__(self, feature_dim=feature_dim, hidden_dim=hidden_dim,
                        output_dim=total_output_dim, num_gc_layers=num_gc_layers,
                        num_buses=num_buses, physics_informed=physics_informed, dropout=dropout)

        self.use_batch_norm = use_batch_norm
        
        # GCN Layers
        self.gc_layers = nn.ModuleList()
        # First layer
        self.gc_layers.append(GCNLayer(feature_dim, hidden_dim, bias=True, activation='relu'))
        # Subsequent layers
        for _ in range(num_gc_layers - 1):
            self.gc_layers.append(GCNLayer(hidden_dim, hidden_dim, bias=True, activation='relu'))

        # Batch Norm (Optional)
        if self.use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_gc_layers)
            ])
        else:
            self.batch_norms = None

        # Output Layer - Always use PhysicsInformedOutput to enforce constraints (non-negative magnitude)
        self.output_layer = PhysicsInformedOutput(hidden_dim, self.output_features_per_bus)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, static_adj: torch.Tensor, bus_types: Optional[torch.Tensor] = None):
        """
        Forward pass.
        
        Args:
            x: Input [batch, num_buses, 10]
            static_adj: Adjacency [batch, num_buses, num_buses]
            
        Returns:
            Predicted state [batch, num_buses, 10]
        """
        
        batch_size = x.size(0)

        # 1. Adaptive Adjacency
        # static_adj is RAW. Mix then normalize.
        adaptive_adj = self.compute_adaptive_adjacency(static_adj, batch_size, normalize=True)

        # 2. GCN Layers
        h = x
        for layer_idx, gc_layer in enumerate(self.gc_layers):
            # Pass is_pre_normalized=True because we normalized adaptive_adj above
            h = gc_layer(h, adaptive_adj, is_pre_normalized=True)
            
            if self.use_batch_norm and self.batch_norms is not None:
                # BatchNorm expects [batch, features, length]
                # Our h is [batch, num_buses, hidden_dim]
                h = h.transpose(1, 2)
                h = self.batch_norms[layer_idx](h)
                h = h.transpose(1, 2)
            
            if layer_idx < len(self.gc_layers): # Apply dropout between layers (and after last layer? logic check)
                 # Original logic applied dropout after every layer.
                 # Let's keep it consistent.
                 h = self.dropout_layer(h)
        
        # 3. Output
        output = self.output_layer(h)
        
        return output

