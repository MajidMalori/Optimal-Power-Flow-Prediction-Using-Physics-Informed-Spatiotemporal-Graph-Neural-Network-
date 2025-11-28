import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .base_adaptive_gcn import BaseAdaptiveGCN
from .professional_gcn_layer import ProfessionalGCNLayer
from utils.forensic_logger import get_logger

class AdaptivePIGCN(BaseModel, BaseAdaptiveGCN):
    """
    Physics-Informed Adaptive Graph Convolutional Network.
    Refactored for Full State Reconstruction (10 features).
    Originally AdaptivePIGCN.
    """
    def __init__(self, feature_dim: int = 10, hidden_dim: int = 64,
                 num_gc_layers: int = 3, num_buses: int = 118, dropout: float = 0.1,
                 embedding_dim: int = 16, phi: float = 0.5, config=None, normalizer=None):
        """
        Args:
            feature_dim: Number of input features (10)
            hidden_dim: Hidden layer dimension
            num_gc_layers: Number of graph convolution layers
            num_buses: Number of buses
            dropout: Dropout rate (0.1 mandatory)
            embedding_dim: Embedding dimension for adaptive adjacency
            phi: Mixing coefficient
        """
        self.input_dim = feature_dim
        self.output_features_per_bus = 10
        total_output_dim = num_buses * self.output_features_per_bus
        
        nn.Module.__init__(self)
        self._skip_super_init = True
        BaseAdaptiveGCN.__init__(self, num_buses=num_buses, embedding_dim=embedding_dim, phi=phi)
        BaseModel.__init__(self, feature_dim=feature_dim, hidden_dim=hidden_dim,
                        output_dim=total_output_dim, num_gc_layers=num_gc_layers,
                        num_buses=num_buses, physics_informed=True, dropout=dropout)

        self.feature_dim = feature_dim
        
        # GCN Layers
        self.gc_layers = nn.ModuleList()
        # First layer
        self.gc_layers.append(ProfessionalGCNLayer(self.input_dim, hidden_dim, bias=True, activation='relu'))
        # Subsequent layers
        for _ in range(num_gc_layers - 1):
            self.gc_layers.append(ProfessionalGCNLayer(hidden_dim, hidden_dim, bias=True, activation='relu'))

        # Batch Norm
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_gc_layers)
        ])

        # Output Layer
        self.output_layer = nn.Linear(hidden_dim, self.output_features_per_bus)
        
        self.dropout_layer = nn.Dropout(dropout)

        self.forensic_logger = None
        self.forward_count = 0

    def set_logger(self, logger):
        self.forensic_logger = logger   

    def forward(self, x: torch.Tensor, static_adj: torch.Tensor, bus_types=None):
        """
        Forward pass.
        
        Args:
            x: Input [batch, num_buses, 10]
            static_adj: Adjacency [batch, num_buses, num_buses]
            
        Returns:
            Predicted state [batch, num_buses, 10]
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

        # 1. Adaptive Adjacency
        # SUSPECT #2 FIX: static_adj is now RAW. Mix then normalize.
        adaptive_adj = self.compute_adaptive_adjacency(static_adj, batch_size, normalize=True)

        # 2. GCN Layers
        for layer_idx, gc_layer in enumerate(self.gc_layers):
            x = gc_layer(x, adaptive_adj, is_pre_normalized=True)
            
            # BatchNorm expects [batch, features, length]
            x = x.transpose(1, 2)
            x = self.batch_norms[layer_idx](x)
            x = x.transpose(1, 2)
            
            if layer_idx < len(self.gc_layers) - 1:
                x = self.dropout_layer(x)
        
        # 3. Output
        x = self.output_layer(x)
        
        return x
