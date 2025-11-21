import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .base_adaptive_gcn import BaseAdaptiveGCN
from .professional_gcn_layer import ProfessionalGCNLayer

class AdaptivePIGCN(BaseModel, BaseAdaptiveGCN):
    def __init__(self, feature_dim: int = 10, hidden_dim: int = 64,
                 num_gc_layers: int = 3, num_buses: int = 118, dropout: float = 0.3,
                 embedding_dim: int = 16, phi: float = 0.5, config=None, normalizer=None):
        """
        Initializes the Adaptive Physics-Informed Graph Convolutional Network.
        
        OPF Mode:
        - Input: feature_dim (10) = [p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren, vm_meas, va_meas]
        - Output: [batch, buses, 4] = [η1_var1, η1_var2, f2_var1, f2_var2]
          Natural parameters: η1 = f1 (direct), η2 = -g+(f2) where g+ is exp or softplus
        
        Args:
            feature_dim: Number of input features per node (measurements)
            hidden_dim: Hidden layer dimension
            num_gc_layers: Number of graph convolution layers
            num_buses: Number of buses in the power system
            dropout: Dropout rate
            embedding_dim: Dimension for learned adjacency embeddings
            phi: Mixing coefficient between static and learned adjacency (0-1)
            config: Configuration object (for generator constraints)
            normalizer: PowerSystemNormalizer (for generator constraints)
        """
        # Input dimension (measurements)
        self.input_dim = feature_dim  # 10 features
        
        # Output dimension (OPF unknowns) - per bus features
        # Always use heteroscedastic: 4 features per bus
        self.output_features_per_bus = 4
        
        # Total output dimension for BaseModel (flattened)
        total_output_dim = num_buses * self.output_features_per_bus
        
        # Initialize nn.Module first
        nn.Module.__init__(self)
        
        # Mark that we're handling initialization manually (for BaseModel.__init__)
        self._skip_super_init = True
        
        # Initialize BaseAdaptiveGCN first (it won't call super due to _skip_super_init flag)
        BaseAdaptiveGCN.__init__(self, num_buses=num_buses, embedding_dim=embedding_dim, phi=phi)
        
        # Then initialize BaseModel (it won't call super due to _skip_super_init flag)
        BaseModel.__init__(self, feature_dim=feature_dim, hidden_dim=hidden_dim,
                        output_dim=total_output_dim, num_gc_layers=num_gc_layers,
                        num_buses=num_buses, physics_informed=True)

        self.feature_dim = feature_dim  # Legacy compatibility
        
        # --- Graph Convolution Layers (Professional GCN with self-loops and normalization) ---
        self.gc_layers = nn.ModuleList()
        assert self.input_dim == feature_dim, f"Input dim mismatch: {self.input_dim} != {feature_dim}"
        # First layer: feature_dim -> hidden_dim
        self.gc_layers.append(ProfessionalGCNLayer(self.input_dim, hidden_dim, bias=True, activation='relu'))
        # Subsequent layers: hidden_dim -> hidden_dim
        for _ in range(num_gc_layers - 1):
            self.gc_layers.append(ProfessionalGCNLayer(hidden_dim, hidden_dim, bias=True, activation='relu'))

        # --- Batch Normalization Layers ---
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_gc_layers)
        ])

        # --- Output Layer ---
        # Output: [batch, buses, 4] = [η1_var1, η1_var2, f2_var1, f2_var2]
        # Natural parameters: η1 = f1 (direct), η2 = -g+(f2) where g+ is exp or softplus
        self.output_layer = nn.Linear(hidden_dim, self.output_features_per_bus)
        
        self.dropout_layer = nn.Dropout(dropout)
        

    def forward(self, x: torch.Tensor, static_adj: torch.Tensor, bus_types=None):
        """
        Forward pass for the AdaptivePIGCN.

        Args:
            x (torch.Tensor): Input measurements. Shape: [batch_size, num_buses, 10]
                             Features: [p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren, vm_partial, va_partial]
            static_adj (torch.Tensor): Static (physical) adjacency matrix. 
                                      Shape: [1, batch_size, num_buses, num_buses] or [batch_size, num_buses, num_buses]

        Returns:
            torch.Tensor: Predicted unknowns. Shape: [batch_size, num_buses, 2]
                         OPF mode: bus-type dependent (PQ: V,θ | PV: Q,θ | Slack: P,Q)
        """
        batch_size = x.size(0)

        # --- 1. Construct the Adaptive Adjacency Matrix ---
        # Use base class method to compute adaptive adjacency (normalizes the combined matrix)
        adaptive_adj = self.compute_adaptive_adjacency(static_adj, batch_size, normalize=True)

        # --- 2. Graph Convolution Layers (Professional GCN with self-loops and normalization) ---
        for i, gc_layer in enumerate(self.gc_layers):
            # ProfessionalGCNLayer: A_norm @ (x @ weight) + bias, then ReLU
            # adaptive_adj is already normalized by compute_adaptive_adjacency
            x = gc_layer(x, adaptive_adj, is_pre_normalized=True)  # [batch_size, num_buses, hidden_dim]
            
            # BatchNorm1d expects [batch, features, length (nodes)]
            x = x.transpose(1, 2)  # [batch_size, hidden_dim, num_buses]
            x = self.batch_norms[i](x)
            x = x.transpose(1, 2)  # [batch_size, num_buses, hidden_dim]
            
            # ReLU is already applied in ProfessionalGCNLayer, but we keep it here for safety
            # (ProfessionalGCNLayer applies ReLU internally, so this is redundant but harmless)
            
            if i < len(self.gc_layers) - 1:
                x = self.dropout_layer(x)
        
        # --- 3. Output Layer ---
        # Output: [batch, buses, 4] = [η1_var1, η1_var2, f2_var1, f2_var2]
        # Natural parameters: η1 = f1 (direct), η2 = -g+(f2) where g+ is exp or softplus
        x = self.output_layer(x)  # [batch_size, num_buses, 4]
        return x