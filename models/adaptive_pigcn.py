import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class StateGraphLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(StateGraphLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x shape: [batch_size, num_nodes, in_features]
        # adj shape: [batch_size, num_nodes, num_nodes]
        support = torch.bmm(adj, x) # Message passing
        output = self.linear(support) # Transformation
        return output

class AdaptivePIGCN(BaseModel):
    def __init__(self, feature_dim: int = 10, hidden_dim: int = 64,
                 num_gc_layers: int = 3, num_buses: int = 118, dropout: float = 0.3,
                 embedding_dim: int = 16, phi: float = 0.5, use_heteroscedastic: bool = False):
        """
        Initializes the Adaptive Physics-Informed Graph Convolutional Network.
        
        OPF Mode:
        - Input: feature_dim (10) = [p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren, vm_meas, va_meas]
        - Output (homoscedastic): [batch, buses, 2] = bus-type dependent unknowns (PQ: V,θ | PV: Q,θ | Slack: P,Q)
        - Output (heteroscedastic): [batch, buses, 4] = [η1_var1, η1_var2, f2_var1, f2_var2]
          Natural parameters: η1 = f1 (direct), η2 = -g+(f2) where g+ is exp or softplus
        
        Args:
            feature_dim: Number of input features per node (measurements)
            hidden_dim: Hidden layer dimension
            num_gc_layers: Number of graph convolution layers
            num_buses: Number of buses in the power system
            dropout: Dropout rate
            embedding_dim: Dimension for learned adjacency embeddings
            phi: Mixing coefficient between static and learned adjacency (0-1)
            use_heteroscedastic: If True, output 4 features (predictions + uncertainties)
        """
        # Input dimension (measurements)
        self.input_dim = feature_dim  # 10 features
        
        # Output dimension (OPF unknowns) - per bus features
        self.use_heteroscedastic = use_heteroscedastic
        self.output_features_per_bus = 4 if use_heteroscedastic else 2  # 4 for heteroscedastic, 2 for homoscedastic
        
        # Total output dimension for BaseModel (flattened)
        total_output_dim = num_buses * self.output_features_per_bus
        
        super().__init__(feature_dim=feature_dim, hidden_dim=hidden_dim,
                        output_dim=total_output_dim, num_gc_layers=num_gc_layers,
                        num_buses=num_buses, physics_informed=True)

        self.phi = phi
        self.num_buses = num_buses
        self.feature_dim = feature_dim  # Legacy compatibility

        # --- Adaptive Graph Learning Components ---
        self.node_embedding1 = nn.Parameter(torch.randn(num_buses, embedding_dim))
        self.node_embedding2 = nn.Parameter(torch.randn(num_buses, embedding_dim))
        
        # --- Graph Convolution Layers (shared between both heads if twin) ---
        self.gc_layers = nn.ModuleList()
        assert self.input_dim == feature_dim, f"Input dim mismatch: {self.input_dim} != {feature_dim}"
        self.gc_layers.append(StateGraphLayer(self.input_dim, hidden_dim))
        # Subsequent layers: hidden_dim -> hidden_dim
        for _ in range(num_gc_layers - 1):
            self.gc_layers.append(StateGraphLayer(hidden_dim, hidden_dim))

        # --- Batch Normalization Layers ---
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_gc_layers)
        ])

        # --- Output Layer ---
        # Homoscedastic: [batch, buses, 2] = predictions only
        # Heteroscedastic: [batch, buses, 4] = [η1_var1, η1_var2, f2_var1, f2_var2]
        #   Natural parameters: η1 = f1 (direct), η2 = -g+(f2) where g+ is exp or softplus
        self.output_layer = nn.Linear(hidden_dim, self.output_features_per_bus)
        
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, static_adj: torch.Tensor):
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
        # Learned (data-driven) adjacency matrix
        learned_adj = F.softmax(F.relu(torch.matmul(self.node_embedding1, self.node_embedding2.T)), dim=1)
        
        # Handle adjacency matrix - ensure it's 3D: [batch_size, num_buses, num_buses]
        if static_adj.dim() == 4:
            static_adj_batch = static_adj.squeeze(0)  # Remove the first dimension
        elif static_adj.dim() == 3:
            static_adj_batch = static_adj  # Already correct shape
        else:
            static_adj_batch = static_adj.unsqueeze(0).expand(batch_size, -1, -1)
            
        # Create learned adjacency batch: [batch_size, num_buses, num_buses]
        # learned_adj is [num_buses, num_buses], expand to [batch_size, num_buses, num_buses]
        learned_adj_batch = learned_adj.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Combine the static and learned matrices
        adaptive_adj = self.phi * static_adj_batch + (1 - self.phi) * learned_adj_batch

        # --- 2. Graph Convolution Layers (shared feature extraction) ---
        for i, gc_layer in enumerate(self.gc_layers):
            x = gc_layer(x, adaptive_adj)
            
            # The output of StateGraphLayer is [batch, nodes, features].
            # BatchNorm1d expects [batch, features, length (nodes)].
            x = x.transpose(1, 2)
            x = self.batch_norms[i](x)
            x = x.transpose(1, 2)
            
            x = F.relu(x)
            
            if i < len(self.gc_layers) - 1:
                x = self.dropout_layer(x)
        
        # --- 3. Output Layer ---
        # Homoscedastic: [batch, buses, 2] = predictions only
        # Heteroscedastic: [batch, buses, 4] = [η1_var1, η1_var2, f2_var1, f2_var2]
        #   Natural parameters: η1 = f1 (direct), η2 = -g+(f2) where g+ is exp or softplus
        x = self.output_layer(x)  # [batch_size, num_buses, output_features_per_bus]
        
        # NOTE: Model outputs raw predictions. No constraints applied in architecture.
        
        return x