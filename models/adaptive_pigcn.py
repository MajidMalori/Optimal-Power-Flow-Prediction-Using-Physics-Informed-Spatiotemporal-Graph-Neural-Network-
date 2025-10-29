import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Placeholder classes for self-contained code ---

class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Base model initializations would go here

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

# --- Corrected and Refactored AdaptivePIGCN ---

class AdaptivePIGCN(BaseModel):
    def __init__(self, feature_dim: int = 10, hidden_dim: int = 64,
                 num_gc_layers: int = 3, num_buses: int = 118, dropout: float = 0.3,
                 embedding_dim: int = 16, phi: float = 0.5):
        """
        Initializes the Adaptive Physics-Informed Graph Convolutional Network.
        """
        output_dim = num_buses * 10  # Updated to 10 features per bus
        super().__init__(feature_dim=feature_dim, hidden_dim=hidden_dim,
                        output_dim=output_dim, num_gc_layers=num_gc_layers,
                        num_buses=num_buses, physics_informed=True)

        self.phi = phi
        self.num_buses = num_buses
        self.feature_dim = feature_dim

        # --- Adaptive Graph Learning Components ---
        self.node_embedding1 = nn.Parameter(torch.randn(num_buses, embedding_dim))
        self.node_embedding2 = nn.Parameter(torch.randn(num_buses, embedding_dim))
        
        # --- CORRECTION 1: Unify GC layers into a single ModuleList for robust layer creation ---
        self.gc_layers = nn.ModuleList()
        # First layer: feature_dim -> hidden_dim
        self.gc_layers.append(StateGraphLayer(feature_dim, hidden_dim))
        # Subsequent layers: hidden_dim -> hidden_dim
        for _ in range(num_gc_layers - 1):
            self.gc_layers.append(StateGraphLayer(hidden_dim, hidden_dim))

        # --- Batch Normalization Layers ---
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_gc_layers)
        ])

        # --- CORRECTED: Node-wise output layer instead of global MLP ---
        # This preserves the graph structure by applying the same transformation to each node
        # The output layer predicts 10 features per node: [vm_pu, va_rad, p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren]
        num_output_features = 10
        self.output_layer = nn.Linear(hidden_dim, num_output_features)
        
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, static_adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AdaptivePIGCN.

        Args:
            x (torch.Tensor): The input node features. Shape: [batch_size, num_buses, feature_dim].
            static_adj (torch.Tensor): The static (physical) adjacency matrix. Shape: [1, batch_size, num_buses, num_buses].

        Returns:
            torch.Tensor: The final model output.
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

        # --- CORRECTION 2: Use a single, clean loop for all GC layers ---
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
        
        x = self.output_layer(x)
        
        # PHYSICAL CONSTRAINTS: Apply ReLU only to parameters that MUST be non-negative
        # Based on physics: vm_pu, p_load, p_conv, p_ren must be positive
        # Can be negative: va_rad, q_load, p_ext, q_ext, q_conv, q_ren (for reactive power control)
        if x.shape[-1] >= 10:  # Ensure we have 10 features
            x[..., 0] = torch.relu(x[..., 0])  # vm_pu ≥ 0 (voltage magnitude always positive)
            x[..., 2] = torch.relu(x[..., 2])  # p_load ≥ 0 (loads consume power)
            x[..., 6] = torch.relu(x[..., 6])  # p_conv ≥ 0 (generators produce power)
            x[..., 8] = torch.relu(x[..., 8])  # p_ren ≥ 0 (renewables produce power)
            # DO NOT apply ReLU to: va_rad [1], q_load [3], p_ext [4], q_ext [5], q_conv [7], q_ren [9]
            # These can be negative for physical reasons (angles, reactive power, slack balancing)
        
        return x