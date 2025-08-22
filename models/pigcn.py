import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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
    def __init__(self, feature_dim: int = 6, hidden_dim: int = 64,
                 num_gc_layers: int = 3, num_buses: int = 118, dropout: float = 0.3,
                 embedding_dim: int = 16, phi: float = 0.5):
        """
        Initializes the Adaptive Physics-Informed Graph Convolutional Network.
        """
        output_dim = num_buses * 6
        super().__init__(feature_dim=feature_dim, hidden_dim=hidden_dim,
                        output_dim=output_dim, num_gc_layers=num_gc_layers,
                        num_buses=num_buses, physics_informed=True)

        self.phi = phi

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

        # --- MLP layers for final prediction ---
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * num_buses, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
        
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, static_adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AdaptivePIGCN.

        Args:
            x (torch.Tensor): The input node features. Shape: [batch_size, num_buses, feature_dim].
            static_adj (torch.Tensor): The static (physical) adjacency matrix. Shape: [num_buses, num_buses].

        Returns:
            torch.Tensor: The final model output.
        """
        batch_size = x.size(0)

        # --- 1. Construct the Adaptive Adjacency Matrix ---
        # Learned (data-driven) adjacency matrix
        learned_adj = F.softmax(F.relu(torch.matmul(self.node_embedding1, self.node_embedding2.T)), dim=1)
        
        # This part correctly handles the `batch1 must be a 3D tensor` error
        # by ensuring the adjacency matrix is 3D.
        static_adj_batch = static_adj.unsqueeze(0).expand(batch_size, -1, -1)
        learned_adj_batch = learned_adj.unsqueeze(0).expand(batch_size, -1, -1)
        
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
            
            # CORRECTION 3: Apply dropout to all but the last layer
            if i < len(self.gc_layers) - 1:
                x = self.dropout_layer(x)
        
        # --- 3. Reshape and pass through MLP for final output ---
        x = x.reshape(batch_size, -1)
        x = self.mlp(x)
        
        return x