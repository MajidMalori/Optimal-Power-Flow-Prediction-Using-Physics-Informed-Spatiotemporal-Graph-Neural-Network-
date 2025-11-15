# models/adaptive_gcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class adaptiveGCN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_gc_layers, num_buses, dropout,
                 embedding_dim: int = 16, phi: float = 0.5, config=None, normalizer=None):
        super(adaptiveGCN, self).__init__()

        self.num_buses = num_buses
        self.dropout = dropout
        self.phi = phi
        self.embedding_dim = embedding_dim
        self.node_embedding1 = nn.Parameter(torch.randn(num_buses, embedding_dim))
        self.node_embedding2 = nn.Parameter(torch.randn(num_buses, embedding_dim))

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(feature_dim, hidden_dim))
        for _ in range(num_gc_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output: 4 features per bus [η1_var1, η1_var2, f2_var1, f2_var2]
        # Natural parameters: η1 = f1 (direct), η2 = -g+(f2) where g+ is exp or softplus
        num_output_features = 4
        self.output_layer = nn.Linear(hidden_dim, num_output_features)
        

    def forward(self, x, static_adj, bus_types: Optional[torch.Tensor] = None):
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, num_buses, feature_dim]
            static_adj: Static adjacency matrix [batch_size, num_buses, num_buses] (consistent shape from data loader)
            bus_types: Optional bus type codes [batch_size, num_buses] (unused, kept for compatibility)
        """
        batch_size = x.size(0)
        learned_adj = F.softmax(F.relu(torch.matmul(self.node_embedding1, self.node_embedding2.T)), dim=1)
        
        # Adjacency matrix is guaranteed to be 3D [batch_size, num_buses, num_buses] from data loader
        physical_adj_batch = static_adj
        
        # Create learned adjacency batch: [batch_size, num_buses, num_buses]
        learned_adj_batch = learned_adj.unsqueeze(0).repeat(batch_size, 1, 1)
        
        A_adp_batch = self.phi * physical_adj_batch + (1 - self.phi) * learned_adj_batch

        h = x
        for layer in self.layers:
            h_aggregated = torch.bmm(A_adp_batch, h)
            h = layer(h_aggregated)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        output = self.output_layer(h)  # [batch_size, num_buses, 4]
        return output