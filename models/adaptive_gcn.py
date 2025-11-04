# models/adaptive_gcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class adaptiveGCN(nn.Module):
    # ... (init function as before) ...
    def __init__(self, feature_dim, hidden_dim, num_gc_layers, num_buses, dropout,
                 embedding_dim: int = 16, phi: float = 0.5):
        super(adaptiveGCN, self).__init__()

        # ... (parameter initializations) ...
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

        # OPF Mode: Output 2 features per bus (unknowns vary by bus type)
        # PQ buses: V, θ | PV buses: Q, θ | Slack buses: P, Q
        num_output_features = 2
        self.output_layer = nn.Linear(hidden_dim, num_output_features)

    def forward(self, x, static_adj):
        # ... (forward pass logic as corrected before) ...
        batch_size = x.size(0)
        learned_adj = F.softmax(F.relu(torch.matmul(self.node_embedding1, self.node_embedding2.T)), dim=1)
        
        # Handle adjacency matrix - ensure it's 3D: [batch_size, num_buses, num_buses]
        if static_adj.dim() == 4:
            physical_adj_batch = static_adj.squeeze(0)  # Remove the first dimension
        elif static_adj.dim() == 3:
            physical_adj_batch = static_adj  # Already correct shape
        else:
            physical_adj_batch = static_adj.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Create learned adjacency batch: [batch_size, num_buses, num_buses]
        learned_adj_batch = learned_adj.unsqueeze(0).repeat(batch_size, 1, 1)
        
        A_adp_batch = self.phi * physical_adj_batch + (1 - self.phi) * learned_adj_batch

        h = x
        for layer in self.layers:
            h_aggregated = torch.bmm(A_adp_batch, h)
            h = layer(h_aggregated)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        output = self.output_layer(h)  # [batch_size, num_buses, 2]
        
        # ROOT CAUSE DETECTION: NO CLIPPING - Let physics loss handle constraints
        # OPF mode: Output shape [batch_size, num_buses, 2] = bus-type dependent unknowns
        # - PQ buses: V (≥0), θ (any) | PV buses: Q (any), θ (any) | Slack buses: P (any), Q (any)
        # - If model predicts invalid values, physics loss will penalize it
        # - This exposes the root cause instead of hiding it with ReLU
        
        return output  # [batch_size, num_buses, 2]