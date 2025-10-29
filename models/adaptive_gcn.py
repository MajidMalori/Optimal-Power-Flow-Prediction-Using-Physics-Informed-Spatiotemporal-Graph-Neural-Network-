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

        num_output_features = 10 
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

        output = self.output_layer(h)
        
        # PHYSICAL CONSTRAINTS: Apply ReLU only to parameters that MUST be non-negative
        # Based on physics: vm_pu, p_load, p_conv, p_ren must be positive
        # Can be negative: va_rad, q_load, p_ext, q_ext, q_conv, q_ren (for reactive power control)
        if output.shape[-1] >= 10:  # Ensure we have 10 features
            output[..., 0] = torch.relu(output[..., 0])  # vm_pu ≥ 0 (voltage magnitude always positive)
            output[..., 2] = torch.relu(output[..., 2])  # p_load ≥ 0 (loads consume power)
            output[..., 6] = torch.relu(output[..., 6])  # p_conv ≥ 0 (generators produce power)
            output[..., 8] = torch.relu(output[..., 8])  # p_ren ≥ 0 (renewables produce power)
            # DO NOT apply ReLU to: va_rad [1], q_load [3], p_ext [4], q_ext [5], q_conv [7], q_ren [9]
            # These can be negative for physical reasons (angles, reactive power, slack balancing)
        
        return output.reshape(output.size(0), -1)