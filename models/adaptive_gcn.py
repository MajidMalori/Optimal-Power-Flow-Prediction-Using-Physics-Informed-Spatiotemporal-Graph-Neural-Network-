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

        # --- START CORRECTION ---
        # The model needs to predict 10 features for each bus (Vm, Va, Pl, Ql, P_ext, Q_ext, P_conv, Q_conv, P_ren, Q_ren).
        # The original output layer was `nn.Linear(hidden_dim, num_buses * 2)`, which
        # was incorrect. The correct output is 10 features per node.
        num_output_features = 10 
        self.output_layer = nn.Linear(hidden_dim, num_output_features)
        # --- END CORRECTION ---

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

        # --- START CORRECTION ---
        # The output layer is now applied directly to the final node embeddings 'h'.
        # The result will have the correct shape [batch_size, num_buses, num_output_features].
        output = self.output_layer(h)
        
        # PHYSICAL CONSTRAINTS: Ensure non-negative values for physically meaningful components
        # p_ext can be negative (power back to grid), but p_conv, p_ren, p_load, q_load cannot
        if output.shape[-1] >= 10:  # Ensure we have 10 features
            # Apply ReLU to voltage magnitude (index 0) to ensure non-negative
            output[..., 0] = torch.relu(output[..., 0])  # vm_pu ≥ 0
            # Apply ReLU to p_conv (index 6) and p_ren (index 8) to ensure non-negative
            output[..., 6] = torch.relu(output[..., 6])  # p_conv ≥ 0
            output[..., 8] = torch.relu(output[..., 8])  # p_ren ≥ 0
            # Apply ReLU to p_load (index 2) and q_load (index 3) to ensure non-negative
            output[..., 2] = torch.relu(output[..., 2])  # p_load ≥ 0
            output[..., 3] = torch.relu(output[..., 3])  # q_load ≥ 0
            # p_ext (index 4) and q_conv (index 7) can remain negative - no constraint
        
        # Flatten to match expected output shape [batch_size, num_buses * 10]
        return output.reshape(output.size(0), -1)
        # --- END CORRECTION ---