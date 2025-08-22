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
        # The model needs to predict 6 features for each bus (Vm, Va, Pl, Ql, Pg, Qg).
        # The original output layer was `nn.Linear(hidden_dim, num_buses * 2)`, which
        # was incorrect. The correct output is 6 features per node.
        num_output_features = 6 
        self.output_layer = nn.Linear(hidden_dim, num_output_features)
        # --- END CORRECTION ---

    def forward(self, x, static_adj):
        # ... (forward pass logic as corrected before) ...
        batch_size = x.size(0)
        learned_adj = F.softmax(F.relu(torch.matmul(self.node_embedding1, self.node_embedding2.T)), dim=1)
        physical_adj_batch = static_adj.unsqueeze(0).expand(batch_size, -1, -1)
        A_adp_batch = self.phi * physical_adj_batch + (1 - self.phi) * learned_adj.unsqueeze(0)

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
        # The final .view() is no longer needed as the shape is already correct.
        return output
        # --- END CORRECTION ---