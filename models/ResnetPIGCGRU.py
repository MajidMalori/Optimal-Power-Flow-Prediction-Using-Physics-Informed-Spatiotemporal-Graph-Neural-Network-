# In models/ResnetPIGCGRU.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class ResnetPIGCGRU(BaseModel):
    """
    A Physics-Informed Graph Convolutional GRU with Residual Connections.
    This version is corrected to handle an unbatched, static adjacency matrix,
    expand it internally, and properly manage hidden states across stacked GRU layers.
    """
    def __init__(self, feature_dim: int, hidden_dim: int, num_gc_layers: int, num_buses: int, rnn_layers: int, dropout: float, 
                 embedding_dim: int = 16, phi: float = 0.5, **kwargs):
        output_dim = feature_dim 
        super().__init__(feature_dim=feature_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_gc_layers=num_gc_layers, 
                         num_buses=num_buses, rnn_type='GRU', rnn_layers=rnn_layers, physics_informed=True, dropout=dropout)

        if not (0.0 <= phi <= 1.0):
            raise ValueError(f"phi must be between 0 and 1, but got {phi}")

        self.phi = phi
        self.embedding_dim = embedding_dim
        self.rnn_layers = rnn_layers
        self.num_buses = num_buses
        self.hidden_dim = hidden_dim

        # Learnable node embeddings for the adaptive graph
        self.node_embedding1 = nn.Parameter(torch.randn(num_buses, embedding_dim))
        self.node_embedding2 = nn.Parameter(torch.randn(num_buses, embedding_dim))

        # Graph Convolutional layers
        self.gc_layers = nn.ModuleList([nn.Linear(feature_dim, hidden_dim)])
        for _ in range(num_gc_layers - 1):
            self.gc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # GRU layers with residual connections and scalable sizing
        flattened_size = hidden_dim * num_buses
        # Use a reduced GRU hidden size for larger systems to prevent memory explosion
        gru_hidden_size = min(flattened_size, max(256, flattened_size // 2))
        
        self.residual_grus = nn.ModuleList()
        self.gru_layer_norms = nn.ModuleList()
        self.gru_projections = nn.ModuleList()  # Project between different sizes

        for _ in range(rnn_layers):
            self.residual_grus.append(nn.GRU(flattened_size, gru_hidden_size, num_layers=1, batch_first=True))
            self.gru_projections.append(nn.Linear(gru_hidden_size, flattened_size))  # Project back to original size
            self.gru_layer_norms.append(nn.LayerNorm(flattened_size))
        
        self.output_transform = nn.Linear(hidden_dim, feature_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the adaptive ResnetPIGCGRU.

        Args:
            x (torch.Tensor): Input features of shape [batch_size, seq_len, num_nodes, feature_dim].
            adj (torch.Tensor): The static, dense physical adjacency matrix of shape [num_nodes, num_nodes].
                               This tensor should NOT be pre-batched.
        """
        batch_size, seq_len, num_nodes, _ = x.shape

        # --- Adaptive Adjacency Matrix Calculation ---
        learned_adj = F.softmax(F.relu(torch.matmul(self.node_embedding1, self.node_embedding2.T)), dim=1)
        A_adp = self.phi * adj + (1 - self.phi) * learned_adj
        A_adp_expanded = A_adp.unsqueeze(0).expand(batch_size * seq_len, num_nodes, num_nodes)

        # --- Spatio-Temporal Processing ---
        # 1. GCN layers for spatial feature extraction
        x_reshaped = x.view(batch_size * seq_len, num_nodes, -1)
        h = x_reshaped
        for gc_layer in self.gc_layers:
            h_aggregated = torch.bmm(A_adp_expanded, h)
            h = F.relu(gc_layer(h_aggregated))
            h = self.dropout_layer(h)
        
        # 2. Reshape for GRU temporal processing
        h_gru_in = h.view(batch_size, seq_len, -1)
        
        # 3. Residual GRU layers with proper hidden state passing
        h_res = h_gru_in
        hidden_state = None  # Initialize hidden state for the first layer
        for i in range(self.rnn_layers):
            residual = h_res
            # Pass the hidden state from the previous layer to the current one.
            # The GRU returns the full output sequence and the final hidden state.
            h_res, hidden_state = self.residual_grus[i](h_res, hidden_state)
            # Project GRU output back to original size for residual connection
            h_res = self.gru_projections[i](h_res)
            h_res = h_res + residual  # Add residual connection
            h_res = self.gru_layer_norms[i](h_res) # Apply layer normalization
            if i < self.rnn_layers - 1:
                h_res = self.dropout_layer(h_res)
        
        gru_out = h_res
        
        # --- Output Transformation ---
        # Select the output from the last time step for prediction.
        last_step_output = gru_out[:, -1, :]
        last_step_per_node = last_step_output.view(batch_size, self.num_buses, self.hidden_dim)
        final_output_per_node = self.output_transform(last_step_per_node)
        
        return final_output_per_node