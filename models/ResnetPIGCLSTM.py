# In models/ResnetPIGCLSTM.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class ResnetPIGCLSTM(BaseModel):
    """
    A Physics-Informed Graph Convolutional LSTM with Residual Connections.
    This version is corrected to handle an unbatched, static adjacency matrix
    and expand it internally, following the provided design guide.
    """
    def __init__(self, feature_dim: int, hidden_dim: int, num_gc_layers: int, num_buses: int, rnn_layers: int, dropout: float,
                 embedding_dim: int = 16, phi: float = 0.5, **kwargs):
        output_dim = feature_dim
        super().__init__(feature_dim=feature_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_gc_layers=num_gc_layers,
                         num_buses=num_buses, rnn_type='LSTM', rnn_layers=rnn_layers, physics_informed=True, dropout=dropout)

        if not (0.0 <= phi <= 1.0):
            raise ValueError(f"phi must be between 0 and 1, but got {phi}")

        self.phi = phi
        self.embedding_dim = embedding_dim
        self.rnn_layers = rnn_layers
        self.num_buses = num_buses
        self.hidden_dim = hidden_dim

        # Learnable node embeddings to create an adaptive graph
        self.node_embedding1 = nn.Parameter(torch.randn(num_buses, embedding_dim))
        self.node_embedding2 = nn.Parameter(torch.randn(num_buses, embedding_dim))

        # Graph Convolutional layers
        self.gc_layers = nn.ModuleList([nn.Linear(feature_dim, hidden_dim)])
        for _ in range(num_gc_layers - 1):
            self.gc_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # LSTM layers with residual connections
        lstm_io_size = hidden_dim * num_buses  # Flatten spatial features for LSTM
        self.residual_lstms = nn.ModuleList()
        self.lstm_layer_norms = nn.ModuleList()

        for _ in range(rnn_layers):
            self.residual_lstms.append(nn.LSTM(lstm_io_size, lstm_io_size, num_layers=1, batch_first=True))
            self.lstm_layer_norms.append(nn.LayerNorm(lstm_io_size))

        self.output_transform = nn.Linear(hidden_dim, feature_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the adaptive ResnetPIGCLSTM.

        Args:
            x (torch.Tensor): Input features of shape [batch_size, seq_len, num_nodes, feature_dim].
            adj (torch.Tensor): The static, dense physical adjacency matrix of shape [num_nodes, num_nodes].
                               This tensor should NOT be pre-batched.
        """
        batch_size, seq_len, num_nodes, _ = x.shape

        # --- Adaptive Adjacency Matrix Calculation ---
        # 1. Create the learned adjacency matrix [num_nodes, num_nodes].
        learned_adj = F.softmax(F.relu(torch.matmul(self.node_embedding1, self.node_embedding2.T)), dim=1)

        # 2. Combine with the physical adjacency matrix using the phi parameter.
        A_adp = self.phi * adj + (1 - self.phi) * learned_adj

        # 3. Expand the single adaptive matrix for efficient batch processing.
        A_adp_expanded = A_adp.unsqueeze(0).expand(batch_size * seq_len, num_nodes, num_nodes)

        # --- Spatio-Temporal Processing ---
        # Reshape input for GCN processing across batch and sequence dimensions.
        x_reshaped = x.view(batch_size * seq_len, num_nodes, -1)

        # GCN layers for spatial feature extraction
        h = x_reshaped
        for gc_layer in self.gc_layers:
            h_aggregated = torch.bmm(A_adp_expanded, h)
            h = F.relu(gc_layer(h_aggregated))
            h = self.dropout_layer(h)

        # Reshape for LSTM processing.
        h_lstm_in = h.view(batch_size, seq_len, -1)
        
        # Residual LSTM layers for temporal feature extraction
        h_res = h_lstm_in
        hidden_state, cell_state = None, None
        for i in range(self.rnn_layers):
            residual = h_res
            # Pass hidden and cell states from the previous layer's output
            h_res, (hidden_state, cell_state) = self.residual_lstms[i](h_res, (hidden_state, cell_state) if hidden_state is not None else None)
            h_res = h_res + residual  # Add residual connection
            h_res = self.lstm_layer_norms[i](h_res) # Apply layer normalization
            if i < self.rnn_layers - 1:
                h_res = self.dropout_layer(h_res)
        
        lstm_out = h_res
        
        # --- Output Transformation ---
        # Select the output of the last time step for prediction.
        last_step_output = lstm_out[:, -1, :]
        
        # Reshape back to a per-node representation.
        last_step_per_node = last_step_output.view(batch_size, self.num_buses, self.hidden_dim)
        
        # Apply the final linear transformation.
        final_output_per_node = self.output_transform(last_step_per_node)
        
        return final_output_per_node