import torch
import torch.nn as nn
from typing import Optional
from .spatiotemporal_base import SpatioTemporalBase
from .graph_rnn_cells import GraphConvLSTMCell

class ResnetPIGCLSTM(SpatioTemporalBase):
    """
    A Physics-Informed Graph Convolutional LSTM with Residual Connections.
    Uses GraphConvLSTM cells with residual connections in the temporal processing layers.
    """
    def __init__(self, feature_dim: int, hidden_dim: int, num_gc_layers: int, num_buses: int, rnn_layers: int, dropout: float,
                 embedding_dim: int = 16, phi: float = 0.5, config=None, normalizer=None, **kwargs):
        super().__init__(
            feature_dim=feature_dim, hidden_dim=hidden_dim, num_gc_layers=num_gc_layers,
            num_buses=num_buses, rnn_layers=rnn_layers, dropout=dropout,
            embedding_dim=embedding_dim, phi=phi, config=config, normalizer=normalizer,
            rnn_type='LSTM', **kwargs
        )

        # GraphConvLSTM cells with residual connections
        self.lstm_cells = nn.ModuleList()
        self.lstm_layer_norms = nn.ModuleList()
        
        for i in range(rnn_layers):
            input_dim = hidden_dim if i == 0 else hidden_dim
            self.lstm_cells.append(
                GraphConvLSTMCell(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_buses=num_buses,
                    dropout=0.0  # Dropout handled externally
                )
            )
            self.lstm_layer_norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x: torch.Tensor, adj: torch.Tensor, bus_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the adaptive ResnetPIGCLSTM with residual connections.

        Args:
            x (torch.Tensor): Input features of shape [batch_size, seq_len, num_nodes, feature_dim].
            adj (torch.Tensor): The static, dense physical adjacency matrix of shape [num_nodes, num_nodes].
            bus_types: Bus type codes [batch_size, num_nodes] (optional)
        """
        batch_size, seq_len, num_nodes, _ = x.shape

        # Compute adaptive adjacency matrix (shared across all timesteps)
        A_adp = self.compute_adaptive_adjacency(adj, batch_size, seq_len)  # [batch_size * seq_len, num_nodes, num_nodes]
        # Reshape to [batch_size, seq_len, num_nodes, num_nodes] for per-timestep processing
        A_adp = A_adp.view(batch_size, seq_len, num_nodes, num_nodes)
        
        # Initialize hidden and cell states for each layer
        h_layers = [torch.zeros(batch_size, num_nodes, self.hidden_dim, device=x.device, dtype=x.dtype)
                    for _ in range(self.rnn_layers)]
        c_layers = [torch.zeros(batch_size, num_nodes, self.hidden_dim, device=x.device, dtype=x.dtype)
                    for _ in range(self.rnn_layers)]
        
        # Process sequence timestep by timestep
        for t in range(seq_len):
            # Get input at timestep t
            x_t = x[:, t, :, :]  # [batch, nodes, feature_dim]
            
            # Get adjacency for this timestep [batch, nodes, nodes]
            A_adp_expanded = A_adp[:, t, :, :]
            
            # Apply spatial processing (GCN layers) to get spatial features
            h_spatial = x_t
            for gc_layer in self.gc_layers:
                # Aggregate features using adaptive adjacency, then transform
                h_aggregated = torch.bmm(A_adp_expanded, h_spatial)  # [batch, nodes, features]
                h_spatial = torch.relu(gc_layer(h_aggregated))
                h_spatial = self.dropout_layer(h_spatial)
            
            # Now h_spatial is [batch, nodes, hidden_dim]
            # Process through LSTM layers with residual connections
            h_input = h_spatial
            for layer_idx, lstm_cell in enumerate(self.lstm_cells):
                residual = h_input  # Store input for residual connection
                h_new, c_new = lstm_cell(
                    h_input, h_layers[layer_idx], c_layers[layer_idx], A_adp_expanded
                )
                h_new = h_new + residual  # Add residual connection
                h_new = self.lstm_layer_norms[layer_idx](h_new)  # Apply layer normalization
                if layer_idx < self.rnn_layers - 1:
                    h_new = self.dropout_layer(h_new)
                h_layers[layer_idx] = h_new
                c_layers[layer_idx] = c_new
                h_input = h_new  # Output of this layer is input to next
        
        # Use hidden state from last layer at final timestep
        last_step_per_node = h_layers[-1]  # [batch, nodes, hidden_dim]
        
        # Apply output transformation and constraints
        return self.apply_output_transformation(last_step_per_node, bus_types)