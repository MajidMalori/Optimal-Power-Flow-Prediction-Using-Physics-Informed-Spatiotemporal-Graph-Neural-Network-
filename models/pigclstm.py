import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class PIGCLSTM(BaseModel):
    """
    A Physics-Informed Graph Convolutional LSTM.
    This model integrates the adaptive adjacency matrix mechanism from adaptiveGCN
    into a sequential LSTM framework to capture spatio-temporal dynamics.
    """
    def __init__(self, feature_dim: int, hidden_dim: int, num_gc_layers: int, num_buses: int, rnn_layers: int, dropout: float, 
                 embedding_dim: int = 16, phi: float = 0.5, **kwargs):
        """
        Args:
            feature_dim (int): The number of input features for each node.
            hidden_dim (int): The dimensionality of the hidden layers in both GCN and LSTM.
            num_gc_layers (int): The number of graph convolution layers to apply at each time step.
            num_buses (int): The number of nodes (buses) in the graph.
            rnn_layers (int): The number of layers in the LSTM.
            dropout (float): The dropout rate.
            embedding_dim (int): The dimensionality of the node embeddings for the adaptive matrix.
            phi (float): The interpolation coefficient for blending physical and learned graphs (0 <= phi <= 1).
        """
        # The final output will have the same number of features as the input/target.
        output_dim = feature_dim 
        super().__init__(
            feature_dim=feature_dim, hidden_dim=hidden_dim, output_dim=output_dim, 
            num_gc_layers=num_gc_layers, num_buses=num_buses, rnn_type='LSTM', 
            rnn_layers=rnn_layers, physics_informed=True, dropout=dropout
        )

        if not (0.0 <= phi <= 1.0):
            raise ValueError(f"phi must be between 0 and 1, but got {phi}")

        self.phi = phi
        self.embedding_dim = embedding_dim

        # --- Start: Components from adaptiveGCN ---
        # Learnable node embeddings to create the latent graph structure.
        self.node_embedding1 = nn.Parameter(torch.randn(num_buses, embedding_dim))
        self.node_embedding2 = nn.Parameter(torch.randn(num_buses, embedding_dim))

        # The graph convolution is now a sequence of linear layers applied after aggregation.
        # This replaces the old `StateGraphLayer`.
        self.gc_layers = nn.ModuleList()
        self.gc_layers.append(nn.Linear(feature_dim, hidden_dim))
        for _ in range(num_gc_layers - 1):
            self.gc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        # --- End: Components from adaptiveGCN ---
        
        # The size of the vector fed into the LSTM is the flattened representation of all node embeddings.
        # CRITICAL FIX: Limit LSTM size to prevent CUDA memory issues
        lstm_io_size = hidden_dim * num_buses
        
        # More aggressive memory optimization for all system sizes
        if num_buses >= 118:
            # Very large systems: use minimal LSTM size
            lstm_hidden_size = min(256, lstm_io_size // 8)
        elif num_buses >= 57:
            # Large systems: use reduced LSTM size
            lstm_hidden_size = min(512, lstm_io_size // 4)
        elif num_buses >= 33:
            # Medium systems: use moderate reduction
            lstm_hidden_size = min(1024, lstm_io_size // 2)
        else:
            # Small systems: use full size but with cap
            lstm_hidden_size = min(2048, lstm_io_size)
            
        self.lstm = nn.LSTM(
            input_size=lstm_io_size, 
            hidden_size=lstm_hidden_size, 
            num_layers=rnn_layers, 
            batch_first=True, 
            dropout=dropout if rnn_layers > 1 else 0.0
        )
        
        # Store the hidden size for later use
        self.lstm_hidden_size = lstm_hidden_size
        
        # Add projection layer for reduced LSTM output (for memory efficiency)
        if lstm_hidden_size != lstm_io_size:
            self.lstm_projection = nn.Linear(lstm_hidden_size, lstm_io_size)
            print(f"PIGCLSTM: Using reduced LSTM size {lstm_hidden_size} -> {lstm_io_size} for {num_buses}-bus system")
        else:
            self.lstm_projection = None
            print(f"PIGCLSTM: Using full LSTM size {lstm_hidden_size} for {num_buses}-bus system")
        
        # Final layer to transform the LSTM's output back to the desired feature dimension per node.
        self.output_transform = nn.Linear(hidden_dim, feature_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the adaptive PIGCLSTM.

        Args:
            x (torch.Tensor): Input features of shape [batch_size, seq_len, num_nodes, feature_dim].
            adj (torch.Tensor): The static, dense physical adjacency matrix of shape [num_nodes, num_nodes].
        """
        batch_size, seq_len, num_nodes, _ = x.shape

        # --- Adaptive Adjacency Matrix Calculation (from adaptiveGCN) ---
        # 1. Create the learned adjacency matrix. This is static and shared across all time steps.
        learned_adj = F.softmax(F.relu(torch.matmul(self.node_embedding1, self.node_embedding2.T)), dim=1)

        # 2. Combine with the physical adjacency matrix.
        # The resulting adaptive matrix is also static for this model version.
        # It is expanded to the batch*sequence dimension for efficient processing.
        A_adp = self.phi * adj + (1 - self.phi) * learned_adj
        A_adp_expanded = A_adp.unsqueeze(0).expand(batch_size * seq_len, -1, -1)

        # --- Spatio-Temporal Processing ---
        # Reshape for efficient GCN processing across the batch and sequence dimensions.
        x_reshaped = x.view(batch_size * seq_len, num_nodes, -1)
        
        # 3. Apply adaptive graph convolution layers.
        h = x_reshaped
        for gc_layer in self.gc_layers:
            # Aggregate features using the adaptive matrix, then transform.
            h_aggregated = torch.bmm(A_adp_expanded, h)
            h = F.relu(gc_layer(h_aggregated))
            h = self.dropout_layer(h)
        
        # 4. Reshape for LSTM processing.
        # The output of the GCN layers is a sequence of graph embeddings.
        h_lstm_in = h.view(batch_size, seq_len, -1)
        
        # 5. Pass the sequence through the LSTM.
        lstm_out, _ = self.lstm(h_lstm_in)
        
        # We only need the output from the final time step for state prediction.
        last_step_output = lstm_out[:, -1, :]
        
        # 6. Reshape the final time step's output to per-node features.
        # CRITICAL FIX: Handle reduced LSTM output size for memory efficiency
        if self.lstm_projection is not None:
            # For larger systems with reduced LSTM size, project back to the original size
            projected_output = self.lstm_projection(last_step_output)
            last_step_per_node = projected_output.view(batch_size, num_nodes, self.hidden_dim)
        else:
            # Original behavior for smaller systems
            last_step_per_node = last_step_output.view(batch_size, num_nodes, self.hidden_dim)
        
        # 7. Apply the final transformation to get the desired output shape.
        final_output_per_node = self.output_transform(last_step_per_node)
        
        # PHYSICAL CONSTRAINTS: Apply ReLU only to parameters that MUST be non-negative
        # Based on physics: vm_pu, p_load, p_conv, p_ren must be positive
        # Can be negative: va_rad, q_load, p_ext, q_ext, q_conv, q_ren (for reactive power control)
        if final_output_per_node.shape[-1] >= 10:  # Ensure we have 10 features
            final_output_per_node[..., 0] = torch.relu(final_output_per_node[..., 0])  # vm_pu ≥ 0 (voltage magnitude always positive)
            final_output_per_node[..., 2] = torch.relu(final_output_per_node[..., 2])  # p_load ≥ 0 (loads consume power)
            final_output_per_node[..., 6] = torch.relu(final_output_per_node[..., 6])  # p_conv ≥ 0 (generators produce power)
            final_output_per_node[..., 8] = torch.relu(final_output_per_node[..., 8])  # p_ren ≥ 0 (renewables produce power)
            # DO NOT apply ReLU to: va_rad [1], q_load [3], p_ext [4], q_ext [5], q_conv [7], q_ren [9]
            # These can be negative for physical reasons (angles, reactive power, slack balancing)
        
        # Return the structured output [batch_size, num_nodes, feature_dim].
        return final_output_per_node