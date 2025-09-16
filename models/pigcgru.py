import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class PIGCGRU(BaseModel):
    """
    A Physics-Informed Graph Convolutional GRU.
    This model integrates the adaptive adjacency matrix mechanism from adaptiveGCN
    into a sequential GRU framework to capture spatio-temporal dynamics.
    """
    def __init__(self, feature_dim: int, hidden_dim: int, num_gc_layers: int, num_buses: int, rnn_layers: int, dropout: float, 
                 embedding_dim: int = 16, phi: float = 0.5, **kwargs):
        """
        Args:
            feature_dim (int): The number of input features for each node.
            hidden_dim (int): The dimensionality of the hidden layers in both GCN and GRU.
            num_gc_layers (int): The number of graph convolution layers to apply at each time step.
            num_buses (int): The number of nodes (buses) in the graph.
            rnn_layers (int): The number of layers in the GRU.
            dropout (float): The dropout rate.
            embedding_dim (int): The dimensionality of the node embeddings for the adaptive matrix.
            phi (float): The interpolation coefficient for blending physical and learned graphs (0 <= phi <= 1).
        """
        # The final output will have the same number of features as the input/target.
        output_dim = feature_dim 
        super().__init__(
            feature_dim=feature_dim, hidden_dim=hidden_dim, output_dim=output_dim, 
            num_gc_layers=num_gc_layers, num_buses=num_buses, rnn_type='GRU', 
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
        
        # The size of the vector fed into the GRU is the flattened representation of all node embeddings.
        # CRITICAL FIX: Limit GRU size to prevent CUDA memory issues
        gru_io_size = hidden_dim * num_buses
        # For larger systems, use a more memory-efficient GRU size
        if num_buses >= 57:
            # Use a reduced GRU hidden size for larger systems to prevent memory explosion
            gru_hidden_size = min(gru_io_size, max(512, gru_io_size // 4))
        else:
            gru_hidden_size = gru_io_size
            
        self.gru = nn.GRU(
            input_size=gru_io_size, 
            hidden_size=gru_hidden_size, 
            num_layers=rnn_layers, 
            batch_first=True, 
            dropout=dropout if rnn_layers > 1 else 0.0
        )
        
        # Store the hidden size for later use
        self.gru_hidden_size = gru_hidden_size
        
        # Add projection layer for reduced GRU output (for memory efficiency)
        if gru_hidden_size != gru_io_size:
            self.gru_projection = nn.Linear(gru_hidden_size, gru_io_size)
        else:
            self.gru_projection = None
        
        # Final layer to transform the GRU's output back to the desired feature dimension per node.
        self.output_transform = nn.Linear(hidden_dim, feature_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the adaptive PIGCGRU.

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
        
        # 4. Reshape for GRU processing.
        # The output of the GCN layers is a sequence of graph embeddings.
        h_gru_in = h.view(batch_size, seq_len, -1)
        
        # 5. Pass the sequence through the GRU.
        gru_out, _ = self.gru(h_gru_in)
        
        # We only need the output from the final time step for state prediction.
        last_step_output = gru_out[:, -1, :]
        
        # 6. Reshape the final time step's output to per-node features.
        # CRITICAL FIX: Handle reduced GRU output size for memory efficiency
        if self.gru_projection is not None:
            # For larger systems with reduced GRU size, project back to the original size
            projected_output = self.gru_projection(last_step_output)
            last_step_per_node = projected_output.view(batch_size, num_nodes, self.hidden_dim)
        else:
            # Original behavior for smaller systems
            last_step_per_node = last_step_output.view(batch_size, num_nodes, self.hidden_dim)
        
        # 7. Apply the final transformation to get the desired output shape.
        final_output_per_node = self.output_transform(last_step_per_node)
        
        # Return the structured output [batch_size, num_nodes, feature_dim].
        return final_output_per_node