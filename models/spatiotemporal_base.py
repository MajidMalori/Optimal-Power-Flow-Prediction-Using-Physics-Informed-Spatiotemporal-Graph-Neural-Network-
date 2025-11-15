import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from abc import abstractmethod
from .base_model import BaseModel


class SpatioTemporalBase(BaseModel):
    """
    Base class for spatio-temporal models (PIGCGRU, PIGCLSTM, ResnetPIGCGRU, ResnetPIGCLSTM).
    Handles common initialization and forward pass logic:
    - Adaptive adjacency matrix calculation
    - Graph convolution layers
    - Output transformation
    - Generator constraints
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int, num_gc_layers: int, 
                 num_buses: int, rnn_layers: int, dropout: float,
                 embedding_dim: int = 16, phi: float = 0.5, 
                 config=None, normalizer=None, rnn_type: str = 'GRU', **kwargs):
        """
        Initialize common components for spatio-temporal models.
        
        Args:
            feature_dim: Input feature dimension per node
            hidden_dim: Hidden dimension for GCN layers
            num_gc_layers: Number of graph convolution layers
            num_buses: Number of buses/nodes
            rnn_layers: Number of RNN layers (for temporal processing)
            dropout: Dropout rate
            embedding_dim: Dimension for learned adjacency embeddings
            phi: Mixing coefficient between static and learned adjacency (0-1)
            config: Configuration object (for generator constraints)
            normalizer: PowerSystemNormalizer (for generator constraints)
            rnn_type: Type of RNN ('GRU' or 'LSTM')
        """
        # Output: [batch, buses, 4] = [η1_var1, η1_var2, f2_var1, f2_var2]
        output_dim = 4
        super().__init__(
            feature_dim=feature_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            num_gc_layers=num_gc_layers, num_buses=num_buses, rnn_type=rnn_type,
            rnn_layers=rnn_layers, physics_informed=True, dropout=dropout
        )
        
        # Validate phi
        if not (0.0 <= phi <= 1.0):
            raise ValueError(f"phi must be between 0 and 1, but got {phi}")
        
        self.phi = phi
        self.embedding_dim = embedding_dim
        
        # Learnable node embeddings for adaptive adjacency matrix
        self.node_embedding1 = nn.Parameter(torch.randn(num_buses, embedding_dim))
        self.node_embedding2 = nn.Parameter(torch.randn(num_buses, embedding_dim))
        
        # Graph Convolutional layers
        self.gc_layers = nn.ModuleList()
        self.gc_layers.append(nn.Linear(feature_dim, hidden_dim))
        for _ in range(num_gc_layers - 1):
            self.gc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output Layer
        # Output: [batch, buses, 4] = [η1_var1, η1_var2, f2_var1, f2_var2]
        # Natural parameters: η1 = f1 (direct), η2 = -g+(f2) where g+ is exp or softplus
        output_features = 4
        self.output_transform = nn.Linear(hidden_dim, output_features)
        self.dropout_layer = nn.Dropout(dropout)
        
    
    def compute_adaptive_adjacency(self, adj: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Compute adaptive adjacency matrix combining physical and learned graphs.
        
        Args:
            adj: Static physical adjacency matrix [num_nodes, num_nodes] or [batch_size, num_nodes, num_nodes]
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Expanded adaptive adjacency matrix [batch_size * seq_len, num_nodes, num_nodes]
        """
        # Create learned adjacency matrix
        learned_adj = F.softmax(F.relu(torch.matmul(self.node_embedding1, self.node_embedding2.T)), dim=1)
        
        # Adjacency matrix is guaranteed to be 3D [batch_size, num_nodes, num_nodes] from data loader
        # Use first batch element for combining with learned adj (all batch elements are identical)
        adj_2d = adj[0]  # Extract 2D matrix [num_nodes, num_nodes]
        
        # Combine with physical adjacency matrix
        A_adp = self.phi * adj_2d + (1 - self.phi) * learned_adj
        
        # Expand for batch and sequence processing: [batch_size * seq_len, num_nodes, num_nodes]
        A_adp_expanded = A_adp.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        
        return A_adp_expanded
    
    def apply_spatial_processing(self, x: torch.Tensor, A_adp_expanded: torch.Tensor) -> torch.Tensor:
        """
        Apply graph convolution layers for spatial feature extraction.
        
        Args:
            x: Input features [batch_size * seq_len, num_nodes, feature_dim]
            A_adp_expanded: Adaptive adjacency matrix [batch_size * seq_len, num_nodes, num_nodes]
            
        Returns:
            Spatial features [batch_size * seq_len, num_nodes, hidden_dim]
        """
        h = x
        for gc_layer in self.gc_layers:
            # Aggregate features using adaptive adjacency, then transform
            h_aggregated = torch.bmm(A_adp_expanded, h)
            h = F.relu(gc_layer(h_aggregated))
            h = self.dropout_layer(h)
        
        return h
    
    def apply_output_transformation(self, last_step_per_node: torch.Tensor, 
                                   bus_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply output transformation.
        
        Args:
            last_step_per_node: Features from last time step [batch_size, num_nodes, hidden_dim]
            bus_types: Bus type codes [batch_size, num_nodes] (unused, kept for compatibility)
            
        Returns:
            Final output [batch_size, num_nodes, 4]
        """
        # Output: [batch, buses, 4] = [η1_var1, η1_var2, f2_var1, f2_var2]
        # Natural parameters: η1 = f1 (direct), η2 = -g+(f2) where g+ is softplus
        final_output = self.output_transform(last_step_per_node)
        return final_output
    
    @abstractmethod
    def forward(self, x: torch.Tensor, adj: torch.Tensor, bus_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass. Must be implemented by subclasses to handle temporal processing.
        
        Args:
            x: Input features [batch_size, seq_len, num_nodes, feature_dim]
            adj: Static adjacency matrix [num_nodes, num_nodes]
            bus_types: Bus type codes [batch_size, num_nodes] (optional)
            
        Returns:
            Output [batch_size, num_nodes, 4]
        """
        pass

