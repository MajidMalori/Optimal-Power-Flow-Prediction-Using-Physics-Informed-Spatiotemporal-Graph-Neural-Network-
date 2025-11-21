import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from abc import abstractmethod
from .base_model import BaseModel
from .base_adaptive_gcn import BaseAdaptiveGCN


class SpatioTemporalBase(BaseModel, BaseAdaptiveGCN):
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
        
        # Initialize nn.Module first
        nn.Module.__init__(self)
        
        # Mark that we're handling initialization manually (for BaseModel.__init__)
        self._skip_super_init = True
        
        # Initialize BaseAdaptiveGCN first (it doesn't call super, so safe)
        BaseAdaptiveGCN.__init__(self, num_buses=num_buses, embedding_dim=embedding_dim, phi=phi)
        
        # Then initialize BaseModel (it won't call super due to _skip_super_init flag)
        BaseModel.__init__(
            self,
            feature_dim=feature_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            num_gc_layers=num_gc_layers, num_buses=num_buses, rnn_type=rnn_type,
            rnn_layers=rnn_layers, physics_informed=True, dropout=dropout
        )
        
        # Graph Convolutional layers (Professional GCN with self-loops and normalization)
        # NOTE: These are kept for backward compatibility but are NO LONGER USED in SpatioTemporalRNN
        # The graph convolution is now integrated inside ProfessionalGraphConvGRUCell/LSTMCell
        # This eliminates redundant double convolution
        from .professional_gcn_layer import ProfessionalGCNLayer
        self.gc_layers = nn.ModuleList()
        # First layer: feature_dim -> hidden_dim
        self.gc_layers.append(ProfessionalGCNLayer(feature_dim, hidden_dim, bias=True, activation='relu'))
        # Subsequent layers: hidden_dim -> hidden_dim
        for _ in range(num_gc_layers - 1):
            self.gc_layers.append(ProfessionalGCNLayer(hidden_dim, hidden_dim, bias=True, activation='relu'))
        
        # Output Layer
        # Output: [batch, buses, 4] = [η1_var1, η1_var2, f2_var1, f2_var2]
        # Natural parameters: η1 = f1 (direct), η2 = -g+(f2) where g+ is exp or softplus
        output_features = 4
        self.output_transform = nn.Linear(hidden_dim, output_features)
        self.dropout_layer = nn.Dropout(dropout)
        
    
    def compute_adaptive_adjacency_for_sequence(self, adj: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Compute adaptive adjacency matrix for temporal sequences.
        
        Args:
            adj: Static physical adjacency matrix [batch_size, num_nodes, num_nodes]
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Expanded adaptive adjacency matrix [batch_size * seq_len, num_nodes, num_nodes]
        """
        # Use base class method to compute adaptive adjacency for one batch
        # Adjacency is guaranteed to be 3D [batch_size, num_nodes, num_nodes] from data loader
        # Use first batch element (all are identical)
        adj_single = adj[0:1]  # [1, num_nodes, num_nodes]
        # Normalize the combined adaptive adjacency
        A_adp_batch = self.compute_adaptive_adjacency(adj_single, batch_size=1, normalize=True)  # [1, num_nodes, num_nodes]
        A_adp_2d = A_adp_batch[0]  # [num_nodes, num_nodes]
        
        # Expand for batch and sequence processing: [batch_size * seq_len, num_nodes, num_nodes]
        A_adp_expanded = A_adp_2d.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        
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
        for i, gc_layer in enumerate(self.gc_layers):
            # ProfessionalGCNLayer: A_norm @ (h @ weight) + bias, then ReLU
            # A_adp_expanded is already normalized by compute_adaptive_adjacency_for_sequence
            h = gc_layer(h, A_adp_expanded, is_pre_normalized=True)  # [batch_size * seq_len, num_nodes, hidden_dim]
            if i < len(self.gc_layers) - 1:  # Apply dropout to all but last layer
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

