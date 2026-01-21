"""
Unified Spatio-Temporal RNN model.
Consolidates PIGCGRU, PIGCLSTM, ResnetPIGCGRU, and ResnetPIGCLSTM into a single model.
"""

import torch
import torch.nn as nn
from typing import Optional
from .graph_rnn_base import GraphRNNBase
from .graph_rnn_cells import GraphConvGRUCell, GraphConvLSTMCell


class GraphRNN(GraphRNNBase):
    """
    Unified Physics-Informed Graph Convolutional RNN.
    
    Supports:
    - RNN types: 'GRU' or 'LSTM'
    - Residual connections: enabled via use_resnet=True
    - Layer normalization: enabled when use_resnet=True
    
    This model consolidates PIGCGRU, PIGCLSTM, ResnetPIGCGRU, and ResnetPIGCLSTM.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int, num_gc_layers: int, 
                 num_buses: int, rnn_layers: int, dropout: float,
                 embedding_dim: int = 16, phi: float = 0.5, 
                 rnn_type: str = 'GRU', use_resnet: bool = False,
                 config=None, normalizer=None, **kwargs):
        """
        Args:
            feature_dim: Input feature dimension per node
            hidden_dim: Hidden dimension for GCN and RNN layers
            num_gc_layers: Number of graph convolution layers
            num_buses: Number of buses/nodes
            rnn_layers: Number of RNN layers
            dropout: Dropout rate
            embedding_dim: Dimension for learned adjacency embeddings
            phi: Mixing coefficient between static and learned adjacency (0-1)
            rnn_type: Type of RNN ('GRU' or 'LSTM')
            use_resnet: If True, add residual connections and layer normalization
            config: Configuration object (unused, kept for compatibility)
            normalizer: PowerSystemNormalizer (unused, kept for compatibility)
        """
        super().__init__(
            feature_dim=feature_dim, hidden_dim=hidden_dim, num_gc_layers=num_gc_layers,
            num_buses=num_buses, rnn_layers=rnn_layers, dropout=dropout,
            embedding_dim=embedding_dim, phi=phi, config=config, normalizer=normalizer,
            rnn_type=rnn_type, **kwargs
        )
        
        if rnn_type not in ['GRU', 'LSTM']:
            raise ValueError(f"rnn_type must be 'GRU' or 'LSTM', got {rnn_type}")
        
        self.rnn_type = rnn_type
        self.use_resnet = use_resnet
        
        # Refactored RNN cell initialization to use integrated GraphConv cells.
        # This eliminates redundant double convolution by performing graph operations
        # directly within the RNN update rule, improving computational efficiency.
        if rnn_type == 'GRU':
            self.rnn_cells = nn.ModuleList([
                GraphConvGRUCell(
                    input_dim=feature_dim if i == 0 else hidden_dim,  # First layer takes raw input
                    hidden_dim=hidden_dim,
                    num_buses=num_buses,
                    dropout=0.0  # Dropout handled externally
                ) for i in range(rnn_layers)
            ])
        else:  # LSTM
            self.rnn_cells = nn.ModuleList([
                GraphConvLSTMCell(
                    input_dim=feature_dim if i == 0 else hidden_dim,  # First layer takes raw input
                    hidden_dim=hidden_dim,
                    num_buses=num_buses,
                    dropout=0.0  # Dropout handled externally
                ) for i in range(rnn_layers)
            ])
        
        # Layer normalization for ResNet variants
        # NOTE: First layer receives feature_dim input, subsequent layers receive hidden_dim
        if use_resnet:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(feature_dim if i == 0 else hidden_dim) for i in range(rnn_layers)
            ])
        
                
        # Forensic logging state
        self.forensic_logger = None
        self.forward_count = 0
    
        
    def set_logger(self, logger):
        """Attach a forensic logger."""
        self.forensic_logger = logger
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor, bus_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the unified spatio-temporal RNN.
        
        Args:
            x: Input features [batch_size, seq_len, num_nodes, feature_dim]
            adj: Static adjacency matrix [batch_size, num_nodes, num_nodes]
            bus_types: Bus type codes [batch_size, num_nodes] (optional)
            
        Returns:
            Output [batch_size, num_nodes, 4]
        """

        # FORENSIC: Log input
        self.forward_count += 1
        if self.forensic_logger and self.forward_count % self.forensic_logger.log_interval == 1:
            self.forensic_logger.log_model_forward(
                f"{self.__class__.__name__}_INPUT",
                {'features': x, 'adjacency': adj, 'bus_types': bus_types},
                None
            )
            self.forensic_logger.logger.debug(f"\n  {self.__class__.__name__} FORWARD PASS #{self.forward_count}:")
            self.forensic_logger.log_tensor_stats("Input features", x, indent=2)
            
        
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Compute adaptive adjacency matrix (shared across all timesteps)
        A_adp = self.compute_adaptive_adjacency_for_sequence(adj, batch_size, seq_len)
        # Reshape to [batch_size, seq_len, num_nodes, num_nodes] for per-timestep processing
        A_adp = A_adp.view(batch_size, seq_len, num_nodes, num_nodes)
        
        # Initialize hidden states
        if self.rnn_type == 'GRU':
            h_layers = [torch.zeros(batch_size, num_nodes, self.hidden_dim, device=x.device, dtype=x.dtype)
                       for _ in range(len(self.rnn_cells))]
            c_layers = None
        else:  # LSTM
            h_layers = [torch.zeros(batch_size, num_nodes, self.hidden_dim, device=x.device, dtype=x.dtype)
                       for _ in range(len(self.rnn_cells))]
            c_layers = [torch.zeros(batch_size, num_nodes, self.hidden_dim, device=x.device, dtype=x.dtype)
                       for _ in range(len(self.rnn_cells))]
        
        # Process sequence timestep by timestep.
        # Removed external GCN pre-processing step as graph convolution is now
        # integrated directly into the GraphConvGRUCell/GraphConvLSTMCell.
        # This optimizes the forward pass by reducing operation count.
        for t in range(seq_len):
            # Get input at timestep t (RAW input, no pre-processing)
            x_t = x[:, t, :, :]  # [batch, nodes, feature_dim]
            
            # Get adjacency for this timestep [batch, nodes, nodes]
            A_adp_t = A_adp[:, t, :, :]
            
            # Process through RNN layers.
            # Pass raw input x_t directly to the first RNN cell.
            # The cell internally handles graph convolution on the concatenated
            # input and hidden state, ensuring correct feature fusion.
            h_input = x_t  # [batch, nodes, feature_dim] for first layer, [batch, nodes, hidden_dim] for subsequent
            for layer_idx, rnn_cell in enumerate(self.rnn_cells):
                # Check if we can use residual connection (only when input_dim == output_dim)
                can_use_residual = self.use_resnet and (layer_idx > 0 or self.rnn_cells[layer_idx].input_dim == self.rnn_cells[layer_idx].hidden_dim)
                
                if can_use_residual:
                    # Implemented Pre-activation normalization (He et al., 2016).
                    # Applies LayerNorm before the operation to improve gradient flow
                    # and training stability in deep residual networks.
                    residual = h_input  # Store input for residual connection
                    h_norm = self.layer_norms[layer_idx](h_input)  # Normalize FIRST
                else:
                    h_norm = h_input
                    residual = None
                
                # Main operation: RNN cell (graph convolution happens inside)
                if self.rnn_type == 'GRU':
                    h_processed = rnn_cell(h_norm, h_layers[layer_idx], A_adp_t)
                else:  # LSTM
                    h_processed, c_new = rnn_cell(h_norm, h_layers[layer_idx], c_layers[layer_idx], A_adp_t)
                    c_layers[layer_idx] = c_new
                
                if can_use_residual:
                    # Add residual connection AFTER operation
                    h_new = h_processed + residual
                    h_new = self.dropout_layer(h_new)
                else:
                    h_new = self.dropout_layer(h_processed)
                
                h_layers[layer_idx] = h_new
                h_input = h_new  # Output of this layer is input to next
        
        # Use hidden state from last layer at final timestep
        last_step_per_node = h_layers[-1]  # [batch, nodes, hidden_dim]
      
        output = self.apply_output_transformation(last_step_per_node)
          
        # FORENSIC: Log output
        if self.forensic_logger and self.forward_count % self.forensic_logger.log_interval == 1:
            self.forensic_logger.log_tensor_stats("Final output", output, indent=2)
            output_std = output.std().item()
            if output_std < 1e-6:
                self.forensic_logger.log_diagnosis(
                    f"MODEL COLLAPSE in forward pass #{self.forward_count}: Output std = {output_std:.2e}"
                )
        
        return output

# ==============================================================================
# Factory Classes for Backward Compatibility
# ==============================================================================

class PIGCLSTM(GraphRNN):
    """Physics-Informed Graph Convolutional LSTM."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, rnn_type='LSTM', use_resnet=False, **kwargs)

class PIGCGRU(GraphRNN):
    """Physics-Informed Graph Convolutional GRU."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, rnn_type='GRU', use_resnet=False, **kwargs)

class ResnetPIGCLSTM(GraphRNN):
    """ResNet + Physics-Informed Graph Convolutional LSTM."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, rnn_type='LSTM', use_resnet=True, **kwargs)

class ResnetPIGCGRU(GraphRNN):
    """ResNet + Physics-Informed Graph Convolutional GRU."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, rnn_type='GRU', use_resnet=True, **kwargs)
