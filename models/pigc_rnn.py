import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .spatiotemporal_base import SpatioTemporalBase
from .professional_graph_rnn_cells import ProfessionalGraphConvGRUCell, ProfessionalGraphConvLSTMCell
from utils.forensic_logger import get_logger

class PhysicsInformedRecurrentNet(SpatioTemporalBase):
    """
    Unified Physics-Informed Graph Convolutional RNN.
    Refactored for Full State Reconstruction.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int, num_gc_layers: int, 
                 num_buses: int, rnn_layers: int, dropout: float,
                 embedding_dim: int = 16, phi: float = 0.5, 
                 rnn_type: str = 'GRU', use_resnet: bool = False,
                 config=None, normalizer=None, **kwargs):
        
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
        
        # Create Professional RNN cells
        if rnn_type == 'GRU':
            self.rnn_cells = nn.ModuleList([
                ProfessionalGraphConvGRUCell(
                    input_dim=feature_dim if i == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    num_buses=num_buses,
                    dropout=0.0 # Handled externally
                ) for i in range(rnn_layers)
            ])
        else:  # LSTM
            self.rnn_cells = nn.ModuleList([
                ProfessionalGraphConvLSTMCell(
                    input_dim=feature_dim if i == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    num_buses=num_buses,
                    dropout=0.0
                ) for i in range(rnn_layers)
            ])
        
        # Layer normalization
        if use_resnet:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(feature_dim if i == 0 else hidden_dim) for i in range(rnn_layers)
            ])
        
        self.forensic_logger = None
        self.forward_count = 0
    
    def set_logger(self, logger):
        self.forensic_logger = logger
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor, bus_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        # FORENSIC: Log input
        self.forward_count += 1
        if self.forensic_logger and self.forensic_logger.log_interval > 0 and self.forward_count % self.forensic_logger.log_interval == 1:
            self.forensic_logger.log_model_forward(
                f"{self.__class__.__name__}_INPUT",
                {'features': x, 'adjacency': adj},
                None
            )
            
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Compute adaptive adjacency (shared across timesteps)
        # SUSPECT #2 FIX: adj is now RAW.
        A_adp = self.compute_adaptive_adjacency_for_sequence(adj, batch_size, seq_len)
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
        
        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :, :]
            A_adp_t = A_adp[:, t, :, :]
            
            h_input = x_t
            for layer_idx, rnn_cell in enumerate(self.rnn_cells):
                can_use_residual = self.use_resnet and (layer_idx > 0 or self.rnn_cells[layer_idx].input_dim == self.rnn_cells[layer_idx].hidden_dim)
                
                if can_use_residual:
                    residual = h_input
                    h_norm = self.layer_norms[layer_idx](h_input)
                else:
                    h_norm = h_input
                    residual = None
                
                if self.rnn_type == 'GRU':
                    h_processed = rnn_cell(h_norm, h_layers[layer_idx], A_adp_t)
                else:  # LSTM
                    h_processed, c_new = rnn_cell(h_norm, h_layers[layer_idx], c_layers[layer_idx], A_adp_t)
                    c_layers[layer_idx] = c_new
                
                if can_use_residual:
                    h_new = h_processed + residual
                    h_new = self.dropout_layer(h_new)
                else:
                    h_new = self.dropout_layer(h_processed)
                
                h_layers[layer_idx] = h_new
                h_input = h_new
        
        # Output transformation on last state
        last_step_per_node = h_layers[-1]
        output = self.apply_output_transformation(last_step_per_node)
          
        return output

# Define model aliases (factories) for back-compat and clean usage
class PIGCLSTM(PhysicsInformedRecurrentNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, rnn_type='LSTM', use_resnet=False, **kwargs)

class PIGCGRU(PhysicsInformedRecurrentNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, rnn_type='GRU', use_resnet=False, **kwargs)

class ResnetPIGCLSTM(PhysicsInformedRecurrentNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, rnn_type='LSTM', use_resnet=True, **kwargs)

class ResnetPIGCGRU(PhysicsInformedRecurrentNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, rnn_type='GRU', use_resnet=True, **kwargs)
