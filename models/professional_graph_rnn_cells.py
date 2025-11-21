"""
Professional Graph Convolutional RNN Cells.

FIXED: These cells integrate graph convolution INSIDE the RNN update rule,
eliminating the redundant double convolution that existed in the previous implementation.

Key improvements:
1. Graph convolution is applied ONCE to a combined input+hidden state
2. Uses ProfessionalGCNLayer for proper normalization and self-loops
3. More efficient and theoretically sound architecture

Based on standard Graph RNN literature (e.g., Seo et al., ICML 2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .professional_gcn_layer import ProfessionalGCNLayer


class ProfessionalGraphConvGRUCell(nn.Module):
    """
    Professional Graph Convolutional GRU Cell.
    
    FIXED: Performs graph convolution ONCE on combined input+hidden state,
    eliminating redundant double convolution.
    
    Architecture:
    1. Concatenate input x and previous hidden state h_prev
    2. Apply graph convolution to combined state (ONCE)
    3. Apply linear transformation to get gate values
    4. Perform GRU update
    
    This is more efficient and theoretically sound than the previous approach
    which convolved input and hidden state separately.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_buses: int, dropout: float = 0.0):
        """
        Args:
            input_dim: Input feature dimension per node
            hidden_dim: Hidden state dimension per node
            num_buses: Number of nodes (for graph convolution)
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_buses = num_buses
        self.dropout = dropout
        
        # Graph convolution layer for combined input+hidden state
        # Input: [batch, nodes, input_dim + hidden_dim]
        # Output: [batch, nodes, hidden_dim * 3] (for reset, update, new gates)
        combined_dim = input_dim + hidden_dim
        self.gcn_layer = ProfessionalGCNLayer(
            in_features=combined_dim,
            out_features=hidden_dim * 3,  # 3 gates: reset, update, new
            bias=True,
            activation=None  # No activation - we'll apply sigmoid/tanh separately
        )
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Professional GraphConvGRU cell.
        
        Args:
            x: Input features [batch, nodes, input_dim]
            h_prev: Previous hidden state [batch, nodes, hidden_dim]
            adj: Adjacency matrix [batch, nodes, nodes] or [nodes, nodes]
            
        Returns:
            New hidden state [batch, nodes, hidden_dim]
        """
        # Step 1: Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=-1)  # [batch, nodes, input_dim + hidden_dim]
        
        # Step 2: Apply graph convolution ONCE to the combined state
        # This is the key fix: single convolution instead of separate convolutions
        gates_conv = self.gcn_layer(combined, adj)  # [batch, nodes, hidden_dim * 3]
        
        # Step 3: Split into three gates
        r, z, n = torch.chunk(gates_conv, 3, dim=-1)  # Each: [batch, nodes, hidden_dim]
        
        # Step 4: Apply GRU update logic
        reset_gate = torch.sigmoid(r)  # r_t
        update_gate = torch.sigmoid(z)  # z_t
        new_gate = torch.tanh(n)  # h_tilde
        
        # Step 5: Compute new hidden state
        # h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
        h_new = (1 - update_gate) * h_prev + update_gate * new_gate
        
        if self.dropout > 0:
            h_new = self.dropout_layer(h_new)
        
        return h_new


class ProfessionalGraphConvLSTMCell(nn.Module):
    """
    Professional Graph Convolutional LSTM Cell.
    
    FIXED: Performs graph convolution ONCE on combined input+hidden state,
    eliminating redundant double convolution.
    
    Architecture:
    1. Concatenate input x and previous hidden state h_prev
    2. Apply graph convolution to combined state (ONCE)
    3. Apply linear transformation to get gate values
    4. Perform LSTM update
    
    This is more efficient and theoretically sound than the previous approach
    which convolved input and hidden state separately.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_buses: int, dropout: float = 0.0):
        """
        Args:
            input_dim: Input feature dimension per node
            hidden_dim: Hidden state dimension per node
            num_buses: Number of nodes (for graph convolution)
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_buses = num_buses
        self.dropout = dropout
        
        # Graph convolution layer for combined input+hidden state
        # Input: [batch, nodes, input_dim + hidden_dim]
        # Output: [batch, nodes, hidden_dim * 4] (for input, forget, output, cell gates)
        combined_dim = input_dim + hidden_dim
        self.gcn_layer = ProfessionalGCNLayer(
            in_features=combined_dim,
            out_features=hidden_dim * 4,  # 4 gates: input, forget, output, cell
            bias=True,
            activation=None  # No activation - we'll apply sigmoid/tanh separately
        )
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor,
                adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Professional GraphConvLSTM cell.
        
        Args:
            x: Input features [batch, nodes, input_dim]
            h_prev: Previous hidden state [batch, nodes, hidden_dim]
            c_prev: Previous cell state [batch, nodes, hidden_dim]
            adj: Adjacency matrix [batch, nodes, nodes] or [nodes, nodes]
            
        Returns:
            Tuple of (new_hidden_state, new_cell_state), each [batch, nodes, hidden_dim]
        """
        # Step 1: Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=-1)  # [batch, nodes, input_dim + hidden_dim]
        
        # Step 2: Apply graph convolution ONCE to the combined state
        # This is the key fix: single convolution instead of separate convolutions
        # Adjacency is pre-normalized in data loader for performance
        gates_conv = self.gcn_layer(combined, adj, is_pre_normalized=True)  # [batch, nodes, hidden_dim * 4]
        
        # Step 3: Split into four gates
        i, f, o, c_tilde = torch.chunk(gates_conv, 4, dim=-1)  # Each: [batch, nodes, hidden_dim]
        
        # Step 4: Apply LSTM update logic
        input_gate = torch.sigmoid(i)  # i_t
        forget_gate = torch.sigmoid(f)  # f_t
        output_gate = torch.sigmoid(o)  # o_t
        cell_gate = torch.tanh(c_tilde)  # c_tilde
        
        # Step 5: Compute new cell state
        # c_t = f_t * c_{t-1} + i_t * c_tilde
        c_new = forget_gate * c_prev + input_gate * cell_gate
        
        # Step 6: Compute new hidden state
        # h_t = o_t * tanh(c_t)
        h_new = output_gate * torch.tanh(c_new)
        
        if self.dropout > 0:
            h_new = self.dropout_layer(h_new)
            c_new = self.dropout_layer(c_new)
        
        return h_new, c_new

