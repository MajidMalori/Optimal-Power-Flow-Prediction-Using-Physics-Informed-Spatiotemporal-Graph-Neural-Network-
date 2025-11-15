"""
Graph Convolutional RNN Cells for scalable spatio-temporal modeling.

These cells process temporal sequences while maintaining graph structure,
enabling better scalability than flattening the entire graph representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GraphConvGRUCell(nn.Module):
    """
    Graph Convolutional GRU Cell.
    
    Processes temporal sequences at the node level while maintaining graph structure.
    Each node's hidden state is updated using graph convolution over its neighbors.
    
    Input: [batch, nodes, features]
    Hidden: [batch, nodes, hidden_dim]
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
        
        # Reset gate: determines how much of previous hidden state to forget
        self.reset_gate_x = nn.Linear(input_dim, hidden_dim)
        self.reset_gate_h = nn.Linear(hidden_dim, hidden_dim)
        
        # Update gate: determines how much of new information to incorporate
        self.update_gate_x = nn.Linear(input_dim, hidden_dim)
        self.update_gate_h = nn.Linear(hidden_dim, hidden_dim)
        
        # New gate: computes new candidate hidden state
        self.new_gate_x = nn.Linear(input_dim, hidden_dim)
        self.new_gate_h = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GraphConvGRU cell.
        
        Args:
            x: Input features [batch, nodes, input_dim]
            h_prev: Previous hidden state [batch, nodes, hidden_dim]
            adj: Adjacency matrix [batch, nodes, nodes] or [nodes, nodes]
            
        Returns:
            New hidden state [batch, nodes, hidden_dim]
        """
        # Ensure adj has batch dimension
        # Adjacency matrix is guaranteed to be 3D [batch_size, num_nodes, num_nodes] from data loader
        # For temporal models, it may need expansion for sequence dimension, but shape is consistent
        
        # Apply graph convolution to hidden state
        # h_aggregated = A @ h_prev: aggregate neighbor information
        h_aggregated = torch.bmm(adj, h_prev)  # [batch, nodes, hidden_dim]
        
        # Reset gate: r_t = sigmoid(W_rx @ x_t + W_rh @ (A @ h_{t-1}))
        r = torch.sigmoid(
            self.reset_gate_x(x) + self.reset_gate_h(h_aggregated)
        )
        
        # Update gate: z_t = sigmoid(W_zx @ x_t + W_zh @ (A @ h_{t-1}))
        z = torch.sigmoid(
            self.update_gate_x(x) + self.update_gate_h(h_aggregated)
        )
        
        # New gate: h_tilde = tanh(W_hx @ x_t + W_hh @ (r_t * (A @ h_{t-1})))
        # Reset gate controls how much of previous state influences new candidate
        h_tilde = torch.tanh(
            self.new_gate_x(x) + self.new_gate_h(r * h_aggregated)
        )
        
        # Final hidden state: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
        h_new = (1 - z) * h_prev + z * h_tilde
        
        if self.dropout > 0:
            h_new = self.dropout_layer(h_new)
        
        return h_new


class GraphConvLSTMCell(nn.Module):
    """
    Graph Convolutional LSTM Cell.
    
    Processes temporal sequences at the node level while maintaining graph structure.
    Each node's hidden and cell states are updated using graph convolution over neighbors.
    
    Input: [batch, nodes, features]
    Hidden: [batch, nodes, hidden_dim]
    Cell: [batch, nodes, hidden_dim]
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
        
        # Input gate: determines how much new information to store
        self.input_gate_x = nn.Linear(input_dim, hidden_dim)
        self.input_gate_h = nn.Linear(hidden_dim, hidden_dim)
        
        # Forget gate: determines how much of previous cell state to forget
        self.forget_gate_x = nn.Linear(input_dim, hidden_dim)
        self.forget_gate_h = nn.Linear(hidden_dim, hidden_dim)
        
        # Output gate: determines how much of cell state to output
        self.output_gate_x = nn.Linear(input_dim, hidden_dim)
        self.output_gate_h = nn.Linear(hidden_dim, hidden_dim)
        
        # Cell gate: computes new candidate cell state
        self.cell_gate_x = nn.Linear(input_dim, hidden_dim)
        self.cell_gate_h = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor, 
                adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of GraphConvLSTM cell.
        
        Args:
            x: Input features [batch, nodes, input_dim]
            h_prev: Previous hidden state [batch, nodes, hidden_dim]
            c_prev: Previous cell state [batch, nodes, hidden_dim]
            adj: Adjacency matrix [batch, nodes, nodes] or [nodes, nodes]
            
        Returns:
            Tuple of (new_hidden_state, new_cell_state), each [batch, nodes, hidden_dim]
        """
        # Ensure adj has batch dimension
        # Adjacency matrix is guaranteed to be 3D [batch_size, num_nodes, num_nodes] from data loader
        # For temporal models, it may need expansion for sequence dimension, but shape is consistent
        
        # Apply graph convolution to hidden state
        # h_aggregated = A @ h_prev: aggregate neighbor information
        h_aggregated = torch.bmm(adj, h_prev)  # [batch, nodes, hidden_dim]
        
        # Input gate: i_t = sigmoid(W_ix @ x_t + W_ih @ (A @ h_{t-1}))
        i = torch.sigmoid(
            self.input_gate_x(x) + self.input_gate_h(h_aggregated)
        )
        
        # Forget gate: f_t = sigmoid(W_fx @ x_t + W_fh @ (A @ h_{t-1}))
        f = torch.sigmoid(
            self.forget_gate_x(x) + self.forget_gate_h(h_aggregated)
        )
        
        # Output gate: o_t = sigmoid(W_ox @ x_t + W_oh @ (A @ h_{t-1}))
        o = torch.sigmoid(
            self.output_gate_x(x) + self.output_gate_h(h_aggregated)
        )
        
        # Cell gate: c_tilde = tanh(W_cx @ x_t + W_ch @ (A @ h_{t-1}))
        c_tilde = torch.tanh(
            self.cell_gate_x(x) + self.cell_gate_h(h_aggregated)
        )
        
        # New cell state: c_t = f_t * c_{t-1} + i_t * c_tilde
        c_new = f * c_prev + i * c_tilde
        
        # New hidden state: h_t = o_t * tanh(c_t)
        h_new = o * torch.tanh(c_new)
        
        if self.dropout > 0:
            h_new = self.dropout_layer(h_new)
            c_new = self.dropout_layer(c_new)
        
        return h_new, c_new

