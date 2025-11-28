"""
Base class for adaptive GCN models.
Contains shared logic for adaptive adjacency matrix computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BaseAdaptiveGCN(nn.Module):
    """
    Base class for adaptive graph convolutional networks.
    Contains shared logic for:
    - Node embeddings for learned adjacency
    - Adaptive adjacency matrix computation (combining physical and learned graphs)
    """
    
    def __init__(self, num_buses: int, embedding_dim: int = 16, phi: float = 0.5):
        """
        Initialize base adaptive GCN components.
        
        Args:
            num_buses: Number of buses/nodes in the graph
            embedding_dim: Dimension for learned adjacency embeddings
            phi: Mixing coefficient between static and learned adjacency (0-1)
        """
        # Only call super() if not in multiple inheritance scenario
        # In multiple inheritance, SpatioTemporalBase will handle nn.Module initialization
        if not hasattr(self, '_skip_super_init'):
            super().__init__()
        
        if not (0.0 <= phi <= 1.0):
            raise ValueError(f"phi must be between 0 and 1, but got {phi}")
        
        self.num_buses = num_buses
        self.embedding_dim = embedding_dim
        self.phi = phi
        
        # Learnable node embeddings for adaptive adjacency matrix
        self.node_embedding1 = nn.Parameter(torch.randn(num_buses, embedding_dim))
        self.node_embedding2 = nn.Parameter(torch.randn(num_buses, embedding_dim))
    
    def compute_adaptive_adjacency(self, static_adj: torch.Tensor, batch_size: int, normalize: bool = True) -> torch.Tensor:
        """
        Compute adaptive adjacency matrix combining physical and learned graphs.
        
        NOTE: static_adj is RAW (from data loader) if data_loader disabled pre-normalization.
        The combined (static + learned) adjacency needs normalization before use in GCN layers.
        
        Args:
            static_adj: Static physical adjacency matrix [batch_size, num_buses, num_buses]
            batch_size: Batch size
            normalize: Whether to normalize the combined adaptive adjacency (default: True)
            
        Returns:
            Adaptive adjacency matrix [batch_size, num_buses, num_buses]
            If normalize=True, this is normalized and ready for GCN layers.
        """
        # Create learned adjacency matrix from node embeddings
        learned_adj = F.softmax(F.relu(torch.matmul(self.node_embedding1, self.node_embedding2.T)), dim=1)
        
        # Adjacency matrix is guaranteed to be 3D [batch_size, num_buses, num_buses] from data loader
        physical_adj_batch = static_adj
        
        # Create learned adjacency batch: [batch_size, num_buses, num_buses]
        learned_adj_batch = learned_adj.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Combine static and learned adjacency matrices
        # If static_adj is raw, mixing it with learned (softmax) is safe.
        # If static_adj was pre-normalized, mixing might be odd, but we disabled that.
        A_adp_batch = self.phi * physical_adj_batch + (1 - self.phi) * learned_adj_batch
        
        # Normalize the combined adaptive adjacency (required for GCN layers)
        if normalize:
            A_adp_batch = self._normalize_adjacency_batch(A_adp_batch)
        
        return A_adp_batch
    
    def _normalize_adjacency_batch(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Normalize a batch of adjacency matrices (for adaptive adjacency).
        
        Args:
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        
        Returns:
            Normalized adjacency matrix [batch_size, num_nodes, num_nodes]
        """
        batch_size, num_nodes, _ = adj.shape
        device = adj.device
        dtype = adj.dtype
        
        # Add self-loops (A_hat = A + I)
        identity = torch.eye(num_nodes, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        adj_hat = adj + identity
        
        # Compute degree matrix
        degree = torch.sum(adj_hat, dim=-1)  # [batch_size, num_nodes]
        epsilon = 1e-8
        degree = degree + epsilon
        
        # Symmetric normalization: D_hat^(-0.5) * A_hat * D_hat^(-0.5)
        degree_inv_sqrt = torch.pow(degree, -0.5)  # [batch_size, num_nodes]
        degree_inv_sqrt = torch.clamp(degree_inv_sqrt, min=0.0, max=1e10)
        degree_matrix_inv_sqrt = torch.diag_embed(degree_inv_sqrt)  # [batch_size, num_nodes, num_nodes]
        
        adj_norm = torch.bmm(torch.bmm(degree_matrix_inv_sqrt, adj_hat), degree_matrix_inv_sqrt)
        return adj_norm

