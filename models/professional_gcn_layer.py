"""
Professional Graph Convolutional Network Layer

This module implements a mathematically sound GCN layer following best practices:
1. Self-loops: A_hat = A + I (preserves node features)
2. Symmetric normalization: D_hat^(-0.5) * A_hat * D_hat^(-0.5) (prevents gradient explosion)
3. Proper GCN operation: output = A_norm @ features @ weight

Based on Kipf & Welling (ICLR 2017): "Semi-Supervised Classification with Graph Convolutional Networks"
Paper: https://arxiv.org/abs/1609.02907

This fixes the fundamental architectural flaws identified in the original implementation:
- Flaw #1: Missing self-loops (nodes forgetting their own features)
- Flaw #2: Missing normalization (gradient explosion from degree imbalance)
- Flaw #3: Incorrect operation order (aggregation then MLP, not proper GCN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProfessionalGCNLayer(nn.Module):
    """
    Professional Graph Convolutional Network Layer.
    
    Implements the standard GCN operation with self-loops and symmetric normalization:
    H^(l+1) = σ(D_hat^(-0.5) * A_hat * D_hat^(-0.5) * H^(l) * W^(l))
    
    Where:
    - A_hat = A + I (adjacency matrix with self-loops)
    - D_hat = degree matrix of A_hat
    - H^(l) = node features at layer l
    - W^(l) = learnable weight matrix
    - σ = activation function (ReLU)
    
    Args:
        in_features: Number of input features per node
        out_features: Number of output features per node
        bias: Whether to include bias term (default: True)
        activation: Activation function (default: ReLU, can be None)
    
    Input:
        x: Node features [batch_size, num_nodes, in_features]
        adj: Adjacency matrix [batch_size, num_nodes, num_nodes] or [num_nodes, num_nodes]
    
    Output:
        output: Transformed node features [batch_size, num_nodes, out_features]
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, activation: str = 'relu'):
        super(ProfessionalGCNLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation
        
        # Learnable weight matrix: W ∈ R^(in_features × out_features)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        # Bias term (optional)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights using Kaiming initialization (good for ReLU)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights and bias."""
        # Kaiming initialization for ReLU activations
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        
        if self.bias is not None:
            # Initialize bias to small values
            nn.init.zeros_(self.bias)
    
    def normalize_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        """
        DEPRECATED: Normalization is now pre-computed during data loading for performance.
        
        This method is kept for backward compatibility with adaptive models that need
        to normalize combined (static + learned) adjacency matrices.
        
        For static models, the adjacency is already pre-normalized in the data loader.
        For adaptive models, this is used to normalize the combined adaptive adjacency.
        
        Args:
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes] or [num_nodes, num_nodes]
        
        Returns:
            Normalized adjacency matrix with self-loops [batch_size, num_nodes, num_nodes]
        """
        # Handle both batched and unbatched adjacency matrices
        if adj.dim() == 2:
            # Unbatched: [num_nodes, num_nodes] -> [1, num_nodes, num_nodes]
            adj = adj.unsqueeze(0)
            was_unbatched = True
        else:
            was_unbatched = False
        
        batch_size, num_nodes, _ = adj.shape
        device = adj.device
        dtype = adj.dtype
        
        # Step 1: Add self-loops (A_hat = A + I) - vectorized
        # Create identity matrix for each batch (efficient expand)
        identity = torch.eye(num_nodes, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        adj_hat = adj + identity
        
        # Step 2: Compute degree matrix D_hat
        # D_hat[i,i] = sum of row i (or column i, since A_hat is symmetric)
        degree = torch.sum(adj_hat, dim=-1)  # [batch_size, num_nodes] - degree of each node
        
        # Handle zero-degree nodes (isolated nodes) to avoid division by zero
        # Add small epsilon to prevent NaN
        epsilon = 1e-8
        degree = degree + epsilon
        
        # Step 3: Symmetric normalization: D_hat^(-0.5) * A_hat * D_hat^(-0.5)
        # Compute D_hat^(-0.5) as a diagonal matrix
        degree_inv_sqrt = torch.pow(degree, -0.5)  # [batch_size, num_nodes]
        degree_inv_sqrt = torch.clamp(degree_inv_sqrt, min=0.0, max=1e10)  # Prevent extreme values
        
        # Create diagonal matrices: D_hat^(-0.5) for each batch
        degree_matrix_inv_sqrt = torch.diag_embed(degree_inv_sqrt)  # [batch_size, num_nodes, num_nodes]
        
        # Symmetric normalization: D_hat^(-0.5) * A_hat * D_hat^(-0.5)
        # This ensures stable information flow regardless of node degree
        adj_norm = torch.bmm(torch.bmm(degree_matrix_inv_sqrt, adj_hat), degree_matrix_inv_sqrt)
        # [batch_size, num_nodes, num_nodes]
        
        # Remove batch dimension if input was unbatched
        if was_unbatched:
            adj_norm = adj_norm.squeeze(0)  # [num_nodes, num_nodes]
        
        return adj_norm
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor, is_pre_normalized: bool = True) -> torch.Tensor:
        """
        Forward pass of the GCN layer.
        
        PERFORMANCE FIX: Adjacency matrix is now expected to be PRE-NORMALIZED during data loading.
        This eliminates millions of redundant matrix operations during training.
        
        For static models: adj is pre-normalized (is_pre_normalized=True, default).
        For adaptive models: adj may need normalization if it's a combined (static+learned) matrix.
        
        Implements: H^(l+1) = σ(A_norm @ (H^(l) @ W^(l)))
        Where A_norm is the pre-normalized adjacency (or normalized on-the-fly for adaptive).
        
        Args:
            x: Node features [batch_size, num_nodes, in_features]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes] or [num_nodes, num_nodes]
                 For static models: PRE-NORMALIZED (computed once in data loader)
                 For adaptive models: May be unnormalized combined matrix (will be normalized here)
            is_pre_normalized: Whether adj is already normalized (default: True for static models)
        
        Returns:
            output: Transformed node features [batch_size, num_nodes, out_features]
        """
        # For static models: adj is already pre-normalized (computed once in data loader)
        # For adaptive models: normalize the combined (static + learned) adjacency
        if is_pre_normalized:
            adj_norm = adj
        else:
            # Only normalize if not pre-normalized (e.g., adaptive models with combined adjacency)
            adj_norm = self.normalize_adjacency(adj)  # [batch_size, num_nodes, num_nodes]
        
        # Handle unbatched adjacency (if adj was 2D, adj_norm is 2D)
        if adj_norm.dim() == 2:
            adj_norm = adj_norm.unsqueeze(0)  # [1, num_nodes, num_nodes]
            was_unbatched = True
        else:
            was_unbatched = False
        
        # Core GCN operation: output = A_norm @ (X @ W)
        # Step 1: Linear transformation: X @ W
        # x: [batch_size, num_nodes, in_features]
        # weight: [in_features, out_features]
        support = torch.matmul(x, self.weight)  # [batch_size, num_nodes, out_features]
        
        # Step 2: Graph convolution: A_norm @ support
        # adj_norm: [batch_size, num_nodes, num_nodes]
        # support: [batch_size, num_nodes, out_features]
        output = torch.bmm(adj_norm, support)  # [batch_size, num_nodes, out_features]
        
        # Step 3: Add bias (if enabled)
        if self.bias is not None:
            output = output + self.bias  # Broadcasting: [batch_size, num_nodes, out_features]
        
        # Step 4: Apply activation function
        if self.activation_name == 'relu':
            output = F.relu(output)
        elif self.activation_name == 'tanh':
            output = torch.tanh(output)
        elif self.activation_name == 'sigmoid':
            output = torch.sigmoid(output)
        elif self.activation_name is None or self.activation_name == 'none':
            pass  # No activation
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
        
        # Remove batch dimension if input was unbatched
        if was_unbatched:
            output = output.squeeze(0)  # [num_nodes, out_features]
        
        return output


class ProfessionalGCNLayerNoActivation(ProfessionalGCNLayer):
    """
    Professional GCN Layer without activation (for output layers or when activation is applied separately).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias, activation=None)

