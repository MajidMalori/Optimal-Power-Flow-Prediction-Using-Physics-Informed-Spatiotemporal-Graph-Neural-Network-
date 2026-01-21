"""
Graph Convolutional Network Layer

This module implements a mathematically sound GCN layer following best practices:
1. Self-loops: A_hat = A + I (preserves node features)
2. Symmetric normalization: D_hat^(-0.5) * A_hat * D_hat^(-0.5) (prevents gradient explosion)
3. Proper GCN operation: output = A_norm @ features @ weight

Based on Kipf & Welling (ICLR 2017): "Semi-Supervised Classification with Graph Convolutional Networks"
Paper: https://arxiv.org/abs/1609.02907

This fixes architectural flaws identified in original implementations:
- Missing self-loops (nodes forgetting their own features)
- Missing normalization (gradient explosion from degree imbalance)
- Incorrect operation order (aggregation then MLP, not proper GCN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network Layer.
    
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
        super(GCNLayer, self).__init__()
        
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
    
    @staticmethod
    def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
        """
        DEPRECATED: Normalization is now pre-computed during data loading for performance.
        
        This method is kept for backward compatibility with adaptive models that need
        to normalize combined (static + learned) adjacency matrices.
        
        Delegates to the centralized utility in utils.contingency_ybus.
        """
        from utils.contingency_ybus import normalize_adjacency
        return normalize_adjacency(adj)
    
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


class GCNLayerNoActivation(GCNLayer):
    """
    GCN Layer without activation (for output layers or when activation is applied separately).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias, activation=None)
