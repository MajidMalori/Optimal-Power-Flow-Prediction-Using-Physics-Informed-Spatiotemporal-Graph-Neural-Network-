import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ResidualGCNBlock(nn.Module):
    """
    A GCN block with a residual connection to prevent over-smoothing in deep GNNs.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        
        # Projection shortcut if dimensions change
        self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, edge_index, edge_weight=None):
        identity = self.shortcut(x)
        
        out = F.relu(self.conv1(x, edge_index, edge_weight))
        out = self.conv2(out, edge_index, edge_weight)
        
        # Add residual connection
        out += identity
        return F.relu(out)


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """
    Kipf & Welling renormalization trick: D^(-0.5) * (A + I) * D^(-0.5).
    Handles both batched [B, N, N] and unbatched [N, N] inputs.
    """
    if adj.dim() == 2:
        adj = adj.unsqueeze(0)
        was_unbatched = True
    else:
        was_unbatched = False

    num_nodes = adj.shape[-1]
    adj_hat = adj + torch.eye(num_nodes, device=adj.device, dtype=adj.dtype).unsqueeze(0)

    degree = adj_hat.sum(dim=2) + 1e-8
    d_inv_sqrt = torch.clamp(degree.pow(-0.5), max=1e10)

    # Efficient element-wise: A_ij * d_i^(-0.5) * d_j^(-0.5)
    adj_norm = adj_hat * d_inv_sqrt.unsqueeze(2) * d_inv_sqrt.unsqueeze(1)

    return adj_norm.squeeze(0) if was_unbatched else adj_norm
