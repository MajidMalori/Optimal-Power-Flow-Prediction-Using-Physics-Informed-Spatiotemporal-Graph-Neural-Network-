"""
ResnetPIGCLSTM: Physics-Informed Graph Convolutional LSTM with Residual Connections.
Thin wrapper around SpatioTemporalRNN for backward compatibility.
"""

from typing import Optional
from .spatiotemporal_rnn import SpatioTemporalRNN


class ResnetPIGCLSTM(SpatioTemporalRNN):
    """
    A Physics-Informed Graph Convolutional LSTM with Residual Connections.
    
    This is a thin wrapper around SpatioTemporalRNN with rnn_type='LSTM' and use_resnet=True.
    Maintained for backward compatibility.
    """
    def __init__(self, feature_dim: int, hidden_dim: int, num_gc_layers: int, num_buses: int, rnn_layers: int, dropout: float,
                 embedding_dim: int = 16, phi: float = 0.5, config=None, normalizer=None, **kwargs):
        """
        Args:
            feature_dim (int): The number of input features for each node.
            hidden_dim (int): The dimensionality of the hidden layers in both GCN and LSTM.
            num_gc_layers (int): The number of graph convolution layers to apply at each time step.
            num_buses (int): The number of nodes (buses) in the graph.
            rnn_layers (int): The number of layers in the LSTM.
            dropout (float): The dropout rate.
            embedding_dim (int): The dimensionality of the node embeddings for the adaptive matrix.
            phi (float): The interpolation coefficient for blending physical and learned graphs (0 <= phi <= 1).
            config: Configuration object (unused, kept for compatibility)
            normalizer: PowerSystemNormalizer (unused, kept for compatibility)
        """
        super().__init__(
            feature_dim=feature_dim, hidden_dim=hidden_dim, num_gc_layers=num_gc_layers,
            num_buses=num_buses, rnn_layers=rnn_layers, dropout=dropout,
            embedding_dim=embedding_dim, phi=phi, rnn_type='LSTM', use_resnet=True,
            config=config, normalizer=normalizer, **kwargs
        )