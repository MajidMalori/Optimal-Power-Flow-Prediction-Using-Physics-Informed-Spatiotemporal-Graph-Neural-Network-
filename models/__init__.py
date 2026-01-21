from models.gcn import GCN
from models.adaptive_gcn import AdaptiveGCN
from models.graph_rnn import GraphRNN, PIGCLSTM, PIGCGRU, ResnetPIGCLSTM, ResnetPIGCGRU

# Expose new names directly
StandardGraphNet = GCN
AdaptiveGraphNet = AdaptiveGCN
PhysicsInformedGraphNet = AdaptiveGCN
PhysicsInformedRecurrentNet = GraphRNN

# For backward compatibility with scripts that import by string name
__all__ = [
    'GCN', 'AdaptiveGCN',
    'GraphRNN',
    'PIGCLSTM', 'PIGCGRU', 'ResnetPIGCLSTM', 'ResnetPIGCGRU',
    'StandardGraphNet', 'AdaptiveGraphNet', 'PhysicsInformedGraphNet',
    'PhysicsInformedRecurrentNet'
]
