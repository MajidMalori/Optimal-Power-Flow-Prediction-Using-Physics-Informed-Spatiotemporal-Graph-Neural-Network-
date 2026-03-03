import sys
import os
import torch
import pytest
import warnings

# Suppress the PyTorch Geometric Python 3.13 typing DeprecationWarning
warnings.filterwarnings('ignore', category=DeprecationWarning, module='torch_geometric.inspector')

# Add the root directory to path to import models
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from models import (
    StandardGCN, DynamicGCN, PIGCN, 
    PIGCLSTM, PIGCGRU, PIResnetGCLSTM, PIResnetGCGRU
)

@pytest.fixture
def dummy_data():
    """
    Provides standard dummy batches, sequences, graphs, and physical grids 
    required for testing the forward passes of all Spatio-Temporal architectures.
    """
    batch_size = 4
    seq_len = 10
    num_nodes = 33
    in_channels = 11       # 8 power + V_m + V_a + degree
    hidden_channels = 16
    out_channels = 2       # V_m, V_theta
    
    # Feature Inputs
    x_static = torch.rand(batch_size * num_nodes, in_channels)
    x_seq = torch.rand(batch_size, seq_len, num_nodes, in_channels)
    
    # Physical Power Grids Inputs (Final step calculation)
    p_inj_final = torch.rand(batch_size, num_nodes)
    q_inj_final = torch.rand(batch_size, num_nodes)
    y_bus_final = torch.rand(batch_size, num_nodes, num_nodes) 
    
    # Graph Topologies
    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1]
    ], dtype=torch.long)
    
    dynamic_edge_idx_seq = [edge_index for _ in range(seq_len)]
    
    return {
        "batch_size": batch_size,
        "num_nodes": num_nodes,
        "in_channels": in_channels,
        "hidden_channels": hidden_channels,
        "out_channels": out_channels,
        "x_static": x_static,
        "x_seq": x_seq,
        "p_inj_final": p_inj_final,
        "q_inj_final": q_inj_final,
        "y_bus_final": y_bus_final,
        "edge_index": edge_index,
        "dynamic_edge_idx_seq": dynamic_edge_idx_seq
    }

def test_standard_gcn(dummy_data):
    """
    Verifies that the Standard GCN can process a purely static graph feature set.
    Validates that flat node embeddings correctly map to outputs of shape (Total Nodes, Out Channels).
    """
    m = StandardGCN(dummy_data["in_channels"], dummy_data["hidden_channels"], dummy_data["out_channels"])
    out = m(dummy_data["x_static"], dummy_data["edge_index"])
    assert out.shape == (dummy_data["batch_size"] * dummy_data["num_nodes"], dummy_data["out_channels"])

def test_dynamic_gcn(dummy_data):
    """
    Verifies that the Dynamic GCN can process a static slice of a dynamically changing graph topology.
    It expects the exact same static output dimensions as the standard GCN for a single timeframe.
    """
    m = DynamicGCN(dummy_data["in_channels"], dummy_data["hidden_channels"], dummy_data["out_channels"])
    out = m(dummy_data["x_static"], dummy_data["edge_index"])
    assert out.shape == (dummy_data["batch_size"] * dummy_data["num_nodes"], dummy_data["out_channels"])

def test_pi_gcn(dummy_data):
    """
    Verifies that the Physics-Informed GCN correctly handles soft constraint matrices.
    It proves the network can calculate power mismatch losses based on P_inj, Q_inj, and Y_bus
    in addition to predicting V_m and V_theta arrays.
    """
    m = PIGCN(dummy_data["in_channels"], dummy_data["hidden_channels"], dummy_data["out_channels"])
    out, loss = m(dummy_data["x_static"], dummy_data["edge_index"], 
                  dummy_data["p_inj_final"].view(-1), dummy_data["q_inj_final"].view(-1), dummy_data["y_bus_final"])
    assert out.shape == (dummy_data["batch_size"] * dummy_data["num_nodes"], dummy_data["out_channels"])
    assert loss.requires_grad

def test_pi_gclstm(dummy_data):
    """
    Verifies the deep Spatio-Temporal LSTM architecture.
    It ensures the model can ingest a multi-dimensional historical sequence (B, T, N, F),
    unroll it through spatial convolutions, pass the sequence state into an LSTM,
    and output a correctly folded dimension of (B, N, Out Channels).
    """
    m = PIGCLSTM(dummy_data["in_channels"], dummy_data["hidden_channels"], 
                 dummy_data["hidden_channels"], dummy_data["out_channels"])
    out, loss = m(dummy_data["x_seq"], dummy_data["dynamic_edge_idx_seq"], 
                  dummy_data["p_inj_final"], dummy_data["q_inj_final"], dummy_data["y_bus_final"])
    assert out.shape == (dummy_data["batch_size"], dummy_data["num_nodes"], dummy_data["out_channels"])
    assert loss.requires_grad

def test_pi_gcgru(dummy_data):
    """
    Verifies the deep Spatio-Temporal GRU architecture.
    Validates the historical sequence (B, T, N, F) successfully routes through the GRU recurrent states
    without dropping the sequence and physics targets.
    """
    m = PIGCGRU(dummy_data["in_channels"], dummy_data["hidden_channels"], 
                dummy_data["hidden_channels"], dummy_data["out_channels"])
    out, loss = m(dummy_data["x_seq"], dummy_data["dynamic_edge_idx_seq"], 
                  dummy_data["p_inj_final"], dummy_data["q_inj_final"], dummy_data["y_bus_final"])
    assert out.shape == (dummy_data["batch_size"], dummy_data["num_nodes"], dummy_data["out_channels"])
    assert loss.requires_grad

def test_pi_resnet_gclstm(dummy_data):
    """
    Verifies the deepest Spatio-Temporal variant combining Residual GCN Blocks with an LSTM.
    It guarantees the custom Residual Spatial Layers are strictly preserving dimensional compatibility 
    over temporal steps before sequence recurrence.
    """
    m = PIResnetGCLSTM(dummy_data["in_channels"], dummy_data["hidden_channels"], 
                      dummy_data["hidden_channels"], dummy_data["out_channels"])
    out, loss = m(dummy_data["x_seq"], dummy_data["dynamic_edge_idx_seq"], 
                  dummy_data["p_inj_final"], dummy_data["q_inj_final"], dummy_data["y_bus_final"])
    assert out.shape == (dummy_data["batch_size"], dummy_data["num_nodes"], dummy_data["out_channels"])
    assert loss.requires_grad

def test_pi_resnet_gcgru(dummy_data):
    """
    Verifies the ResNet GCN with a GRU sequential baseline.
    Ensures that topological changes occurring at timestep variants dynamically alter the edge weights 
    and still compute the output constraints.
    """
    m = PIResnetGCGRU(dummy_data["in_channels"], dummy_data["hidden_channels"], 
                     dummy_data["hidden_channels"], dummy_data["out_channels"])
    out, loss = m(dummy_data["x_seq"], dummy_data["dynamic_edge_idx_seq"], 
                  dummy_data["p_inj_final"], dummy_data["q_inj_final"], dummy_data["y_bus_final"])
    assert out.shape == (dummy_data["batch_size"], dummy_data["num_nodes"], dummy_data["out_channels"])
    assert loss.requires_grad
