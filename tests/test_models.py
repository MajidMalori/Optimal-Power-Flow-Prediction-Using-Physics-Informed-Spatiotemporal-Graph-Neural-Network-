import os
import sys
import warnings

import pytest
import torch
import lightning as L

# Suppress the PyTorch Geometric Python 3.13 typing DeprecationWarning
warnings.filterwarnings('ignore', category=DeprecationWarning, module='torch_geometric.inspector')



from models import (
    StandardGCN, DynamicGCN, PIGCN, 
    PIGCLSTM, PIGCGRU, PIResnetGCLSTM, PIResnetGCGRU,
    PowerFlowDataModule
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
    num_targets = 10       # Full target tensor width
    num_branches = 5       # Dummy branch count
    
    # Feature Inputs - now matching (B, N, F) for spatial only too
    x_static = torch.rand(batch_size, num_nodes, in_channels)
    x_seq = torch.rand(batch_size, seq_len, num_nodes, in_channels)
    # Full 10-column targets matching data_module collate format
    targets = torch.rand(batch_size, num_nodes, num_targets)
    targets_seq = torch.rand(batch_size, num_nodes, num_targets)
    
    # Graph Topologies
    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1]
    ], dtype=torch.long)
    
    # Lists of tensors for edge_index to match collate_fn
    batch_edge_index = [edge_index for _ in range(batch_size)]
    batch_edge_index_seq = [[edge_index for _ in range(seq_len)] for _ in range(batch_size)]
    
    # Topology IDs
    topology_ids = torch.zeros(batch_size, dtype=torch.long)
    topology_ids_seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    # Physics tensors (dummy Ybus and branch data)
    ybus = torch.randn(num_nodes, num_nodes, dtype=torch.complex128)
    branch_from = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    branch_to = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    branch_max_s_pu = torch.full((num_branches,), 10.0, dtype=torch.float32)
    
    # Dummy batches for Lightning steps
    static_batch = {
        "features": x_static,
        "edge_index": batch_edge_index,
        "topology_ids": topology_ids,
        "targets": targets,
        "ybus": ybus,
        "branch_from": branch_from,
        "branch_to": branch_to,
        "branch_max_s_pu": branch_max_s_pu,
    }
    
    seq_batch = {
        "features": x_seq,
        "edge_index_seq": batch_edge_index_seq,
        "topology_ids": topology_ids_seq,
        "targets": targets_seq,
        "ybus": ybus,
        "branch_from": branch_from,
        "branch_to": branch_to,
        "branch_max_s_pu": branch_max_s_pu,
    }

    return {
        "batch_size": batch_size,
        "num_nodes": num_nodes,
        "in_channels": in_channels,
        "hidden_channels": hidden_channels,
        "out_channels": out_channels,
        "x_static": x_static,
        "x_seq": x_seq,
        "edge_index": batch_edge_index,
        "dynamic_edge_idx_seq": batch_edge_index_seq,
        "static_batch": static_batch,
        "seq_batch": seq_batch
    }

def test_standard_gcn(dummy_data):
    m = StandardGCN(dummy_data["in_channels"], dummy_data["hidden_channels"], dummy_data["out_channels"])
    out = m(dummy_data["x_static"], dummy_data["edge_index"])
    assert out.shape == (dummy_data["batch_size"], dummy_data["num_nodes"], dummy_data["out_channels"])
    trainer = L.Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False)
    # create a dummy dataloader
    dl = torch.utils.data.DataLoader([dummy_data["static_batch"]], batch_size=None)
    trainer.fit(m, train_dataloaders=dl, val_dataloaders=dl)

def test_dynamic_gcn(dummy_data):
    m = DynamicGCN(dummy_data["in_channels"], dummy_data["hidden_channels"], dummy_data["out_channels"])
    out = m(dummy_data["x_static"], dummy_data["edge_index"])
    assert out.shape == (dummy_data["batch_size"], dummy_data["num_nodes"], dummy_data["out_channels"])
    trainer = L.Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False)
    dl = torch.utils.data.DataLoader([dummy_data["static_batch"]], batch_size=None)
    trainer.fit(m, train_dataloaders=dl, val_dataloaders=dl)

def test_pi_gcn(dummy_data):
    m = PIGCN(dummy_data["in_channels"], dummy_data["hidden_channels"], dummy_data["out_channels"])
    out = m(dummy_data["x_static"], dummy_data["edge_index"])
    assert out.shape == (dummy_data["batch_size"], dummy_data["num_nodes"], dummy_data["out_channels"])
    trainer = L.Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False)
    dl = torch.utils.data.DataLoader([dummy_data["static_batch"]], batch_size=None)
    trainer.fit(m, train_dataloaders=dl, val_dataloaders=dl)

def test_pi_gclstm(dummy_data):
    m = PIGCLSTM(dummy_data["in_channels"], dummy_data["hidden_channels"], 
                 dummy_data["hidden_channels"], dummy_data["out_channels"])
    out = m(dummy_data["x_seq"], dummy_data["dynamic_edge_idx_seq"])
    # Original models reshape to (batch, nodes, out_channels), but target is (batch*nodes, out_channels)
    assert out.shape == (dummy_data["batch_size"], dummy_data["num_nodes"], dummy_data["out_channels"])
    trainer = L.Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False)
    dl = torch.utils.data.DataLoader([dummy_data["seq_batch"]], batch_size=None)
    trainer.fit(m, train_dataloaders=dl, val_dataloaders=dl)

def test_pi_gcgru(dummy_data):
    m = PIGCGRU(dummy_data["in_channels"], dummy_data["hidden_channels"], 
                dummy_data["hidden_channels"], dummy_data["out_channels"])
    out = m(dummy_data["x_seq"], dummy_data["dynamic_edge_idx_seq"])
    assert out.shape == (dummy_data["batch_size"], dummy_data["num_nodes"], dummy_data["out_channels"])
    trainer = L.Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False)
    dl = torch.utils.data.DataLoader([dummy_data["seq_batch"]], batch_size=None)
    trainer.fit(m, train_dataloaders=dl, val_dataloaders=dl)

def test_pi_resnet_gclstm(dummy_data):
    m = PIResnetGCLSTM(dummy_data["in_channels"], dummy_data["hidden_channels"], 
                      dummy_data["hidden_channels"], dummy_data["out_channels"])
    out = m(dummy_data["x_seq"], dummy_data["dynamic_edge_idx_seq"])
    assert out.shape == (dummy_data["batch_size"], dummy_data["num_nodes"], dummy_data["out_channels"])
    trainer = L.Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False)
    dl = torch.utils.data.DataLoader([dummy_data["seq_batch"]], batch_size=None)
    trainer.fit(m, train_dataloaders=dl, val_dataloaders=dl)

def test_pi_resnet_gcgru(dummy_data):
    m = PIResnetGCGRU(dummy_data["in_channels"], dummy_data["hidden_channels"], 
                     dummy_data["hidden_channels"], dummy_data["out_channels"])
    out = m(dummy_data["x_seq"], dummy_data["dynamic_edge_idx_seq"])
    assert out.shape == (dummy_data["batch_size"], dummy_data["num_nodes"], dummy_data["out_channels"])
    trainer = L.Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False)
    dl = torch.utils.data.DataLoader([dummy_data["seq_batch"]], batch_size=None)
    trainer.fit(m, train_dataloaders=dl, val_dataloaders=dl)

def test_datamodule_setup():
    """Verify that DataModule can initialize without failing."""
    dm = PowerFlowDataModule(data_dir="data/03_processed", case_name="case33")
    assert dm.batch_size == 32
    assert dm.seq_len == 1
    assert hasattr(dm, "collate_fn")

