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
    
    # Feature Inputs
    x_static = torch.rand(batch_size * num_nodes, in_channels)
    x_seq = torch.rand(batch_size, seq_len, num_nodes, in_channels)
    targets = torch.rand(batch_size * num_nodes, out_channels)
    targets_seq = torch.rand(batch_size, num_nodes, out_channels)
    
    # Graph Topologies
    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1]
    ], dtype=torch.long)
    
    dynamic_edge_idx_seq = [edge_index for _ in range(seq_len)]
    
    # Dummy batches for Lightning steps
    static_batch = {
        "features": x_static,
        "edge_index": edge_index,
        "targets": targets
    }
    
    seq_batch = {
        "features": x_seq,
        "edge_index_seq": dynamic_edge_idx_seq,
        "targets": targets_seq
    }

    return {
        "batch_size": batch_size,
        "num_nodes": num_nodes,
        "in_channels": in_channels,
        "hidden_channels": hidden_channels,
        "out_channels": out_channels,
        "x_static": x_static,
        "x_seq": x_seq,
        "edge_index": edge_index,
        "dynamic_edge_idx_seq": dynamic_edge_idx_seq,
        "static_batch": static_batch,
        "seq_batch": seq_batch
    }

def test_standard_gcn(dummy_data):
    m = StandardGCN(dummy_data["in_channels"], dummy_data["hidden_channels"], dummy_data["out_channels"])
    out = m(dummy_data["x_static"], dummy_data["edge_index"])
    assert out.shape == (dummy_data["batch_size"] * dummy_data["num_nodes"], dummy_data["out_channels"])
    trainer = L.Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False)
    # create a dummy dataloader
    dl = torch.utils.data.DataLoader([dummy_data["static_batch"]], batch_size=None)
    trainer.fit(m, train_dataloaders=dl, val_dataloaders=dl)

def test_dynamic_gcn(dummy_data):
    m = DynamicGCN(dummy_data["in_channels"], dummy_data["hidden_channels"], dummy_data["out_channels"])
    out = m(dummy_data["x_static"], dummy_data["edge_index"])
    assert out.shape == (dummy_data["batch_size"] * dummy_data["num_nodes"], dummy_data["out_channels"])
    trainer = L.Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False)
    dl = torch.utils.data.DataLoader([dummy_data["static_batch"]], batch_size=None)
    trainer.fit(m, train_dataloaders=dl, val_dataloaders=dl)

def test_pi_gcn(dummy_data):
    m = PIGCN(dummy_data["in_channels"], dummy_data["hidden_channels"], dummy_data["out_channels"])
    out = m(dummy_data["x_static"], dummy_data["edge_index"])
    assert out.shape == (dummy_data["batch_size"] * dummy_data["num_nodes"], dummy_data["out_channels"])
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

