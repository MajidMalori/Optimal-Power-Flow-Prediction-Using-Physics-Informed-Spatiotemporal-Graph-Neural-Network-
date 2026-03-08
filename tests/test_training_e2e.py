import os
import pytest
import torch
import json
import numpy as np
import lightning as L
from torch.utils.data import DataLoader

from src.models import PIGCN, PowerFlowDataModule
from scripts.preprocess_data import denormalize_predictions
from src.constants import TargetIndices

# Identify if dummy processed data is available from previous tests
PROCESSED_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "data", "03_processed"
)
HAVE_PROCESSED_DATA = os.path.exists(os.path.join(PROCESSED_DATA_DIR, "case33", "train_features.pt"))


@pytest.mark.skipif(not HAVE_PROCESSED_DATA, reason="Requires preprocessed data from 'make test' / src/data pipeline")
def test_lightning_fast_dev_run_and_denormalize(case_name):
    """
    End-to-End Test:
    1. Loads the actual preprocessed dataset using PowerFlowDataModule.
    2. Initializes a Physics-Informed model (PIGCN).
    3. Runs a complete Lightning training/validation cycle using fast_dev_run=True (1 batch).
    4. Validates that the prediction outputs can be safely denormalized back to correct units.
    """
    
    # 1. Initialize Datamodule
    dm = PowerFlowDataModule(
        data_dir=PROCESSED_DATA_DIR,
        case_name=case_name,
        batch_size=8,
        seq_len=1
    )
    dm.setup(stage="fit")
    
    # Check that physics tensors loaded correctly
    assert len(dm.branch_from) > 0, "branch_from not loaded"
    assert len(dm.branch_to) > 0, "branch_to not loaded"
    assert len(dm.branch_max_s_pu) > 0, "branch_max_s_pu not loaded"
    
    # 2. Initialize Model
    # Determine input channels from a batch
    batch = next(iter(dm.train_dataloader()))
    in_channels = batch["features"].shape[-1]
    
    model = PIGCN(
        in_channels=in_channels,
        out_channels=2,
        hidden_channels=32,
        num_layers=2
    )
    
    # 3. Fast Dev Run Training Loop (Executes 1 Train & 1 Val batch)
    trainer = L.Trainer(
        fast_dev_run=True, # Runs 1 batch of train, val, test to find any bugs
        logger=False,
        enable_checkpointing=False
    )
    
    # This will exercise the _shared_step, compute data loss, and compute physics loss
    trainer.fit(model, datamodule=dm)
    
    # If the trainer finished without exception, the physics loss and model shapes are aligned.
    
    # 4. Denormalization Test
    # Simulate a prediction from the model during inference.
    model.eval()
    with torch.no_grad():
        preds = model(batch["features"], batch["edge_index"])  # [B, N, 2]
        
    preds = preds.numpy()
    batch_size = preds.shape[0]
    num_nodes = preds.shape[1]
    
    # Targets prediction outputs are VM and VA. 
    # To use `denormalize_predictions`, we need to map the 2 output channels to a full 10-channel array
    # or just denormalize the VM column (index 0 for our predictions).
    
    norm_path = os.path.join(PROCESSED_DATA_DIR, case_name, "normalization.json")
    with open(norm_path, "r") as f:
        meta = json.load(f)
        
    # We construct a dummy output tensor representing the 10-column target format 
    # where the last 2 columns are VM and VA, just to test the denormalize function logic.
    mock_full_preds = np.zeros((batch_size, num_nodes, 10))
    mock_full_preds[..., TargetIndices.VM] = preds[..., 0]   # Assign predicted VM deviation
    mock_full_preds[..., TargetIndices.VA] = preds[..., 1]   # Assign predicted VA deviation
    
    # Let's also mock a power target to see if it gets scaled by s_base
    mock_full_preds[..., TargetIndices.P_LOAD] = 0.5  # 0.5 p.u.
    
    # Run Denormalization
    denorm_preds = denormalize_predictions(mock_full_preds, meta)
    
    # Verify VM shifted back ~1.0
    # Before denorm, model predicts deviation around 0. 
    # After denorm, it should add vm_nominal (1.0).
    assert not np.isnan(denorm_preds).any(), "Denormalized predictions contain NaNs"
    
    # Verify Power mapping
    # 0.5 p.u. * s_base
    assert np.allclose(denorm_preds[..., TargetIndices.P_LOAD], 0.5 * meta['s_base']), "Power denormalization failed to scale by S_base."
    
    # Verify VM mapping
    # 0 (deviation) + 1.0 (nominal) = 1.0
    original_vm_mean = preds[..., 0].mean()
    denorm_vm_mean = denorm_preds[..., TargetIndices.VM].mean()
    # Allowing minor precision variations
    assert np.isclose(denorm_vm_mean, original_vm_mean + meta['vm_nominal']), "VM denormalization failed to shift by vm_nominal."
    
    print("End-to-End training loop (1 batch) and denormalization logic completely validated.")
