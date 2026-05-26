import os
import json
import torch
import numpy as np
from src.constants import TargetIndices, PredictionIndices
from scripts.preprocess_data import denormalize_predictions

def predict_voltages_for_state(state, model, model_name, device, meta):
    """
    Run the neural network model on the BenchmarkState features and active edges
    to predict voltage magnitudes and angles, and denormalize them to physical units.
    """
    # Reconstruct tensor x and edge_index from BenchmarkState
    x = torch.tensor(state.features, dtype=torch.float32).unsqueeze(0).to(device)  # [1, N, F]
    edge_index = torch.tensor(state.active_edges, dtype=torch.long).t().to(device)  # [2, E]
    
    # Recurrent models require [B, Seq, N, F]
    is_recurrent = any(m in model_name for m in ["LSTM", "GRU"])
    if is_recurrent:
        # Add seq dimension: [B, Seq, N, F]
        x = x.unsqueeze(1)  # [1, 1, N, F]
        model_edge_index = [[edge_index]]
    else:
        model_edge_index = [edge_index]
        
    with torch.no_grad():
        preds = model(x, model_edge_index)
        
    # Extract prediction arrays depending on tensor shape
    if preds.dim() == 4:
        pred_vm = preds[0, -1, :, PredictionIndices.VM].cpu().numpy()
        pred_va = preds[0, -1, :, PredictionIndices.VA].cpu().numpy()
    else:
        pred_vm = preds[0, :, PredictionIndices.VM].cpu().numpy()
        pred_va = preds[0, :, PredictionIndices.VA].cpu().numpy()
        
    # Denormalize predictions using standard metadata structure
    mock_full = np.zeros((x.shape[2] if is_recurrent else x.shape[1], TargetIndices.NUM_TARGETS))
    mock_full[..., TargetIndices.VM] = pred_vm
    mock_full[..., TargetIndices.VA] = pred_va
    denorm = denormalize_predictions(mock_full[np.newaxis, ...], meta)[0]
    
    pred_vm_phys = denorm[..., TargetIndices.VM]
    pred_va_phys = denorm[..., TargetIndices.VA]
    
    return pred_vm_phys, pred_va_phys
