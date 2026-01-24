import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Tuple, List, Any
from datetime import datetime
from config import FeatureIndices
from utils.visualization import create_model_comparison_plot
from utils.metrics import compute_moopf_metrics

def evaluate_performance(model, data_loader, device, config, normalizer, is_sequential=False, return_denormalized=False):
    """
    Unified evaluation function - MEMORY OPTIMIZED.
    
    Args:
        return_denormalized (bool): 
            If True, returns metrics on DENORMALIZED (physical) data (MSE, MAE, RMSE).
            If False, returns metrics on NORMALIZED data (matches training loss).
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            adj = batch['adjacency'].to(device, non_blocking=True)
            
            out = model(features, adj)
            
            # Handle shape consistency for flattened outputs
            if out.dim() == 2:
                batch_size = out.shape[0]
                num_features = 10
                if out.shape[1] % num_features == 0:
                    num_buses = out.shape[1] // num_features
                    out = out.view(batch_size, num_buses, num_features)
                    if targets.dim() == 2 and targets.shape[1] == batch_size * num_buses * num_features:
                        targets = targets.view(batch_size, num_buses, num_features)
            
            # Compute metrics incrementally (memory efficient)
            if return_denormalized:
                out = normalizer.denormalize(out)
                targets = normalizer.denormalize(targets)
                
            # FIX: Use reduction='mean' to match training loss and avoid scaling by batch size * num_elements
            # We want the average MSE per element (per bus, per feature)
            batch_mse = torch.nn.functional.mse_loss(out, targets, reduction='mean').item()
            batch_mae = torch.nn.functional.l1_loss(out, targets, reduction='mean').item()
            
            # Weight by batch size to compute correct global mean later
            batch_size_actual = out.shape[0]
            
            total_mse += batch_mse * batch_size_actual
            total_mae += batch_mae * batch_size_actual
            total_samples += batch_size_actual
            
            # Clear batch from GPU immediately
            del out, targets, features, adj
    
    # Final metrics
    # total_mse is sum(mean_batch_mse * batch_size), so dividing by total_samples gives global mean
    mse = total_mse / total_samples
    mae = total_mae / total_samples
    rmse = np.sqrt(mse)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }

def save_results(metrics, results_df, config, output_dir=None):
    """
    Saves evaluation results.
    
    Args:
        metrics: Dictionary of scalar metrics
        results_df: DataFrame with detailed results
        config: Config object (for backward compatibility)
        output_dir: Explicit output directory (takes priority over config.CURRENT_RUN_DIR)
    """
    # Use explicit output_dir if provided, otherwise fall back to config
    if output_dir is not None:
        save_dir = output_dir
    elif hasattr(config, 'get_experimental_results_dir'):
        save_dir = config.get_experimental_results_dir()
    else:
        save_dir = config.CURRENT_RUN_DIR
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save Scalar Metrics
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
            
    # Save DataFrame
    # Save DataFrame
    if not results_df.empty:
        # Always save detailed results (files are small)
        results_df.to_csv(os.path.join(save_dir, 'detailed_results.csv'), index=False)

def evaluate_model_with_uncertainty(model, test_loader, device, config, normalizer, is_sequential=False):
    """
    Evaluate model and return predictions with uncertainty data for plotting.
    MEMORY OPTIMIZED: Uses batched MC Dropout instead of repeating entire batch.
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    all_bus_types = []
    all_model_outputs = []
    all_renewable_fractions = []
    
    mc_samples = 50
    mc_batch_size = 10  # Process MC samples in smaller batches to reduce memory
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            adj = batch['adjacency'].to(device, non_blocking=True)
            bus_types = batch.get('bus_types', None)
            renewable_fraction = batch['renewable_fraction']
            
            batch_size = features.shape[0]
            pred_shape = None
            
            # MEMORY OPTIMIZED MC Dropout: Process in smaller batches
            model.train()  # Enable dropout
            predictions_mc_list = []
            
            # Process MC samples in chunks to avoid OOM
            for mc_chunk_start in range(0, mc_samples, mc_batch_size):
                mc_chunk_end = min(mc_chunk_start + mc_batch_size, mc_samples)
                mc_chunk_size = mc_chunk_end - mc_chunk_start
                
                # Repeat features/adj for this MC chunk only
                if features.dim() == 4:
                    features_chunk = features.repeat(mc_chunk_size, 1, 1, 1)
                elif features.dim() == 3:
                    features_chunk = features.repeat(mc_chunk_size, 1, 1)
                else:
                    features_chunk = features.repeat(mc_chunk_size, 1)
                
                adj_chunk = adj.repeat(mc_chunk_size, 1, 1) if adj.dim() == 3 else adj.repeat(mc_chunk_size, 1, 1)
                
                # Forward pass for this MC chunk
                pred_chunk = model(features_chunk, adj_chunk)
                if pred_shape is None:
                    pred_shape = pred_chunk.shape[1:]
                predictions_mc_list.append(pred_chunk.view(mc_chunk_size, batch_size, *pred_shape))
                
                # Clear chunk from GPU
                del features_chunk, adj_chunk, pred_chunk
            
            # Stack all MC chunks: [mc_samples, batch_size, ...]
            predictions_mc = torch.cat(predictions_mc_list, dim=0)
            del predictions_mc_list
            
            # Compute mean and std
            predictions_mean = predictions_mc.mean(dim=0)
            predictions_std = predictions_mc.std(dim=0)
            del predictions_mc
            
            # Move to CPU immediately to free GPU memory
            all_predictions.append(predictions_mean.cpu())
            all_targets.append(targets.cpu())
            all_uncertainties.append(predictions_std.cpu())
            all_model_outputs.append(predictions_mean.cpu())
            all_renewable_fractions.append(renewable_fraction.cpu() if torch.is_tensor(renewable_fraction) else renewable_fraction)
            
            if bus_types is not None:
                all_bus_types.append(bus_types.cpu() if torch.is_tensor(bus_types) else bus_types)
            
            # Clear batch from GPU
            del features, targets, adj
    
    # Concatenate on CPU (already moved there)
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    uncertainties = torch.cat(all_uncertainties, dim=0)
    
    # Handle shape consistency for flattened outputs
    if predictions.dim() == 2:
        batch_size = predictions.shape[0]
        num_features = 2
        if predictions.shape[1] % num_features == 0:
            num_buses = predictions.shape[1] // num_features
            predictions = predictions.view(batch_size, num_buses, num_features)
            uncertainties = uncertainties.view(batch_size, num_buses, num_features)
            
    # Calculate metrics (already on CPU)
    mse = float(torch.mean((predictions - targets) ** 2).item())
    mae = float(torch.mean(torch.abs(predictions - targets)).item())
    
    # Denormalize (on CPU, more memory efficient for large datasets)
    predictions_phys = normalizer.denormalize(predictions)
    targets_phys = normalizer.denormalize(targets)
    
    # Convert to numpy
    predictions_phys = predictions_phys.numpy()
    targets_phys = targets_phys.numpy()
    uncertainties = uncertainties.numpy()
    
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(np.sqrt(mse))
    }
    
    # Concatenate and convert to numpy (already on CPU)
    model_outputs_norm = torch.cat(all_model_outputs, dim=0) if all_model_outputs else None
    bus_types_np = torch.cat(all_bus_types, dim=0).numpy() if all_bus_types else None
    renewable_fractions_np = torch.cat(all_renewable_fractions, dim=0).numpy() if all_renewable_fractions else None
    
    # For calibration diagram: Keep model_outputs in normalized space
    if model_outputs_norm is not None:
        model_outputs_norm_np = model_outputs_norm.numpy()
        targets_norm_np = targets.numpy()
    else:
        model_outputs_norm_np = None
        targets_norm_np = None
    
    uncertainty_data = {
        'predictions': predictions_phys,  # Denormalized for predicted vs actual plots
        'targets': targets_phys,  # Denormalized for predicted vs actual plots
        'uncertainties': uncertainties,  # Normalized (std of normalized predictions)
        'model_outputs': model_outputs_norm_np,  # Normalized for calibration (matches uncertainties)
        'targets_norm': targets_norm_np,  # Normalized targets for calibration
        'bus_types': bus_types_np,
        'renewable_fractions': renewable_fractions_np
    }
    
    return metrics, uncertainty_data


def evaluate_moopf_objectives_normalized(model, test_loader, config, device, normalizer, is_physics_informed=True):
    """
    Evaluate Multi-Objective Optimal Power Flow objectives using the new Audit Metrics.
    """
    # ENABLE UNIVERSAL EVALUATION:
    # We remove the check "if not is_physics_informed: return {}, None"
    # This allows non-physics models to be evaluated on physical consistency (MOOPF score).
    
    model.eval()
    
    all_power_loss_scores = []  # Will store tensors
    all_voltage_stability_scores = []  # Will store tensors
    all_carbon_scores = []  # Will store tensors
    all_mse_per_sample = []
    all_renewable_fractions = []
    
    # MOOPF weights from config.yaml
    w_loss = config.MOOPF_WEIGHT_LOSS
    w_vdev = config.MOOPF_WEIGHT_VDEV
    w_carbon = config.MOOPF_WEIGHT_CARBON

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating MOOPF Objectives"):
            features = batch['features'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            adj = batch['adjacency'].to(device, non_blocking=True)
            ybus = batch['ybus_matrix'].to(device, non_blocking=True)
            renewable_fraction = batch['renewable_fraction']
            
            # Forward pass
            predictions = model(features, adj)
            
            # Handle shape consistency
            if predictions.dim() == 2:
                batch_size = predictions.shape[0]
                num_features = 10 # Assumed 10 for PI models
                if predictions.shape[1] % num_features == 0:
                    num_buses = predictions.shape[1] // num_features
                    predictions = predictions.view(batch_size, num_buses, num_features)
            
            # Denormalize for physics metrics
            preds_phys = normalizer.denormalize(predictions)
            targets_phys = normalizer.denormalize(targets)
            
            # Calculate per-sample MSE (move to CPU immediately)
            mse_per_sample = ((preds_phys - targets_phys) ** 2).mean(dim=(1, 2)).cpu()
            all_mse_per_sample.append(mse_per_sample)
            
            # Compute MOOPF metrics (now returns per-sample metrics)
            metrics = compute_moopf_metrics(preds_phys, ybus, base_mva=normalizer.base_mva)
            
            batch_size = renewable_fraction.shape[0] if torch.is_tensor(renewable_fraction) and renewable_fraction.dim() > 0 else 1
            
            # Extract per-sample metrics (move to CPU immediately to free GPU memory)
            all_power_loss_scores.append(metrics['power_loss_score'].cpu())
            all_voltage_stability_scores.append(metrics['voltage_stability_score'].cpu())
            all_carbon_scores.append(metrics['carbon_score'].cpu())
            all_renewable_fractions.append(renewable_fraction.cpu() if torch.is_tensor(renewable_fraction) else renewable_fraction)
            
            # Clear batch from GPU
            del features, targets, adj, ybus, predictions, preds_phys, targets_phys
    
    # Convert to numpy (already on CPU)
    all_mse_per_sample = torch.cat(all_mse_per_sample, dim=0).numpy().tolist()
    all_renewable_fractions = torch.cat(all_renewable_fractions, dim=0).numpy().flatten() if all_renewable_fractions else []
    
    # Concatenate per-sample metrics from all batches (already on CPU)
    all_power_loss_scores = torch.cat(all_power_loss_scores, dim=0).numpy()
    all_voltage_stability_scores = torch.cat(all_voltage_stability_scores, dim=0).numpy()
    all_carbon_scores = torch.cat(all_carbon_scores, dim=0).numpy()
    
    # Calculate batch averages for summary metrics
    moopf_results = {
        'power_loss': float(np.mean(all_power_loss_scores)),
        'voltage_deviation': float(np.mean(all_voltage_stability_scores)),
        'carbon_emissions': float(np.mean(all_carbon_scores)),
        'mse_per_sample': all_mse_per_sample
    }
    
    # Calculate MOOPF score
    moopf_score = (w_loss * moopf_results['power_loss'] + 
                   w_vdev * moopf_results['voltage_deviation'] + 
                   w_carbon * moopf_results['carbon_emissions'])
    moopf_results['moopf_score'] = moopf_score
    moopf_results['mse_score'] = moopf_score # For compatibility
    
    # Create renewable impact dataframe with per-sample metrics
    renewable_impact_data = pd.DataFrame({
        'power_loss': all_power_loss_scores,
        'voltage_deviation': all_voltage_stability_scores,
        'carbon_emissions': all_carbon_scores,
        'renewable_fraction': all_renewable_fractions
    })
    
    return moopf_results, renewable_impact_data
