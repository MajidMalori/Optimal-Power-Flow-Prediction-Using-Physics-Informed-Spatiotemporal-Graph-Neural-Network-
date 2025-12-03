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
    Unified evaluation function.
    
    Args:
        return_denormalized (bool): 
            If True, returns metrics on DENORMALIZED (physical) data (MSE, MAE, RMSE).
            If False, returns metrics on NORMALIZED data (matches training loss).
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            adj = batch['adjacency'].to(device)
            
            out = model(features, adj)
            all_preds.append(out)  # Keep on GPU
            all_targets.append(targets)  # Keep on GPU
            
    # Concatenate on GPU, convert to CPU once
    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Handle shape consistency for flattened outputs
    if predictions.dim() == 2:
        batch_size = predictions.shape[0]
        num_features = 10 # Assumed 10 for PI models
        if predictions.shape[1] % num_features == 0:
            num_buses = predictions.shape[1] // num_features
            predictions = predictions.view(batch_size, num_buses, num_features)
            if targets.dim() == 2 and targets.shape[1] == predictions.shape[0] * num_buses * num_features:
                 targets = targets.view(batch_size, num_buses, num_features)

    if return_denormalized:
        # Denormalize
        predictions = normalizer.denormalize(predictions)
        targets = normalizer.denormalize(targets)
    
    mse = torch.nn.functional.mse_loss(predictions, targets).item()
    mae = torch.nn.functional.l1_loss(predictions, targets).item()
    rmse = np.sqrt(mse)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }

def evaluate_renewable_impacts_from_predictions(preds_phys, uncertainties, normalizer):
    """
    Evaluates renewable penetration impact from already-computed MC Dropout predictions.
    This avoids re-running MC Dropout (reuses results for efficiency).
    
    Args:
        preds_phys: Denormalized predictions [samples, buses, 10]
        uncertainties: MC Dropout uncertainties [samples, buses, features]
        normalizer: PowerSystemNormalizer instance (for base_mva)
    
    Returns:
        DataFrame with renewable impact analysis
    """
    # Calculate Renewable Fraction per sample
    # Requires 10-feature output to identify P_REN/P_CONV
    if preds_phys.shape[-1] == 10:
        p_ren = preds_phys[..., FeatureIndices.P_REN].sum(dim=1)
        p_conv = preds_phys[..., FeatureIndices.P_CONV].sum(dim=1)
        ren_frac = p_ren / (p_ren + p_conv + 1e-6)
        
        vm = preds_phys[..., FeatureIndices.VM]
        voltage_dev = torch.mean(torch.abs(vm - 1.0), dim=1)
        
        p_load = preds_phys[..., FeatureIndices.P_LOAD].sum(dim=1)
        p_gen = (preds_phys[..., FeatureIndices.P_EXT_GRID] + 
                 preds_phys[..., FeatureIndices.P_CONV] + 
                 preds_phys[..., FeatureIndices.P_REN]).sum(dim=1)
        power_loss = torch.abs(p_gen - p_load) / normalizer.base_mva 
        
        carbon = (preds_phys[..., FeatureIndices.P_CONV] + 
                  torch.nn.functional.relu(preds_phys[..., FeatureIndices.P_EXT_GRID])).sum(dim=1)
        
        uncertainty = torch.mean(uncertainties, dim=(1, 2)) 
        
        results_df = pd.DataFrame({
            'renewable_fraction': ren_frac.cpu().numpy(),
            'voltage_deviation': voltage_dev.cpu().numpy(),
            'power_loss': power_loss.cpu().numpy(),
            'carbon_proxy': carbon.cpu().numpy(),
            'uncertainty': uncertainty.cpu().numpy()
        })
        return results_df
    else:
        return pd.DataFrame()


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
    if not results_df.empty:
        results_df.to_csv(os.path.join(save_dir, 'detailed_results.csv'), index=False)
    # Quietly save results (no verbose output)
    # print(f"Results saved to {save_dir}")

def evaluate_model_with_uncertainty(model, test_loader, device, config, normalizer, is_sequential=False):
    """
    Evaluate model and return predictions with uncertainty data for plotting.
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    all_bus_types = []
    all_model_outputs = []
    all_renewable_fractions = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            adj = batch['adjacency'].to(device)
            bus_types = batch.get('bus_types', None)
            renewable_fraction = batch['renewable_fraction']  # REQUIRED - fail fast if missing
            
            # Forward pass with MC Dropout for uncertainty (vectorized)
            model.train()  # Enable dropout
            mc_samples = 50
            batch_size = features.shape[0]
            
            # Vectorize: stack all MC samples in one batch dimension
            # Handle 2D [batch, features], 3D [batch, seq_len, features], and 4D [batch, seq_len, buses, features]
            if features.dim() == 4:
                # Sequential model: [batch, seq_len, buses, features] -> [mc_samples * batch, seq_len, buses, features]
                features_expanded = features.repeat(mc_samples, 1, 1, 1)
            elif features.dim() == 3:
                # Sequential model (flattened) or other 3D: [batch, seq_len, features] -> [mc_samples * batch, seq_len, features]
                features_expanded = features.repeat(mc_samples, 1, 1)
            else:
                # 2D: [batch, features] -> [mc_samples * batch, features]
                features_expanded = features.repeat(mc_samples, 1)
            
            # Adjacency matrix: always 3D [batch, nodes, nodes]
            if adj.dim() == 3:
                adj_expanded = adj.repeat(mc_samples, 1, 1)
            else:
                adj_expanded = adj.repeat(mc_samples, 1, 1)
            
            predictions_mc = model(features_expanded, adj_expanded)
            # Reshape: [mc_samples * batch_size, ...] -> [mc_samples, batch_size, ...]
            pred_shape = predictions_mc.shape[1:]
            predictions_mc = predictions_mc.view(mc_samples, batch_size, *pred_shape)
            
            predictions_mean = predictions_mc.mean(dim=0)
            predictions_std = predictions_mc.std(dim=0)
            
            # Keep on GPU for now (convert to CPU once at end)
            all_predictions.append(predictions_mean)
            all_targets.append(targets)
            all_uncertainties.append(predictions_std)
            all_model_outputs.append(predictions_mean)
            all_renewable_fractions.append(renewable_fraction)
            
            if bus_types is not None:
                all_bus_types.append(bus_types)
    
    # Concatenate on GPU (faster than CPU)
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
            
    # Calculate metrics on GPU (before denormalization)
    mse = float(torch.mean((predictions - targets) ** 2).item())
    mae = float(torch.mean(torch.abs(predictions - targets)).item())
    
    # Denormalize on GPU (avoid CPU round-trip)
    predictions_phys = normalizer.denormalize(predictions)
    targets_phys = normalizer.denormalize(targets)
    
    # Convert to numpy only once at the end
    predictions_phys = predictions_phys.cpu().numpy()
    targets_phys = targets_phys.cpu().numpy()
    uncertainties = uncertainties.cpu().numpy()
    
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(np.sqrt(mse))
    }
    
    # Concatenate on GPU then convert to numpy (more efficient)
    model_outputs_norm = torch.cat(all_model_outputs, dim=0) if all_model_outputs else None
    bus_types_np = torch.cat(all_bus_types, dim=0).cpu().numpy() if all_bus_types else None
    renewable_fractions_np = torch.cat(all_renewable_fractions, dim=0).cpu().numpy()
    
    # For calibration diagram: Keep model_outputs in normalized space
    # because uncertainties are in normalized space, and calibration should compare
    # uncertainties (normalized) with errors (normalized) for proper scale matching
    if model_outputs_norm is not None:
        model_outputs_norm_np = model_outputs_norm.cpu().numpy()
        targets_norm_np = targets.cpu().numpy()
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
    if not is_physics_informed:
        return {}, None
    
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
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            adj = batch['adjacency'].to(device)
            ybus = batch['ybus_matrix'].to(device)
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
            
            # Calculate per-sample MSE (keep on GPU, convert once at end)
            mse_per_sample = ((preds_phys - targets_phys) ** 2).mean(dim=(1, 2))
            all_mse_per_sample.append(mse_per_sample)
            
            # Compute MOOPF metrics (now returns per-sample metrics)
            metrics = compute_moopf_metrics(preds_phys, ybus, base_mva=normalizer.base_mva)
            
            batch_size = renewable_fraction.shape[0] if renewable_fraction.dim() > 0 else 1
            
            # Extract per-sample metrics (tensors on GPU)
            all_power_loss_scores.append(metrics['power_loss_score'])  # [batch_size] tensor
            all_voltage_stability_scores.append(metrics['voltage_stability_score'])  # [batch_size] tensor
            all_carbon_scores.append(metrics['carbon_score'])  # [batch_size] tensor
            all_renewable_fractions.append(renewable_fraction)  # Keep on GPU
    
    # Convert to CPU once at the end (batch conversion)
    all_mse_per_sample = torch.cat(all_mse_per_sample, dim=0).cpu().numpy().tolist()
    all_renewable_fractions = torch.cat(all_renewable_fractions, dim=0).cpu().numpy().flatten() if all_renewable_fractions else []
    
    # Concatenate per-sample metrics from all batches
    all_power_loss_scores = torch.cat(all_power_loss_scores, dim=0).cpu().numpy()
    all_voltage_stability_scores = torch.cat(all_voltage_stability_scores, dim=0).cpu().numpy()
    all_carbon_scores = torch.cat(all_carbon_scores, dim=0).cpu().numpy()
    
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
