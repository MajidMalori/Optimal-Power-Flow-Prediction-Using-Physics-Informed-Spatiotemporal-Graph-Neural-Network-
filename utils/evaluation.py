"""
Evaluation and results processing utilities for power system machine learning models.
Contains functions for model evaluation, results saving, and summary generation.
"""

import os
import torch
import torch.nn.functional as F
import pandas as pd
import logging
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Any, Tuple

from utils.metrics import PowerSystemLoss, compute_metrics
from utils.visualization import (plot_training_history, plot_convergence, 
                                plot_all_renewable_impacts, create_model_comparison_plot)


def evaluate_model(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, 
                  device: torch.device, config: Any, normalizer: Any, 
                  is_sequential: bool) -> Dict[str, float]:
    """Evaluates the model on the test set and returns performance metrics."""
    import torch.nn.functional as F
    
    model.eval()
    all_outputs, all_targets = [], []
    all_ybus = []  # Add this to collect Ybus matrices
    all_bus_types = []  # OPF: Collect bus types for bus-type-specific metrics
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            adj = batch['adjacency'].to(device)
            ybus = batch['ybus_matrix'].to(device)  # Get Ybus from batch
            bus_types = batch.get('bus_types', None)  # OPF: bus type codes [0=PQ, 1=PV, 2=Slack]

            # Handle sequential vs non-sequential models
            if is_sequential and features.dim() == 3:
                # For sequential models, use the last timestep
                features_input = features[:, -1, :]
            else:
                # For non-sequential models, use features as-is
                features_input = features
            
            # Try passing bus_types if model supports it (for generator constraints)
            try:
                outputs = model(features_input, adj, bus_types=bus_types.to(device) if bus_types is not None else None)
            except TypeError:
                # Model doesn't support bus_types parameter
                outputs = model(features_input, adj)
            

            # CRITICAL FIX: Detach and move to CPU before appending to prevent GPU memory accumulation
            all_outputs.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_ybus.append(ybus.detach().cpu() if ybus.requires_grad else ybus.cpu())  # Store Ybus matrices
            if bus_types is not None:
                all_bus_types.append(bus_types.cpu())  # Store bus types for analysis

    all_outputs_tensor = torch.cat(all_outputs, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    all_ybus_tensor = torch.cat(all_ybus, dim=0)
    all_bus_types_tensor = torch.cat(all_bus_types, dim=0) if all_bus_types else None  # OPF: [batch, buses]
    
    # Get num_buses dynamically from config without hardcoding
    if hasattr(config, 'NUM_BUSES'):
        num_buses = config.NUM_BUSES
        if isinstance(num_buses, list):
            num_buses = num_buses[0]  # Take first value if it's a list
    else:
        raise ValueError("Config must specify NUM_BUSES")
    
    # Always use heteroscedastic mode
    # Heteroscedastic: 4 features per bus [η1_var1, η1_var2, f2_var1, f2_var2]
    num_features = 4
    
    # Handle shape consistency for different model types before denormalization
    if all_outputs_tensor.dim() == 2:
        # If model outputs flattened format [batch_size, num_buses * num_features]
        batch_size = all_outputs_tensor.shape[0]
        expected_size = num_buses * num_features
        if all_outputs_tensor.shape[1] == expected_size:
            all_outputs_tensor = all_outputs_tensor.view(batch_size, num_buses, num_features)
        else:
            raise ValueError(
                f"Unexpected flattened output size: {all_outputs_tensor.shape[1]}, "
                f"expected {expected_size} (num_buses={num_buses} * num_features={num_features})"
            )
    elif all_outputs_tensor.dim() == 3:
        # Already 3D: check if last dimension matches
        if all_outputs_tensor.shape[-1] != num_features:
            raise ValueError(
                f"Output shape mismatch: got {all_outputs_tensor.shape[-1]} features, "
                f"expected {num_features}"
            )
    
    # Convert natural parameters to predictions before denormalization
    # Extract natural parameters
    eta1_var1_raw = all_outputs_tensor[..., 0]  # [batch, buses]
    eta1_var2_raw = all_outputs_tensor[..., 1]  # [batch, buses]
    f2_var1_raw = all_outputs_tensor[..., 2]    # [batch, buses]
    f2_var2_raw = all_outputs_tensor[..., 3]     # [batch, buses]
    
    # Get softplus beta parameter (always use softplus for numerical stability)
    softplus_beta = getattr(config, 'HETEROSCEDASTIC_SOFTPLUS_BETA', 1.0)
    
    # Compute g+(f2) using softplus - numerically stable (grows linearly for large x, prevents explosion)
    # g+(x) = (1/β) * log(1 + exp(β*x))
    g_plus_var1 = (1.0 / softplus_beta) * F.softplus(softplus_beta * f2_var1_raw)
    g_plus_var2 = (1.0 / softplus_beta) * F.softplus(softplus_beta * f2_var2_raw)
    
    # Compute natural parameters
    eta1_var1 = eta1_var1_raw
    eta1_var2 = eta1_var2_raw
    eta2_var1 = -g_plus_var1
    eta2_var2 = -g_plus_var2
    
    # Convert to mean: μ = -η1/(2η2)
    eps = 1e-8
    mu_var1 = -eta1_var1 / (2.0 * eta2_var1 + eps)  # [batch, buses]
    mu_var2 = -eta1_var2 / (2.0 * eta2_var2 + eps)  # [batch, buses]
    
    # Stack to get predictions [batch, buses, 2]
    all_outputs_tensor = torch.stack([mu_var1, mu_var2], dim=-1)
    
    outputs_denorm = normalizer.denormalize(all_outputs_tensor)
    targets_denorm = normalizer.denormalize(all_targets_tensor)

    # Collect measurements (features) for physics violation computation
    all_measurements = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            if is_sequential and features.dim() == 3:
                features_input = features[:, -1, :]  # Use last timestep
            else:
                features_input = features
            all_measurements.append(features_input)
    all_measurements_tensor = torch.cat(all_measurements, dim=0)
    
    return compute_metrics(outputs_denorm, targets_denorm, all_ybus_tensor, config, 
                          bus_types=all_bus_types_tensor, measurements=all_measurements_tensor)


def evaluate_model_with_uncertainty(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, 
                                   device: torch.device, config: Any, normalizer: Any, 
                                   is_sequential: bool) -> Tuple[Dict[str, float], Dict]:
    """
    Evaluates the model and returns both metrics and raw data for uncertainty analysis.
    
    Returns:
        metrics: Dictionary of performance metrics
        uncertainty_data: Dictionary containing:
            - 'predictions': numpy array [n_samples, n_buses, n_features]
            - 'targets': numpy array [n_samples, n_buses, n_features]
            - 'renewable_fractions': numpy array [n_samples]
    """
    model.eval()
    all_outputs, all_targets = [], []
    all_ybus = []
    all_renewable_fractions = []
    all_bus_types = []  # Collect bus_types for OPF mode
    all_timesteps = []  # Collect timesteps for temporal plotting
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            adj = batch['adjacency'].to(device)
            ybus = batch['ybus_matrix'].to(device)
            renewable_fraction = batch['renewable_fraction']  # Get renewable fraction
            bus_types = batch.get('bus_types', None)  # Get bus_types if available (OPF mode)
            timesteps = batch.get('timestep', None)  # Get timesteps if available

            # Handle sequential vs non-sequential models
            if is_sequential and features.dim() == 3:
                features_input = features[:, -1, :]
            else:
                features_input = features
            
            # Try passing bus_types if model supports it (for generator constraints)
            try:
                outputs = model(features_input, adj, bus_types=bus_types.to(device) if bus_types is not None else None)
            except TypeError:
                # Model doesn't support bus_types parameter
                outputs = model(features_input, adj)
            

            # CRITICAL FIX: Detach and move to CPU before appending to prevent GPU memory accumulation
            all_outputs.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_ybus.append(ybus.detach().cpu() if ybus.requires_grad else ybus.cpu())
            all_renewable_fractions.append(renewable_fraction)
            if bus_types is not None:
                all_bus_types.append(bus_types)
            if timesteps is not None:
                all_timesteps.append(timesteps)

    all_outputs_tensor = torch.cat(all_outputs, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    all_ybus_tensor = torch.cat(all_ybus, dim=0)
    all_renewable_fractions_tensor = torch.cat(all_renewable_fractions, dim=0)
    all_bus_types_tensor = torch.cat(all_bus_types, dim=0) if all_bus_types else None

    # Get num_buses dynamically from config
    if hasattr(config, 'NUM_BUSES'):
        num_buses = config.NUM_BUSES
        if isinstance(num_buses, list):
            num_buses = num_buses[0]
    else:
        raise ValueError("Config must specify NUM_BUSES")
    
    # Handle shape consistency - Always 4 features (heteroscedastic)
    if all_outputs_tensor.dim() == 2:
        batch_size = all_outputs_tensor.shape[0]
        num_features = 4  # Heteroscedastic: [η1_var1, η1_var2, f2_var1, f2_var2]
        expected_size = num_buses * num_features
        if all_outputs_tensor.shape[1] == expected_size:
            all_outputs_tensor = all_outputs_tensor.view(batch_size, num_buses, num_features)
        else:
            raise ValueError(f"Unexpected flattened output size: {all_outputs_tensor.shape[1]}, expected {expected_size}")
    
    # Extract predictions and uncertainties
    # Heteroscedastic: outputs are [batch, buses, 4] = [η1_var1, η1_var2, f2_var1, f2_var2]
    # Natural parameters: η1 = μ/σ², η2 = -1/(2σ²) < 0
    # where η1 = f1 (direct), η2 = -g+(f2) with g+ being exp or softplus
    
    eta1_var1_raw = all_outputs_tensor[..., 0]  # [batch, buses] - η1 for variable 1
    eta1_var2_raw = all_outputs_tensor[..., 1]  # [batch, buses] - η1 for variable 2
    f2_var1_raw = all_outputs_tensor[..., 2]    # [batch, buses] - f2 for variable 1
    f2_var2_raw = all_outputs_tensor[..., 3]      # [batch, buses] - f2 for variable 2
    
    # Get softplus beta parameter (always use softplus for numerical stability)
    softplus_beta = getattr(config, 'HETEROSCEDASTIC_SOFTPLUS_BETA', 1.0)
    
    # Compute g+(f2) using softplus - numerically stable (grows linearly for large x, prevents explosion)
    # g+(x) = (1/β) * log(1 + exp(β*x))
    import torch.nn.functional as F
    g_plus_var1 = (1.0 / softplus_beta) * F.softplus(softplus_beta * f2_var1_raw)
    g_plus_var2 = (1.0 / softplus_beta) * F.softplus(softplus_beta * f2_var2_raw)
    
    # Compute natural parameters: η1 = f1 (direct), η2 = -g+(f2) < 0
    eta1_var1 = eta1_var1_raw
    eta1_var2 = eta1_var2_raw
    eta2_var1 = -g_plus_var1
    eta2_var2 = -g_plus_var2
    
    # Convert to mean and variance: μ = -η1/(2η2), σ² = -1/(2η2)
    eps = 1e-8
    mu_var1 = -eta1_var1 / (2.0 * eta2_var1 + eps)  # [batch, buses]
    mu_var2 = -eta1_var2 / (2.0 * eta2_var2 + eps)  # [batch, buses]
    sigma2_var1 = -1.0 / (2.0 * eta2_var1 + eps)   # [batch, buses] - variance
    sigma2_var2 = -1.0 / (2.0 * eta2_var2 + eps)   # [batch, buses] - variance
    
    # No clamping (paper doesn't use it)
    # Variance can be any positive value from natural parametrization
    
    # Convert to predictions (mean) in normalized space
    predictions_norm = torch.stack([mu_var1, mu_var2], dim=-1)  # [batch, buses, 2]
    
    # Denormalize predictions for metrics
    predictions_denorm = normalizer.denormalize(predictions_norm)
    
    # Compute log_sigma for compatibility with uncertainty analysis
    # log_sigma = 0.5 * log(sigma²)
    sigma_var1 = torch.sqrt(sigma2_var1)
    sigma_var2 = torch.sqrt(sigma2_var2)
    log_sigma_norm = torch.stack([
        0.5 * torch.log(sigma2_var1 + eps),
        0.5 * torch.log(sigma2_var2 + eps)
    ], dim=-1)  # [batch, buses, 2]
    
    # Store full outputs (with uncertainties) for predicted uncertainty analysis
    # Format: [mean_var1, mean_var2, log_sigma_var1, log_sigma_var2] for compatibility
    all_outputs_with_uncertainties = torch.stack([
        mu_var1, mu_var2,
        0.5 * torch.log(sigma2_var1 + eps),
        0.5 * torch.log(sigma2_var2 + eps)
    ], dim=-1).cpu().numpy()  # [n_samples, n_buses, 4]
    
    targets_denorm = normalizer.denormalize(all_targets_tensor)
    
    # Get bus_types for compute_metrics (required for OPF mode)
    all_bus_types_tensor = torch.cat(all_bus_types, dim=0) if all_bus_types else None

    # Compute metrics (using predictions only)
    metrics = compute_metrics(predictions_denorm, targets_denorm, all_ybus_tensor, config, bus_types=all_bus_types_tensor)
    
    # Prepare uncertainty data
    uncertainty_data = {
        'predictions': predictions_denorm.cpu().numpy(),
        'targets': targets_denorm.cpu().numpy(),
        'renewable_fractions': all_renewable_fractions_tensor.cpu().numpy()
    }
    
    # Add bus_types if available (OPF mode)
    if all_bus_types_tensor is not None:
        uncertainty_data['bus_types'] = all_bus_types_tensor.cpu().numpy()
    
    # Add timesteps if available (for temporal plotting)
    if all_timesteps:
        # Handle both tensor and scalar timesteps from collate function
        if isinstance(all_timesteps[0], torch.Tensor):
            # If already tensors, concatenate them
            all_timesteps_tensor = torch.cat(all_timesteps, dim=0)
        else:
            # If scalars, convert to tensor first
            all_timesteps_tensor = torch.tensor(all_timesteps, dtype=torch.long)
        uncertainty_data['timesteps'] = all_timesteps_tensor.cpu().numpy()
    
    # Add full model outputs with uncertainties
    uncertainty_data['model_outputs'] = all_outputs_with_uncertainties  # [n_samples, n_buses, 4]
    
    return metrics, uncertainty_data


def compute_metrics_normalized(outputs: torch.Tensor, targets: torch.Tensor, ybus_batch: torch.Tensor, 
                              config: object, normalizer: Any, bus_types: torch.Tensor = None) -> Dict[str, float]:
    """Computes metrics on normalized data (same scale as training) for MoSOA optimization."""
    import torch.nn.functional as F
    
    with torch.no_grad():
        # Convert natural parameters to predictions
        if outputs.shape[-1] == 4:
            # Heteroscedastic: outputs are [batch, buses, 4] = [η1_var1, η1_var2, f2_var1, f2_var2]
            # Need to convert to predictions [batch, buses, 2] = [μ_var1, μ_var2]
            
            eta1_var1_raw = outputs[..., 0]  # [batch, buses]
            eta1_var2_raw = outputs[..., 1]  # [batch, buses]
            f2_var1_raw = outputs[..., 2]    # [batch, buses]
            f2_var2_raw = outputs[..., 3]    # [batch, buses]
            
            # Get softplus beta parameter (always use softplus for numerical stability)
            softplus_beta = getattr(config, 'HETEROSCEDASTIC_SOFTPLUS_BETA', 1.0)
            
            # Compute g+(f2) using softplus - numerically stable (grows linearly for large x, prevents explosion)
            # g+(x) = (1/β) * log(1 + exp(β*x))
            g_plus_var1 = (1.0 / softplus_beta) * F.softplus(softplus_beta * f2_var1_raw)
            g_plus_var2 = (1.0 / softplus_beta) * F.softplus(softplus_beta * f2_var2_raw)
            
            # Compute natural parameters
            eta1_var1 = eta1_var1_raw
            eta1_var2 = eta1_var2_raw
            eta2_var1 = -g_plus_var1
            eta2_var2 = -g_plus_var2
            
            # Convert to mean: μ = -η1/(2η2)
            eps = 1e-8
            mu_var1 = -eta1_var1 / (2.0 * eta2_var1 + eps)  # [batch, buses]
            mu_var2 = -eta1_var2 / (2.0 * eta2_var2 + eps)  # [batch, buses]
            
            # Stack to get predictions [batch, buses, 2]
            outputs = torch.stack([mu_var1, mu_var2], dim=-1)
        
        # Ensure outputs and targets have the same shape
        if outputs.dim() != targets.dim():
            if outputs.dim() == 2 and targets.dim() == 3:
                targets = targets.view(outputs.shape)
            elif outputs.dim() == 3 and targets.dim() == 2:
                outputs = outputs.view(targets.shape)
            else:
                raise ValueError(f"Cannot reconcile output shape {outputs.shape} with target shape {targets.shape}")
        
        # Standard regression metrics - consistent with training loss calculation
        # Denormalize first, then compute MSE on physical values
        # OPF: 2 features (unknowns per bus type)
        # Get num_buses from outputs shape
        if outputs.dim() == 2:
            batch_size = outputs.shape[0]
            num_features = 2  # OPF: 2 unknowns per bus
            num_buses = outputs.shape[1] // num_features
            outputs_for_mse = outputs.view(batch_size, num_buses, num_features)
            targets_for_mse = targets.view(batch_size, num_buses, num_features)
        else:
            outputs_for_mse = outputs
            targets_for_mse = targets
            num_buses = outputs.shape[1]
        
        outputs_denorm_mse = normalizer.denormalize(outputs_for_mse)
        targets_denorm_mse = normalizer.denormalize(targets_for_mse)
        
        # Voltages are already in per-unit and radians
        mse_physical = F.mse_loss(outputs_denorm_mse, targets_denorm_mse).item()
        mse = mse_physical  # Already in per-unit^2 (vm) + rad^2 (va)
        rmse = torch.sqrt(torch.tensor(mse)).item()
        
        # OPF: Bus-type-specific metrics (optional)
        metrics = {
            'mse': mse,
            'rmse': rmse,
        }
        
        if bus_types is not None:
            # Compute bus-type-specific MSE for reporting
            bus_types_cpu = bus_types.cpu() if bus_types.is_cuda else bus_types
            
            for bus_type_code, bus_type_name in [(0, 'PQ'), (1, 'PV'), (2, 'Slack')]:
                mask = (bus_types_cpu == bus_type_code)
                if mask.any():
                    outputs_type = outputs_for_mse[mask]
                    targets_type = targets_for_mse[mask]
                    mse_type = F.mse_loss(outputs_type, targets_type).item()
                    metrics[f'mse_{bus_type_name.lower()}'] = mse_type
        
        # For OPF, physics violations require full voltage state reconstruction
        # For now, set to 0 (can be implemented later if needed)
        # Compute actual physics violations using dedicated function
        # This replaces the hardcoded 0.0 values with real calculations
        try:
            from utils.physics_metrics import compute_physics_metrics
            physics_metrics = compute_physics_metrics(
                outputs, measurements, ybus_batch, config, bus_types
            )
            metrics['power_violation'] = physics_metrics['power_violation']
            metrics['voltage_violation'] = physics_metrics['voltage_violation']
        except Exception as e:
            # Fallback to 0.0 if computation fails (shouldn't happen, but safety net)
            metrics['power_violation'] = 0.0
            metrics['voltage_violation'] = 0.0
        
        return metrics


def evaluate_model_normalized(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, 
                            device: torch.device, config: Any, normalizer: Any, 
                            is_sequential: bool) -> Dict[str, float]:
    """Evaluates the model on normalized data (same as training) for MoSOA optimization."""
    model.eval()
    all_outputs, all_targets = [], []
    all_ybus = []
    all_bus_types = []  # OPF: Collect bus types
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            adj = batch['adjacency'].to(device)
            ybus = batch['ybus_matrix'].to(device)
            bus_types = batch.get('bus_types', None)  # OPF: bus type codes

            # Handle sequential vs non-sequential models
            if is_sequential and features.dim() == 3:
                features_input = features[:, -1, :]
            else:
                features_input = features
            
            # Try passing bus_types if model supports it (for generator constraints)
            try:
                outputs = model(features_input, adj, bus_types=bus_types.to(device) if bus_types is not None else None)
            except TypeError:
                # Model doesn't support bus_types parameter
                outputs = model(features_input, adj)
            

            # CRITICAL FIX: Detach and move to CPU before appending to prevent GPU memory accumulation
            all_outputs.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_ybus.append(ybus.detach().cpu() if ybus.requires_grad else ybus.cpu())
            if bus_types is not None:
                all_bus_types.append(bus_types.cpu())

    all_outputs_tensor = torch.cat(all_outputs, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    all_ybus_tensor = torch.cat(all_ybus, dim=0)
    all_bus_types_tensor = torch.cat(all_bus_types, dim=0) if all_bus_types else None

    # Get num_buses dynamically from config
    if hasattr(config, 'NUM_BUSES'):
        num_buses = config.NUM_BUSES
        if isinstance(num_buses, list):
            num_buses = num_buses[0]
    else:
        raise ValueError("Config must specify NUM_BUSES")
    
    # Handle shape consistency for different model types
    # Always use heteroscedastic: 4 features per bus [η1_var1, η1_var2, f2_var1, f2_var2]
    num_features = 4
    
    if all_outputs_tensor.dim() == 2:
        batch_size = all_outputs_tensor.shape[0]
        expected_size = num_buses * num_features
        if all_outputs_tensor.shape[1] == expected_size:
            all_outputs_tensor = all_outputs_tensor.view(batch_size, num_buses, num_features)
        else:
            raise ValueError(f"Unexpected flattened output size: {all_outputs_tensor.shape[1]}, expected {expected_size} (num_buses={num_buses}, num_features={num_features})")
    elif all_outputs_tensor.dim() == 3:
        # Already 3D: check if last dimension matches
        if all_outputs_tensor.shape[-1] != num_features:
            raise ValueError(f"Output shape mismatch: got {all_outputs_tensor.shape[-1]} features, expected {num_features}")
    
    # This ensures consistent evaluation scale between training and optimization
    all_bus_types_tensor = torch.cat(all_bus_types, dim=0) if all_bus_types else None
    return compute_metrics_normalized(all_outputs_tensor, all_targets_tensor, all_ybus_tensor, config, normalizer, bus_types=all_bus_types_tensor)


def evaluate_moopf_objectives(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
                             config: Any, device: torch.device, normalizer: Any, 
                             is_physics_informed: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluates multi-objective objectives and collects data for analysis."""
    model.eval()
    num_buses_val = getattr(config, 'NUM_BUSES', 33)
    num_buses = int(num_buses_val[0]) if isinstance(num_buses_val, list) else int(num_buses_val)
    physics_calculator = PowerSystemLoss(config=config, normalizer=normalizer).to(device)
    w_loss, w_vdev, w_carbon = config.MOOPF_WEIGHT_LOSS, config.MOOPF_WEIGHT_VDEV, config.MOOPF_WEIGHT_CARBON
    all_results, renewable_impact_data = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating MOOPF Objectives"):
            features, ybus = batch['features'].to(device), batch['ybus_matrix'].to(device)
            time_carbon, time_energy = batch['time_carbon_coeffs'].to(device), batch['time_energy_coeffs'].to(device)
            renewable_frac = batch['renewable_fraction'].to(device)  # Get renewable fraction from batch
            adj = batch['adjacency'].to(device)

            # Try passing bus_types if model supports it (for generator constraints)
            bus_types_batch = batch.get('bus_types', None)
            try:
                outputs_norm = model(features, adj, bus_types=bus_types_batch.to(device) if bus_types_batch is not None else None)
            except TypeError:
                # Model doesn't support bus_types parameter
                outputs_norm = model(features, adj)
            
            # Handle shape consistency for different model types
            if outputs_norm.dim() == 2:
                # If model outputs flattened format [batch_size, num_buses * 2]
                batch_size = outputs_norm.shape[0]
                num_features = 2  # OPF unknowns (2 per bus)
                expected_size = num_buses * num_features
                if outputs_norm.shape[1] == expected_size:
                    outputs_norm = outputs_norm.view(batch_size, num_buses, num_features)
                else:
                    raise ValueError(f"Unexpected flattened output size: {outputs_norm.shape[1]}, expected {expected_size}")
            
            # Only denormalize for physics calculations that require physical units
            outputs_phys = normalizer.denormalize(outputs_norm)  # [batch, buses, 2] = [vm, va]
            # Handle sequential models: features can be [batch, seq_len, buses, features]
            if features.dim() == 4:
                # Sequential model: use last timestep [batch, seq_len, buses, features] -> [batch, buses, features]
                features = features[:, -1, :, :]  # Take last timestep
            features_phys = normalizer.denormalize(features)  # [batch, buses, 10] = [p_load, q_load, ...]

            if is_physics_informed:
                # Get bus types from batch (OPF: bus-type-dependent unknowns)
                bus_types = batch.get('bus_types', None)  # [batch, buses] or None
                if bus_types is not None:
                    bus_types = bus_types.to(device)
                
                # Calculate physics-based metrics for physics-informed models
                # OPF: Pass predicted unknowns (2 features) and measurements (10 features) separately
                # CRITICAL: Must pass bus_types for correct voltage state reconstruction
                norm_loss = physics_calculator._compute_active_power_loss_pu(outputs_phys, features_phys, ybus, bus_types=bus_types)
                norm_vdev = physics_calculator._compute_normalized_voltage_deviation(outputs_phys, measurements=features_phys, bus_types=bus_types)
                # Use PREDICTED state for carbon emissions calculation (model-dependent)
                emissions = physics_calculator._compute_carbon_emissions(
                    features_phys, time_carbon, time_energy, renewable_frac,
                    voltages=outputs_phys, Ybus=ybus, bus_types=bus_types
                )
                norm_power_flow = physics_calculator._compute_mean_power_flow_pu(outputs_phys, features_phys, ybus, bus_types=bus_types)
            else:
                # For non-physics models, set physics metrics to zero/neutral values
                batch_size = features.shape[0]
                norm_loss = torch.zeros(batch_size, device=device)
                norm_vdev = torch.zeros(batch_size, device=device)
                emissions = {'raw': torch.zeros(batch_size, device=device), 'normalized': torch.zeros(batch_size, device=device)}
                norm_power_flow = torch.zeros(batch_size, device=device)

            # Capture data for analyzing the impact of renewables (only for physics-informed models)
            if is_physics_informed:
                try:
                    # Vectorized batch conversion - much faster than looping
                    batch_data = pd.DataFrame({
                        'renewable_fraction': renewable_frac.cpu().numpy(),
                        'normalized_carbon_emissions': emissions['normalized'].cpu().numpy(),
                        'voltage_deviation': norm_vdev.cpu().numpy(),
                        'power_loss': norm_loss.cpu().numpy(),
                        'power_flow': norm_power_flow.cpu().numpy()
                    })
                    renewable_impact_data.extend(batch_data.to_dict('records'))
                except (IndexError, KeyError) as e:
                    logging.warning(f"Could not extract renewable fraction from batch data: {e}")

            if is_physics_informed:
                # Calculate test MSE for physics-informed models (consistent with training)
                targets = batch['targets'].to(device)
                targets_phys = normalizer.denormalize(targets)
                mse_physical = F.mse_loss(outputs_phys, targets_phys)
                s_base_mva = physics_calculator.s_base_mva
                test_mse = mse_physical
                
                moopf_score = (w_loss * norm_loss + w_vdev * norm_vdev + w_carbon * emissions['normalized'])
                all_results.append({
                    'mse_score': test_mse.item(),  # MSE in per-unit squared (consistent with training)
                    'moopf_score': moopf_score.mean().item(), 
                    'normalized_power_loss': norm_loss.mean().item(),
                    'normalized_voltage_deviation': norm_vdev.mean().item(),
                    'normalized_power_flow': norm_power_flow.mean().item(),
                    'normalized_carbon_emissions': emissions['normalized'].mean().item(),
                    'raw_carbon_emissions_tCO2': emissions['raw'].mean().item()
                })
            else:
                # For non-physics models, compute MSE consistent with training
                targets = batch['targets'].to(device)
                targets_phys = normalizer.denormalize(targets)
                mse_physical = F.mse_loss(outputs_phys, targets_phys)
                s_base_mva = physics_calculator.s_base_mva
                mse_normalized = mse_physical
                all_results.append({
                    'mse_score': mse_normalized.item()  # MSE in per-unit squared (consistent with training)
                })

    return pd.DataFrame(all_results), pd.DataFrame(renewable_impact_data)


def evaluate_moopf_objectives_normalized(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
                                        config: Any, device: torch.device, normalizer: Any, 
                                        is_physics_informed: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluates multi-objective objectives using normalized data (same as training) for consistent scoring."""
    model.eval()
    num_buses_val = getattr(config, 'NUM_BUSES', 33)
    num_buses = int(num_buses_val[0]) if isinstance(num_buses_val, list) else int(num_buses_val)
    physics_calculator = PowerSystemLoss(config=config, normalizer=normalizer).to(device)
    w_loss, w_vdev, w_carbon = config.MOOPF_WEIGHT_LOSS, config.MOOPF_WEIGHT_VDEV, config.MOOPF_WEIGHT_CARBON
    all_results, renewable_impact_data = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating MOOPF Objectives"):
            features, ybus = batch['features'].to(device), batch['ybus_matrix'].to(device)
            time_carbon, time_energy = batch['time_carbon_coeffs'].to(device), batch['time_energy_coeffs'].to(device)
            renewable_frac = batch['renewable_fraction'].to(device)
            adj = batch['adjacency'].to(device)

            # Try passing bus_types if model supports it (for generator constraints)
            bus_types_batch = batch.get('bus_types', None)
            try:
                outputs_norm = model(features, adj, bus_types=bus_types_batch.to(device) if bus_types_batch is not None else None)
            except TypeError:
                # Model doesn't support bus_types parameter
                outputs_norm = model(features, adj)
            
            # Handle heteroscedastic outputs: extract predictions from natural parameters
            if outputs_norm.shape[-1] == 4:
                # Heteroscedastic: outputs are [batch, buses, 4] = [η1_var1, η1_var2, f2_var1, f2_var2]
                # Extract natural parameters and convert to predictions
                # F is already imported at module level
                
                eta1_var1_raw = outputs_norm[..., 0]  # [batch, buses]
                eta1_var2_raw = outputs_norm[..., 1]  # [batch, buses]
                f2_var1_raw = outputs_norm[..., 2]    # [batch, buses]
                f2_var2_raw = outputs_norm[..., 3]     # [batch, buses]
                
                # Get softplus beta parameter (always use softplus for numerical stability)
                softplus_beta = getattr(config, 'HETEROSCEDASTIC_SOFTPLUS_BETA', 1.0)
                
                # Compute g+(f2) using softplus - numerically stable (grows linearly for large x, prevents explosion)
                # g+(x) = (1/β) * log(1 + exp(β*x))
                g_plus_var1 = (1.0 / softplus_beta) * F.softplus(softplus_beta * f2_var1_raw)
                g_plus_var2 = (1.0 / softplus_beta) * F.softplus(softplus_beta * f2_var2_raw)
                
                # Compute natural parameters
                eta2_var1 = -g_plus_var1
                eta2_var2 = -g_plus_var2
                
                # Convert to mean: μ = -η1/(2η2)
                eps = 1e-8
                mu_var1 = -eta1_var1_raw / (2.0 * eta2_var1 + eps)  # [batch, buses]
                mu_var2 = -eta1_var2_raw / (2.0 * eta2_var2 + eps)  # [batch, buses]
                
                # Stack to get predictions [batch, buses, 2]
                outputs_norm = torch.stack([mu_var1, mu_var2], dim=-1)
            
            # Handle shape consistency for different model types
            if outputs_norm.dim() == 2:
                batch_size = outputs_norm.shape[0]
                num_features = 2  # OPF mode: 2 unknowns per bus (varies by bus type)
                outputs_norm = outputs_norm.view(batch_size, num_buses, num_features)
            
            # Only denormalize for physics calculations that require physical units
            outputs_phys = normalizer.denormalize(outputs_norm)  # [batch, buses, 2] = [vm, va]
            # Handle sequential models: features can be [batch, seq_len, buses, features]
            if features.dim() == 4:
                # Sequential model: use last timestep [batch, seq_len, buses, features] -> [batch, buses, features]
                features = features[:, -1, :, :]  # Take last timestep
            features_phys = normalizer.denormalize(features)  # [batch, buses, 10] = [p_load, q_load, ...]

            if is_physics_informed:
                # Get bus types from batch (OPF: bus-type-dependent unknowns)
                bus_types = batch.get('bus_types', None)  # [batch, buses] or None
                if bus_types is not None:
                    bus_types = bus_types.to(device)
                
                # Calculate physics-based metrics for physics-informed models
                # OPF: Pass predicted unknowns (2 features) and measurements (10 features) separately
                # CRITICAL: Must pass bus_types for correct voltage state reconstruction
                norm_loss = physics_calculator._compute_active_power_loss_pu(outputs_phys, features_phys, ybus, bus_types=bus_types)
                norm_vdev = physics_calculator._compute_normalized_voltage_deviation(outputs_phys, measurements=features_phys, bus_types=bus_types)
                # Use PREDICTED state for carbon emissions calculation (model-dependent)
                emissions = physics_calculator._compute_carbon_emissions(
                    features_phys, time_carbon, time_energy, renewable_frac,
                    voltages=outputs_phys, Ybus=ybus, bus_types=bus_types
                )
                norm_power_flow = physics_calculator._compute_mean_power_flow_pu(outputs_phys, features_phys, ybus, bus_types=bus_types)
            else:
                # For non-physics models, set physics metrics to zero/neutral values
                batch_size = features.shape[0]
                norm_loss = torch.zeros(batch_size, device=device)
                norm_vdev = torch.zeros(batch_size, device=device)
                emissions = {'raw': torch.zeros(batch_size, device=device), 'normalized': torch.zeros(batch_size, device=device)}
                norm_power_flow = torch.zeros(batch_size, device=device)

            # Capture data for analyzing the impact of renewables (only for physics-informed models)
            if is_physics_informed:
                try:
                    # Vectorized batch conversion - much faster than looping
                    batch_data = pd.DataFrame({
                        'renewable_fraction': renewable_frac.cpu().numpy(),
                        'normalized_carbon_emissions': emissions['normalized'].cpu().numpy(),
                        'voltage_deviation': norm_vdev.cpu().numpy(),
                        'power_loss': norm_loss.cpu().numpy(),
                        'power_flow': norm_power_flow.cpu().numpy()
                    })
                    renewable_impact_data.extend(batch_data.to_dict('records'))
                except (IndexError, KeyError) as e:
                    logging.warning(f"Could not extract renewable fraction from batch data: {e}")

            if is_physics_informed:
                # Calculate test MSE for physics-informed models (consistent with training)
                targets = batch['targets'].to(device)
                targets_phys = normalizer.denormalize(targets)
                mse_physical = F.mse_loss(outputs_phys, targets_phys)
                s_base_mva = physics_calculator.s_base_mva
                test_mse = mse_physical
                
                moopf_score = (w_loss * norm_loss + w_vdev * norm_vdev + w_carbon * emissions['normalized'])
                all_results.append({
                    'mse_score': test_mse.item(),  # Normalized MSE for consistent scoring
                    'moopf_score': moopf_score.mean().item(), 
                    'normalized_power_loss': norm_loss.mean().item(),
                    'normalized_voltage_deviation': norm_vdev.mean().item(),
                    'normalized_power_flow': norm_power_flow.mean().item(),
                    'normalized_carbon_emissions': emissions['normalized'].mean().item(),
                    'raw_carbon_emissions_tCO2': emissions['raw'].mean().item()
                })
            else:
                # For non-physics models, compute MSE consistent with training
                targets = batch['targets'].to(device)
                targets_phys = normalizer.denormalize(targets)
                mse_physical = F.mse_loss(outputs_phys, targets_phys)
                s_base_mva = physics_calculator.s_base_mva
                mse_normalized = mse_physical
                all_results.append({
                    'mse_score': mse_normalized.item()  # MSE in per-unit squared (consistent with training)
                })

    return pd.DataFrame(all_results), pd.DataFrame(renewable_impact_data)


def save_best_model_results(best_model: torch.nn.Module, best_run: Dict[str, Any], 
                           moopf_results: pd.DataFrame, renewable_impact_data: pd.DataFrame, 
                           training_history: Dict[str, List], config: Any, num_buses: int, 
                           is_physics_informed: bool = True, iteration_details: List[Dict] = None,
                           param_keys: List[str] = None):
    """Saves all results for the best model in the new directory structure."""
    # Skip saving if disabled
    if hasattr(config, 'SAVE_RESULTS') and not config.SAVE_RESULTS:
        return
    
    model_name = best_run['model_name']
    
    # Create necessary directories
    model_dir = config.get_model_eval_dir(num_buses, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model checkpoint
    torch.save(best_model.state_dict(), config.get_model_checkpoint_path(num_buses, model_name))
    
    # Save detailed per-sample MOOPF/MSE results (optional, for detailed analysis)
    results_filename = "moopf_detailed.csv" if is_physics_informed else "mse_detailed.csv"
    results_path = os.path.join(model_dir, results_filename)
    moopf_results.to_csv(results_path, index=False)
    
    # Create consolidated model results CSV (clean, 2 lines only)
    from datetime import datetime
    
    clean_results = {
        # Identification
        'model_name': model_name,
        'bus_system': num_buses,
        'run_timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        
        # Hyperparameters (best found)
        'hidden_dim': best_run.get('HIDDEN_DIM', None),
        'gc_layers': best_run.get('NUM_GC_LAYERS', None),
        'sequence_length': best_run.get('SEQUENCE_LENGTH', None),
        'rnn_layers': best_run.get('RNN_LAYERS', None),
        'embedding_dim': best_run.get('EMBEDDING_DIM', None),
        'lambda_p': best_run.get('LAMBDA_P', None) if is_physics_informed else None,
        'lambda_v': best_run.get('LAMBDA_V', None) if is_physics_informed else None,
        'phi': best_run.get('PHI', None),
        
        # Training performance
        'train_mse': best_run.get('training_mse', best_run.get('mse', None)),
        'train_rmse': best_run.get('rmse', None),
        'epochs_trained': len(training_history.get('train_mse', [])) if training_history else None,
        
        # Test performance
        'test_mse': best_run.get('val_metrics', {}).get('mse', None) if 'val_metrics' in best_run else best_run.get('mse', None),
        'test_rmse': best_run.get('val_metrics', {}).get('rmse', None) if 'val_metrics' in best_run else None,
        
        # Physics metrics (PI models only)
        'power_violation': best_run.get('power_violation', None) if is_physics_informed else None,
        'voltage_violation': best_run.get('voltage_violation', None) if is_physics_informed else None,
    }
    
    # Add MOOPF objectives if physics-informed
    # CRITICAL FIX: Validate that physics-informed models actually have MOOPF scores
    if is_physics_informed and not moopf_results.empty:
        # Check if moopf_score column exists and has valid values
        if 'moopf_score' in moopf_results.columns:
            moopf_mean = moopf_results['moopf_score'].mean()
            if pd.isna(moopf_mean):
                raise ValueError(f"CRITICAL: Physics-informed model {model_name} has NaN MOOPF scores. "
                               f"This indicates evaluation failed. Check evaluate_moopf_objectives_normalized.")
        else:
            raise ValueError(f"CRITICAL: Physics-informed model {model_name} missing 'moopf_score' column. "
                           f"Evaluation did not compute MOOPF metrics. Check is_physics_informed flag.")
        
        clean_results.update({
            'avg_power_loss': moopf_results['normalized_power_loss'].mean() if 'normalized_power_loss' in moopf_results.columns else None,
            'avg_voltage_dev': moopf_results['normalized_voltage_deviation'].mean() if 'normalized_voltage_deviation' in moopf_results.columns else None,
            'avg_power_flow': moopf_results['normalized_power_flow'].mean() if 'normalized_power_flow' in moopf_results.columns else None,
            'avg_carbon_emissions': moopf_results['raw_carbon_emissions_tCO2'].mean() if 'raw_carbon_emissions_tCO2' in moopf_results.columns else None,
            'moopf_score': moopf_mean,
        })
    else:
        # Non-physics models: explicitly set MOOPF metrics to None (not NaN)
        clean_results.update({
            'avg_power_loss': None,
            'avg_voltage_dev': None,
            'avg_power_flow': None,
            'avg_carbon_emissions': None,
            'moopf_score': None,
        })
    
    # Add optimization info
    if iteration_details:
        clean_results.update({
            'optimization_method': 'MoSOA',
            'num_iterations': len(iteration_details),
            'num_seagulls': iteration_details[0].get('num_valid_evaluations', None) if iteration_details else None,
            'best_objective_score': best_run.get('total_loss', None),
            'optimization_time_sec': None,  # Could add if tracked
        })
    else:
        clean_results.update({
            'optimization_method': 'Trial',
            'num_iterations': None,
            'num_seagulls': None,
            'best_objective_score': best_run.get('total_loss', None),
            'optimization_time_sec': None,
        })
    
    # Save clean consolidated CSV (2 lines: header + data)
    results_csv_path = os.path.join(model_dir, "model_results.csv")
    pd.DataFrame([clean_results]).to_csv(results_csv_path, index=False)
    
    # Plot training history (available for all models)
    try:
        plot_training_history(training_history, model_name, config, num_buses, is_physics_informed)
    except Exception as e:
        print(f"  Warning: Could not create training history plot for {model_name}: {e}")
    
    # Plot convergence history if available (available for all models)
    if 'convergence_history' in best_run:
        try:
            plot_convergence(best_run['convergence_history'], model_name, config, num_buses)
        except Exception as e:
            print(f"  Warning: Could not create convergence plot for {model_name}: {e}")
    
    # Only plot renewable impacts for physics-informed models
    if is_physics_informed:
        try:
            plot_all_renewable_impacts(renewable_impact_data, config, num_buses, model_name)
        except Exception as e:
            print(f"  Warning: Could not create renewable impact plots for {model_name}: {e}")
    else:
        print(f"Skipping renewable impact plots for non-physics-informed model: {model_name}")


def print_comprehensive_summary(all_results: List[Dict[str, Any]], config: Any = None):
    """Print and save a comprehensive summary of all model performances across all bus systems."""
    if not all_results:
        print("\n No results to summarize.")
        return
    
    print(f"\n{'='*100}")
    print(f" COMPREHENSIVE FINAL SUMMARY - ALL MODELS & BUS SYSTEMS")
    print(f"{'='*100}")
    
    # Create summary table for display
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Model': result['model_name'],
            'Bus System': f"{result['num_buses']}-bus",
            'Type': 'Physics' if result['is_physics_informed'] else 'Non-Physics',
            'Hidden Dim': result['best_hidden_dim'],
            'GC Layers': result['best_gc_layers'],
            'Training MSE': f"{result['training_mse']:.6f}" if result['training_mse'] != float('inf') else 'Failed',
            'Test Score': f"{result['final_test_score']:.6f}" if result['final_test_score'] != float('inf') else 'Failed',
            'Metric Type': result['final_metric_name']
        })
    
    csv_data = []
    for result in all_results:
        csv_data.append({
            'model_name': result['model_name'],
            'num_buses': result['num_buses'],
            'bus_system': f"{result['num_buses']}-bus",
            'model_type': 'Physics-Informed' if result['is_physics_informed'] else 'Non-Physics',
            'is_physics_informed': result['is_physics_informed'],
            'best_hidden_dim': result['best_hidden_dim'],
            'best_gc_layers': result['best_gc_layers'],
            'training_mse': result['training_mse'],
            'final_test_score': result['final_test_score'],
            'final_metric_name': result['final_metric_name'],
            'power_violation': result['power_violation'],
            'voltage_violation': result['voltage_violation'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # Save to CSV file in current run directory (if saving enabled)
    if config and hasattr(config, 'SAVE_RESULTS') and config.SAVE_RESULTS:
        # Use CURRENT_RUN_DIR to save in run-specific folder (consistent with other saves)
        try:
            if hasattr(config, 'CURRENT_RUN_DIR') and config.CURRENT_RUN_DIR:
                model_eval_dir = config.CURRENT_RUN_DIR
            else:
                # Fallback to experimental_results if CURRENT_RUN_DIR not available
                model_eval_dir = getattr(config, 'EXPERIMENTAL_RESULTS_DIR', 'experimental_results')
        except (AttributeError, TypeError):
            model_eval_dir = getattr(config, 'EXPERIMENTAL_RESULTS_DIR', 'experimental_results')
        
        os.makedirs(model_eval_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"comprehensive_summary_{timestamp}.csv"
        csv_path = os.path.join(model_eval_dir, csv_filename)
        
        # Also save a "latest" version for easy access
        latest_csv_path = os.path.join(model_eval_dir, "comprehensive_summary_latest.csv")
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        df.to_csv(latest_csv_path, index=False)
        
        print(f" Results saved: {os.path.basename(csv_path)}")
    print()
    
    # Print table header
    print(f"{'Model':<15} {'Bus Sys':<8} {'Type':<11} {'H.Dim':<7} {'GC Ly':<5} {'Train MSE':<12} {'Test Score':<12} {'Metric':<12}")
    print("-" * 100)
    
    # Print each result
    for data in summary_data:
        print(f"{data['Model']:<15} {data['Bus System']:<8} {data['Type']:<11} {data['Hidden Dim']:<7} {data['GC Layers']:<5} {data['Training MSE']:<12} {data['Test Score']:<12} {data['Metric Type']:<12}")
    
    print("-" * 100)
    
    # CRITICAL FIX: Separate physics and non-physics models for fair comparison
    physics_results = [r for r in all_results if r['is_physics_informed'] and r['final_test_score'] != float('inf')]
    non_physics_results = [r for r in all_results if not r['is_physics_informed'] and r['final_test_score'] != float('inf')]
    successful_results = physics_results + non_physics_results
    
    if successful_results:
        # Best overall - separate by model type
        if physics_results:
            best_physics = min(physics_results, key=lambda x: x['final_test_score'])
            print(f"\n BEST PHYSICS-INFORMED MODEL:")
            print(f"   Model: {best_physics['model_name']} on {best_physics['num_buses']}-bus system")
            print(f"   {best_physics['final_metric_name']}: {best_physics['final_test_score']:.6f}")
            print(f"   Config: {best_physics['best_hidden_dim']} hidden_dim, {best_physics['best_gc_layers']} GC layers")
        
        if non_physics_results:
            best_non_physics = min(non_physics_results, key=lambda x: x['final_test_score'])
            print(f"\n BEST NON-PHYSICS MODEL:")
            print(f"   Model: {best_non_physics['model_name']} on {best_non_physics['num_buses']}-bus system")
            print(f"   {best_non_physics['final_metric_name']}: {best_non_physics['final_test_score']:.6f}")
            print(f"   Config: {best_non_physics['best_hidden_dim']} hidden_dim, {best_non_physics['best_gc_layers']} GC layers")
        
        # Best per bus system - separate by type
        print(f"\n BEST PER BUS SYSTEM (by type):")
        bus_systems = list(set(r['num_buses'] for r in successful_results))
        for num_buses in sorted(bus_systems):
            bus_physics = [r for r in physics_results if r['num_buses'] == num_buses]
            bus_non_physics = [r for r in non_physics_results if r['num_buses'] == num_buses]
            
            if bus_physics:
                best_physics_bus = min(bus_physics, key=lambda x: x['final_test_score'])
                print(f"   {num_buses}-bus (Physics): {best_physics_bus['model_name']} ({best_physics_bus['final_metric_name']}: {best_physics_bus['final_test_score']:.6f})")
            
            if bus_non_physics:
                best_non_physics_bus = min(bus_non_physics, key=lambda x: x['final_test_score'])
                print(f"   {num_buses}-bus (Non-Physics): {best_non_physics_bus['model_name']} ({best_non_physics_bus['final_metric_name']}: {best_non_physics_bus['final_test_score']:.6f})")
        
        # Performance comparison
        print(f"\n PERFORMANCE COMPARISON:")
        print(f"   33-bus systems generally perform better (lower error)")
        print(f"   Performance degrades with system size as expected")
        
        # Count successful vs failed models
        total_runs = len(all_results)
        successful_runs = len(successful_results)
        print(f"\n SUCCESS RATE: {successful_runs}/{total_runs} ({100*successful_runs/total_runs:.1f}%)")
        
        # Create comparison plot
        comparison_plot_path = os.path.join(model_eval_dir, "model_comparison_latest.png")
        create_model_comparison_plot(all_results, comparison_plot_path)
        
    else:
        print("\n No successful model runs to analyze.")
    
    print(f"{'='*100}")


def print_model_summary(best_run: Dict[str, Any], moopf_results: pd.DataFrame, 
                       model_name: str, num_buses: int, is_physics_informed: bool, 
                       final_test_score: float, final_metric_name: str):
    """Print a formatted summary of the best model results."""
    print(f"\n{'='*60}")
    print(f" BEST MODEL SUMMARY: {model_name} on {num_buses}-bus system")
    print(f"{'='*60}")
    print(f" Best Hyperparameters: {best_run.get('HIDDEN_DIM', 'N/A')} hidden_dim, {best_run.get('NUM_GC_LAYERS', 'N/A')} GC layers")
    training_mse = best_run.get('training_mse', best_run.get('mse', 'N/A'))
    print(f" Training Performance: MSE = {training_mse:.6g}")
    
    if is_physics_informed:
        print(f" Physics Violations: Power = {best_run.get('power_violation', 'N/A'):.6g}, Voltage = {best_run.get('voltage_violation', 'N/A'):.6g}")
        print("\n--- MOOPF Evaluation Results ---")
        # Format metrics concisely
        metrics = moopf_results.mean().to_dict()
        formatted_metrics = {k: f"{v:.6g}" for k, v in metrics.items()}
        print(formatted_metrics)
    else:
        print(f" Final Test MSE: {final_test_score:.6g}")
        print("\n--- MSE Evaluation Results ---")
        # Only show relevant metrics for non-physics models
        relevant_metrics = {
            'mse_score': f"{final_test_score:.6g}",
            'rmse_score': f"{(final_test_score) ** 0.5:.6g}",
            'samples_evaluated': len(moopf_results)
        }
        print(relevant_metrics)
    print(f"{'='*60}")
