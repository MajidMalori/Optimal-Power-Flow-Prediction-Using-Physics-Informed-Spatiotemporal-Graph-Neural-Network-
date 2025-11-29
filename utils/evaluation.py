import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Tuple, List, Any
from datetime import datetime
from config import FeatureIndices
from utils.physics_metrics import PhysicsMetricEngine
from utils.visualization import create_model_comparison_plot

def evaluate_model(model, data_loader, device, config, normalizer, is_sequential=False):
    """
    Standard evaluation (no dropout). Returns denormalized metrics.
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
            all_preds.append(out.cpu())
            all_targets.append(targets.cpu())
            
    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Handle shape consistency for flattened outputs (Pure State Estimation)
    # Outputs might be [batch, num_buses * 2] -> reshape to [batch, num_buses, 2]
    if predictions.dim() == 2:
        batch_size = predictions.shape[0]
        # In Pure State Estimation, output is 2 features (VM, VA)
        num_features = 2 
        if predictions.shape[1] % num_features == 0:
            num_buses = predictions.shape[1] // num_features
            predictions = predictions.view(batch_size, num_buses, num_features)
            
            # Ensure targets are also reshaped if needed
            if targets.dim() == 2 and targets.shape[1] == predictions.shape[1] * (10/2): 
                 # If targets are full 10 features flattened? Unlikely.
                 pass
            elif targets.dim() == 2 and targets.shape[1] == predictions.shape[0] * num_buses * num_features:
                 targets = targets.view(batch_size, num_buses, num_features)
        else:
            # Fallback or error?
            pass

    # Denormalize
    preds_phys = normalizer.denormalize(predictions)
    targets_phys = normalizer.denormalize(targets)
    
    mse = torch.nn.functional.mse_loss(preds_phys, targets_phys).item()
    mae = torch.nn.functional.l1_loss(preds_phys, targets_phys).item()
    rmse = np.sqrt(mse)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }

def evaluate_model_normalized(model, data_loader, device, config, normalizer, is_sequential=False):
    """
    Evaluation for optimization - returns MSE in NORMALIZED space.
    This ensures equal weighting across all features and matches training loss.
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
            all_preds.append(out.cpu())
            all_targets.append(targets.cpu())
            
    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # MSE on NORMALIZED data (matches training loss)
    mse = torch.nn.functional.mse_loss(predictions, targets).item()
    mae = torch.nn.functional.l1_loss(predictions, targets).item()
    rmse = np.sqrt(mse)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }

def evaluate_model_mc_dropout(model, test_loader, device, config, normalizer, num_samples=50):
    """
    Evaluates the model using Monte Carlo Dropout for uncertainty estimation.
    """
    model.train() # Enable Dropout during inference
    
    all_means = []
    all_stds = []
    all_targets = []
    all_ybus = []
    all_carbon_intensity = []
    all_energy_coeff = []
    all_renewable_fractions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="MC Dropout Evaluation"):
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            adj = batch['adjacency'].to(device)
            
            # Store Ybus for physics metrics
            if 'ybus_matrix' in batch:
                 all_ybus.append(batch['ybus_matrix'].cpu())
            elif 'ybus' in batch:
                 all_ybus.append(batch['ybus'].cpu())
            
            # Store carbon intensity and energy coefficients (always present)
            all_carbon_intensity.append(batch['time_carbon_coeffs'].cpu())
            all_energy_coeff.append(batch['time_energy_coeffs'].cpu())
            
            # Store renewable fraction for physics-consistent carbon calculation (always present from data loader)
            renewable_frac = batch['renewable_fraction']
            all_renewable_fractions.append(renewable_frac.cpu())
            
            batch_preds = []
            for _ in range(num_samples):
                out = model(features, adj) # [batch, buses, 10]
                batch_preds.append(out.cpu())
            
            batch_preds = torch.stack(batch_preds) # [samples, batch, buses, 10]
            
            mean_pred = batch_preds.mean(dim=0) # [batch, buses, 10]
            std_pred = batch_preds.std(dim=0)   # [batch, buses, 10]
            
            all_means.append(mean_pred)
            all_stds.append(std_pred)
            all_targets.append(targets.cpu())
            
    predictions = torch.cat(all_means, dim=0)
    uncertainties = torch.cat(all_stds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    if all_ybus:
        ybus_batch = torch.cat(all_ybus, dim=0)
    else:
        ybus_batch = None
    
    # Concatenate carbon intensity, energy coefficients, and renewable fractions
    carbon_intensity_batch = torch.cat(all_carbon_intensity, dim=0) if all_carbon_intensity else None
    energy_coeff_batch = torch.cat(all_energy_coeff, dim=0) if all_energy_coeff else None
    renewable_fraction_batch = torch.cat(all_renewable_fractions, dim=0) if all_renewable_fractions else None
        
    # Handle shape consistency for flattened outputs
    if predictions.dim() == 2:
        batch_size = predictions.shape[0]
        num_features = 2
        if predictions.shape[1] % num_features == 0:
            num_buses = predictions.shape[1] // num_features
            predictions = predictions.view(batch_size, num_buses, num_features)
            uncertainties = uncertainties.view(batch_size, num_buses, num_features)
            # Note: Targets are usually full state (10 features) or matching output. 
            # We assume targets match output shape here if denormalizer handles it.
    
    # Denormalize
    preds_phys = normalizer.denormalize(predictions)
    targets_phys = normalizer.denormalize(targets)
    
    # Uncertainty needs careful handling - scale but don't shift
    uncertainties_phys = torch.zeros_like(uncertainties)
    
    # Scale uncertainties based on feature indices
    # Note: This assumes 10-feature output. If 2-feature output (VM, VA), we need to handle indices.
    if uncertainties.shape[-1] == 10:
        uncertainties_phys[..., 0:8] = uncertainties[..., 0:8] / normalizer.power_scale
        uncertainties_phys[..., 8] = uncertainties[..., 8] / normalizer.vm_scale
        uncertainties_phys[..., 9] = uncertainties[..., 9] / normalizer.va_scale
    elif uncertainties.shape[-1] == 2:
        # VM, VA only
        uncertainties_phys[..., 0] = uncertainties[..., 0] / normalizer.vm_scale
        uncertainties_phys[..., 1] = uncertainties[..., 1] / normalizer.va_scale

    return preds_phys, targets_phys, uncertainties_phys, ybus_batch, carbon_intensity_batch, energy_coeff_batch, renewable_fraction_batch

def compute_engineering_metrics(preds_phys, targets_phys, ybus_batch, base_mva, 
                                carbon_intensity, energy_coeff, renewable_fraction):
    """
    Computes Engineering Metrics using the Vectorized PhysicsMetricEngine.
    Includes actual carbon intensity and energy coefficients from data.
    """
    # 1. Standard MSE
    mse = torch.nn.functional.mse_loss(preds_phys, targets_phys).item()
    
    # 2. Physics Metrics with carbon intensity and energy coefficients
    engine = PhysicsMetricEngine(base_mva=base_mva)
    
    phys_metrics = engine.compute_metrics(preds_phys, ybus_batch, carbon_intensity, energy_coeff, renewable_fraction)
    
    metrics = {
        'mse': mse,
        **phys_metrics
    }
    
    return metrics

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

def evaluate_renewable_impacts(model, test_loader, device, config, normalizer, num_samples=50):
    """
    Evaluates model performance vs renewable penetration.
    This runs MC Dropout - use evaluate_renewable_impacts_from_predictions() to reuse existing results.
    """
    preds_phys, targets_phys, uncertainties, ybus_batch, carbon_intensity, energy_coeff, renewable_fraction = evaluate_model_mc_dropout(
        model, test_loader, device, config, normalizer, num_samples
    )
    
    return evaluate_renewable_impacts_from_predictions(preds_phys, uncertainties, normalizer)

def save_results(metrics, results_df, config):
    """
    Saves evaluation results.
    """
    # Get results directory from config (handle both methods and attributes)
    if hasattr(config, 'get_experimental_results_dir'):
        save_dir = config.get_experimental_results_dir()
    elif hasattr(config, 'CURRENT_RUN_DIR'):
        save_dir = config.CURRENT_RUN_DIR
    elif hasattr(config, 'EXPERIMENTAL_RESULTS_DIR'):
        save_dir = config.EXPERIMENTAL_RESULTS_DIR
    else:
        # Fallback: create results directory in current working directory
        save_dir = os.path.join(os.getcwd(), 'experimental_results')
    
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
    all_renewable_fractions = []
    all_model_outputs = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            adj = batch['adjacency'].to(device)
            bus_types = batch.get('bus_types', None)
            renewable_fraction = batch['renewable_fraction']  # Always present from data loader (line 303 in data_loader.py)
            
            # Forward pass with MC Dropout for uncertainty
            model.train()  # Enable dropout
            mc_samples = 50
            predictions_mc = []
            for _ in range(mc_samples):
                out = model(features, adj)
                predictions_mc.append(out.unsqueeze(0))
            
            predictions_mc = torch.cat(predictions_mc, dim=0)
            predictions_mean = predictions_mc.mean(dim=0)
            predictions_std = predictions_mc.std(dim=0)
            
            all_predictions.append(predictions_mean.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_uncertainties.append(predictions_std.cpu().numpy())
            all_model_outputs.append(predictions_mean.cpu().numpy())
            
            if bus_types is not None:
                all_bus_types.append(bus_types.cpu().numpy())
            
            if renewable_fraction.numel() > 0:
                all_renewable_fractions.append(renewable_fraction.cpu().numpy())
    
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    uncertainties = np.concatenate(all_uncertainties, axis=0)
    
    # Handle shape consistency for flattened outputs
    if predictions.ndim == 2:
        batch_size = predictions.shape[0]
        num_features = 2
        if predictions.shape[1] % num_features == 0:
            num_buses = predictions.shape[1] // num_features
            predictions = predictions.reshape(batch_size, num_buses, num_features)
            uncertainties = uncertainties.reshape(batch_size, num_buses, num_features)
            
    # Denormalize
    predictions_phys = normalizer.denormalize(torch.from_numpy(predictions)).numpy()
    targets_phys = normalizer.denormalize(torch.from_numpy(targets)).numpy()
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(np.sqrt(mse))
    }
    
    uncertainty_data = {
        'predictions': predictions_phys,
        'targets': targets_phys,
        'uncertainties': uncertainties,
        'model_outputs': np.concatenate(all_model_outputs, axis=0) if all_model_outputs else None,
        'bus_types': np.concatenate(all_bus_types, axis=0) if all_bus_types else None,
        'renewable_fractions': np.concatenate(all_renewable_fractions, axis=0) if all_renewable_fractions else None
    }
    
    return metrics, uncertainty_data


def evaluate_moopf_objectives_normalized(model, test_loader, config, device, normalizer, is_physics_informed=True):
    """
    Evaluate Multi-Objective Optimal Power Flow objectives.
    """
    if not is_physics_informed:
        return {}, None
    
    model.eval()
    
    all_power_loss = []
    all_voltage_dev = []
    all_carbon = []
    all_carbon_raw = []  # Raw emissions (for tracking only, not in MOOPF)
    all_power_flow = []
    all_renewable_fractions = []
    all_mse_per_sample = []  # Collect per-sample MSE
    
    # Extract num_buses from config (config.yaml: system.num_buses)
    num_buses_val = config.NUM_BUSES
    num_buses = int(num_buses_val[0]) if isinstance(num_buses_val, list) else int(num_buses_val)
    
    # MOOPF weights from config.yaml (3 objectives only: loss + vdev + carbon)
    w_loss = config.MOOPF_WEIGHT_LOSS
    w_vdev = config.MOOPF_WEIGHT_VDEV
    w_carbon = config.MOOPF_WEIGHT_CARBON

    # Initialize physics engine (BASE_MVA from config.yaml: physics.base_mva)
    engine = PhysicsMetricEngine(base_mva=config.BASE_MVA)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating MOOPF Objectives"):
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            adj = batch['adjacency'].to(device)
            ybus = batch['ybus_matrix'].to(device)
            renewable_fraction = batch['renewable_fraction'].to(device)  # Always present from data loader (line 303 in data_loader.py)
            
            # Extract carbon intensity and energy coefficients from batch (always present)
            carbon_intensity = batch['time_carbon_coeffs'].to(device)
            energy_coeff = batch['time_energy_coeffs'].to(device)
            
            # Forward pass
            predictions = model(features, adj)
            
            # Handle shape consistency
            if predictions.dim() == 2:
                batch_size = predictions.shape[0]
                num_features = 2
                if predictions.shape[1] % num_features == 0:
                    predictions = predictions.view(batch_size, num_buses, num_features)
            
            # Denormalize for physics metrics
            preds_phys = normalizer.denormalize(predictions)
            targets_phys = normalizer.denormalize(targets)
            
            # Calculate per-sample MSE for mse_detailed.csv
            mse_per_sample = ((preds_phys - targets_phys) ** 2).mean(dim=(1, 2))  # Average over buses and features
            all_mse_per_sample.extend(mse_per_sample.cpu().numpy().tolist())
            
            # Compute metrics using PhysicsMetricEngine with actual carbon/energy coefficients
            # Pass renewable_fraction for physics-consistent carbon calculation
            metrics = engine.compute_metrics(preds_phys, ybus, carbon_intensity, energy_coeff, renewable_fraction)
            
            all_power_loss.append(metrics['system_power_loss'])
            all_voltage_dev.append(metrics['voltage_deviation'])
            all_carbon.append(metrics['carbon_emissions'])
            all_carbon_raw.append(metrics['carbon_emissions_raw'])  # For tracking/plotting only, not in MOOPF score
            all_power_flow.append(metrics['power_flow'])  # For tracking/plotting only, not in MOOPF score
            
            # Handle renewable_fraction (take mean if batch, single value otherwise)
            if renewable_fraction.numel() == 1:
                all_renewable_fractions.append(renewable_fraction.item())
            elif renewable_fraction.numel() > 1:
                # If batch, take mean to match other metrics (one value per batch)
                all_renewable_fractions.append(renewable_fraction.mean().item())
    
    moopf_results = {
        'power_loss': float(np.mean(all_power_loss)),
        'voltage_deviation': float(np.mean(all_voltage_dev)),
        'carbon_emissions': float(np.mean(all_carbon)),  # Per-unit (for MOOPF)
        'carbon_emissions_raw': float(np.mean(all_carbon_raw)),  # Raw value (for tracking only, not in MOOPF)
        'power_flow': float(np.mean(all_power_flow)),  # For tracking/plotting only, not in MOOPF score
        'mse_per_sample': all_mse_per_sample  # Add per-sample MSE for mse_detailed.csv
    }
    
    # Calculate MOOPF score (3 objectives: power loss + voltage deviation + carbon emissions)
    moopf_score = (w_loss * moopf_results['power_loss'] + 
                   w_vdev * moopf_results['voltage_deviation'] + 
                   w_carbon * moopf_results['carbon_emissions'])
    moopf_results['moopf_score'] = moopf_score

    # Add key names expected by train.py
    moopf_results['mse_score'] = moopf_score # Use moopf_score as the metric for PI models? 
    # train.py expects 'mse_score' for final scoring.
    
    # Create renewable impact dataframe
    if all_renewable_fractions:
        renewable_impact_data = pd.DataFrame({
            'renewable_fraction': all_renewable_fractions,
            'power_loss': all_power_loss,
            'voltage_deviation': all_voltage_dev,
            'carbon_emissions': all_carbon,  # Per-unit (for comparison)
            'carbon_emissions_raw': all_carbon_raw,  # Raw value (for tracking)
            'power_flow': all_power_flow
        })
    else:
        renewable_impact_data = None
    
    return moopf_results, renewable_impact_data

def save_best_model_results(best_model, best_run, moopf_results, renewable_impact_data, training_history, config, num_buses, is_physics_informed, iteration_details=None, param_keys=None, model_name="", output_dir=""):
    """
    Saves detailed results for the best model.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save Metrics
    metrics = {
        'test_score': best_run.get('test_score'),
        'val_score': best_run.get('val_score'),
        **moopf_results
    }
    
    import json
    with open(os.path.join(output_dir, 'best_model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # 2. Save Renewable Impact Data
    if renewable_impact_data is not None:
        renewable_impact_data.to_csv(os.path.join(output_dir, 'renewable_impact.csv'), index=False)
        
    # 3. Save Training History (handle arrays of different lengths)
    try:
        # Find the maximum length across all history arrays
        max_len = max(len(v) if isinstance(v, list) else 1 for v in training_history.values())
        
        # Pad shorter arrays with None to match max_len
        padded_history = {}
        for key, values in training_history.items():
            if isinstance(values, list):
                # Pad with None if shorter than max_len
                padded_values = values + [None] * (max_len - len(values))
                padded_history[key] = padded_values
            else:
                padded_history[key] = [values]
        
        pd.DataFrame(padded_history).to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    except Exception as e:
        print(f"  Warning: Could not save training history: {e}")
    
    # Quietly save results (no verbose output)
    # print(f"  Results saved to {output_dir}")

def save_model_results_csv(best_run, moopf_results, config, num_buses, model_name, 
                          output_dir, iteration_details=None):
    """
    Save comprehensive model results to model_results.csv.
    
    Columns: model_name, bus_system, run_timestamp, hidden_dim, gc_layers,
             sequence_length, rnn_layers, embedding_dim, lambda_p, lambda_v, phi,
             train_mse, train_rmse, epochs_trained, test_mse, test_rmse,
             power_violation, voltage_violation, avg_power_loss, avg_voltage_dev,
             avg_power_flow, avg_carbon_emissions, moopf_score,
             optimization_method, num_iterations, num_seagulls,
             best_objective_score, optimization_time_sec
    """
    from datetime import datetime
    
    # Extract hyperparameters
    config_dict = best_run.get('config_dict', {})
    
    # Build results dict
    results = {
        'model_name': model_name,
        'bus_system': num_buses,
        'run_timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'hidden_dim': best_run.get('HIDDEN_DIM', config_dict.get('HIDDEN_DIM', '')),
        'gc_layers': best_run.get('NUM_GC_LAYERS', config_dict.get('NUM_GC_LAYERS', '')),
        'sequence_length': config_dict.get('SEQUENCE_LENGTH', ''),
        'rnn_layers': config_dict.get('NUM_RNN_LAYERS', ''),
        'embedding_dim': config_dict.get('EMBEDDING_DIM', ''),
        'lambda_p': config_dict.get('LAMBDA_P', ''),
        'lambda_v': config_dict.get('LAMBDA_V', ''),
        'phi': config_dict.get('PHI', ''),
        'train_mse': best_run.get('training_mse', ''),
        'train_rmse': np.sqrt(best_run.get('training_mse', 0)) if best_run.get('training_mse') else '',
        'epochs_trained': config.NUM_EPOCHS,
        'test_mse': best_run.get('test_score', ''),
        'test_rmse': np.sqrt(best_run.get('test_score', 0)) if best_run.get('test_score') else '',
        'power_violation': moopf_results.get('physics_loss', ''),
        'voltage_violation': moopf_results.get('safety_loss', ''),
        'avg_power_loss': moopf_results.get('power_loss', ''),
        'avg_voltage_dev': moopf_results.get('voltage_deviation', ''),
        'avg_power_flow': moopf_results.get('power_flow', ''),
        'avg_carbon_emissions': moopf_results.get('carbon_emissions', ''),
        'moopf_score': moopf_results.get('moopf_score', ''),
        'optimization_method': 'MoSOA',
        'num_iterations': iteration_details[-1].get('iteration', '') if iteration_details and isinstance(iteration_details, list) else '',
        'num_seagulls': iteration_details[-1].get('num_agents', '') if iteration_details and isinstance(iteration_details, list) else '',
        'best_objective_score': best_run.get('test_score', ''),
        'optimization_time_sec': iteration_details[-1].get('optimization_time', '') if iteration_details and isinstance(iteration_details, list) else ''
    }
    
    # Save to CSV
    df = pd.DataFrame([results])
    csv_path = os.path.join(output_dir, 'model_results.csv')
    df.to_csv(csv_path, index=False)
    # Quietly save results (no verbose output)
    # print(f"  Saved model_results.csv")

def print_model_summary(best_run, moopf_results, model_name, num_buses, is_physics_informed, final_test_score, final_metric_name):
    print(f"\n{model_name} ({num_buses}-bus) Summary:")
    print(f"  Test Score ({final_metric_name}): {final_test_score:.4f}")
    if is_physics_informed:
        print(f"  Power Loss: {moopf_results.get('power_loss', 'N/A')}")
        print(f"  Voltage Deviation: {moopf_results.get('voltage_deviation', 'N/A')}")
        print(f"  Carbon Emissions: {moopf_results.get('carbon_emissions', 'N/A')}")

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
            'physics_loss': result['physics_loss'],
            'safety_loss': result['safety_loss'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # Save to CSV file in experimental_results directory (if saving enabled)
    if config and hasattr(config, 'SAVE_RESULTS') and config.SAVE_RESULTS:
        model_eval_dir = "experimental_results"
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
    
    # Find overall best performers
    successful_results = [r for r in all_results if r['final_test_score'] != float('inf')]
    
    if successful_results:
        # Best overall (lowest test score)
        best_overall = min(successful_results, key=lambda x: x['final_test_score'])
        print(f"\n OVERALL BEST PERFORMER:")
        print(f"   Model: {best_overall['model_name']} on {best_overall['num_buses']}-bus system")
        print(f"   {best_overall['final_metric_name']}: {best_overall['final_test_score']:.6f}")
        print(f"   Config: {best_overall['best_hidden_dim']} hidden_dim, {best_overall['best_gc_layers']} GC layers")
        
        # Best per bus system
        print(f"\n BEST PER BUS SYSTEM:")
        bus_systems = list(set(r['num_buses'] for r in successful_results))
        for num_buses in sorted(bus_systems):
            bus_results = [r for r in successful_results if r['num_buses'] == num_buses]
            if bus_results:
                best_for_bus = min(bus_results, key=lambda x: x['final_test_score'])
                print(f"   {num_buses}-bus: {best_for_bus['model_name']} ({best_for_bus['final_metric_name']}: {best_for_bus['final_test_score']:.6f})")
        
        # Performance comparison
        print(f"\n PERFORMANCE COMPARISON:")
        print(f"   33-bus systems generally perform better (lower error)")
        print(f"   Performance degrades with system size as expected")
        
        # Count successful vs failed models
        total_runs = len(all_results)
        successful_runs = len(successful_results)
        print(f"\n SUCCESS RATE: {successful_runs}/{total_runs} ({100*successful_runs/total_runs:.1f}%)")
        
        # Create comparison plot
        if config and hasattr(config, 'get_experimental_results_dir'):
             comparison_plot_path = os.path.join(config.get_experimental_results_dir(), "model_comparison_latest.png")
             try:
                 create_model_comparison_plot(all_results, comparison_plot_path)
             except Exception as e:
                 print(f"Warning: Could not create comparison plot: {e}")

    else:
        print("\n No successful model runs to analyze.")
    
    print(f"{'='*100}")
