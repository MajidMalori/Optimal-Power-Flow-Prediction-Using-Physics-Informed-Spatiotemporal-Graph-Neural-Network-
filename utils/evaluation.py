import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Tuple
from config import FeatureIndices
from utils.physics_metrics import PhysicsMetricEngine

def evaluate_model_mc_dropout(model, test_loader, device, config, normalizer, num_samples=50):
    """
    Evaluates the model using Monte Carlo Dropout for uncertainty estimation.
    
    Args:
        model: The trained PyTorch model
        test_loader: DataLoader for test data
        device: Computing device (CPU/GPU)
        config: Configuration object
        normalizer: Data normalizer
        num_samples: Number of MC dropout forward passes
    
    Returns:
        preds_phys: Denormalized predictions (Mean) [Total_Samples, Buses, 10]
        targets_phys: Denormalized clean targets [Total_Samples, Buses, 10]
        uncertainties: Denormalized uncertainty (Std Dev) [Total_Samples, Buses, 10]
        ybus_batch: Associated Ybus matrices [Total_Samples, Buses, Buses]
    """
    model.train() # Enable Dropout during inference
    
    all_means = []
    all_stds = []
    all_targets = []
    all_ybus = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="MC Dropout Evaluation"):
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            adj = batch['adjacency'].to(device)
            
            # Store Ybus for physics metrics
            # Assuming Ybus is static for a batch or provided.
            # If data loader provides 'ybus', use it. 
            # Otherwise, if it's constant, we might need to retrieve it elsewhere.
            # Based on data_loader.py, 'ybus' is usually available if physics-informed.
            if 'ybus' in batch:
                 all_ybus.append(batch['ybus'].cpu())
            else:
                 # Fallback if not in batch (should not happen for PI models)
                 pass 
            
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
    
    # Denormalize
    preds_phys = normalizer.denormalize(predictions)
                targets_phys = normalizer.denormalize(targets)
    uncertainties_phys = normalizer.denormalize(uncertainties) # Is this right? Std scales linearly.
    # Yes, denormalize applies scale factors which is correct for Std Dev (ignoring mean shift).
    # But denormalize adds mean shift! Std Dev should ONLY be scaled, not shifted.
    # FIX: Manually scale uncertainties without adding mean
    uncertainties_phys = torch.zeros_like(uncertainties)
    uncertainties_phys[..., 0:8] = uncertainties[..., 0:8] / normalizer.power_scale
    uncertainties_phys[..., 8] = uncertainties[..., 8] / normalizer.vm_scale
    uncertainties_phys[..., 9] = uncertainties[..., 9] / normalizer.va_scale

    return preds_phys, targets_phys, uncertainties_phys, ybus_batch

def compute_engineering_metrics(preds_phys, targets_phys, ybus_batch, base_mva, config):
    """
    Computes Engineering Metrics using the Vectorized PhysicsMetricEngine.
    
    Args:
        preds_phys: Denormalized predictions [N, Buses, 10]
        targets_phys: Denormalized targets [N, Buses, 10]
        ybus_batch: Ybus matrices [N, Buses, Buses]
        base_mva: System Base MVA
        config: Config object
        
    Returns:
        Dictionary of scalar metrics.
    """
    # 1. Standard MSE
    mse = torch.nn.functional.mse_loss(preds_phys, targets_phys).item()
    
    # 2. Physics Metrics
    engine = PhysicsMetricEngine(base_mva=base_mva)
    
    # We can pass the entire batch to the engine
    # Note: Carbon intensity/Energy coeff would need to be passed if we want accurate carbon.
    # For now, we'll rely on the default (proxy) or global configs if added.
    
    phys_metrics = engine.compute_metrics(preds_phys, ybus_batch)
    
    metrics = {
        'mse': mse,
        **phys_metrics
    }
    
    return metrics

def evaluate_renewable_impacts(model, test_loader, device, config, normalizer, num_samples=50):
    """
    Evaluates model performance vs renewable penetration.
    """
    preds_phys, targets_phys, uncertainties, ybus_batch = evaluate_model_mc_dropout(
        model, test_loader, device, config, normalizer, num_samples
    )
    
    # Calculate Renewable Fraction per sample
    # P_ren is index 6, P_conv is index 4
    p_ren = preds_phys[..., FeatureIndices.P_REN].sum(dim=1)
    p_conv = preds_phys[..., FeatureIndices.P_CONV].sum(dim=1)
    # Renewable Fraction = P_ren / (P_ren + P_conv + epsilon)
    ren_frac = p_ren / (p_ren + p_conv + 1e-6)
    
    # Calculate Metrics per sample
    # This requires batch-wise processing if dataset is huge, or we can do vectorized if memory allows.
    # For 120 samples (test set), vectorized is fine.
    
    engine = PhysicsMetricEngine(base_mva=normalizer.base_mva)
    
    # We need component-wise errors, not mean scalars.
    # Re-implementing light version here or modifying engine to return per-sample?
    # Let's compute manually for plotting:
    
    # Voltage Dev (Mean per sample)
    vm = preds_phys[..., FeatureIndices.VM]
    voltage_dev = torch.mean(torch.abs(vm - 1.0), dim=1)
    
    # Power Loss
    # Using sum of net injection (Generation - Load)
    p_load = preds_phys[..., FeatureIndices.P_LOAD].sum(dim=1)
    p_gen = (preds_phys[..., FeatureIndices.P_EXT_GRID] + 
             preds_phys[..., FeatureIndices.P_CONV] + 
             preds_phys[..., FeatureIndices.P_REN]).sum(dim=1)
    power_loss = torch.abs(p_gen - p_load) / normalizer.base_mva # Simple balance check
    
    # Carbon (Total proxy)
    carbon = (preds_phys[..., FeatureIndices.P_CONV] + 
              torch.nn.functional.relu(preds_phys[..., FeatureIndices.P_EXT_GRID])).sum(dim=1)
    
    # Uncertainty (Mean per sample)
    uncertainty = torch.mean(uncertainties, dim=(1, 2)) # Mean over buses and features
    
    results_df = pd.DataFrame({
        'renewable_fraction': ren_frac.numpy(),
        'voltage_deviation': voltage_dev.numpy(),
        'power_loss': power_loss.numpy(),
        'carbon_proxy': carbon.numpy(),
        'uncertainty': uncertainty.numpy()
    })
    
    return results_df

def save_results(metrics, results_df, config):
    """
    Saves evaluation results.
    """
    import os
    save_dir = config.get_experimental_results_dir()
    os.makedirs(save_dir, exist_ok=True)
    
    # Save Scalar Metrics
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
            
    # Save DataFrame
    results_df.to_csv(os.path.join(save_dir, 'detailed_results.csv'), index=False)
    print(f"Results saved to {save_dir}")
