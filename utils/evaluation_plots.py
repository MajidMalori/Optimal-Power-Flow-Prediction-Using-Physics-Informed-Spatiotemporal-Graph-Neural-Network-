"""
Critical Evaluation Plots for Model Diagnostics

Provides essential visualizations for diagnosing regression models:
1. Predicted vs. Actual Scatter Plots (by bus type)
2. Error Distribution Histograms (by bus type)
3. Calibration Plot (Reliability Diagram) for heteroscedastic models
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Any
import warnings
from scipy.stats import linregress, norm
from config import FeatureIndices, ModelOutputIndices

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='matplotlib')


def plot_predicted_vs_actual(predictions: np.ndarray, targets: np.ndarray, 
                             bus_types: np.ndarray, case_name: str, 
                             output_dir: str, model_name: str = ""):
    """
    Generate predicted vs. actual scatter plots, separated by bus type.
    
    For OPF mode:
    - PQ buses: [V, θ] (voltage magnitude, angle)
    - PV buses: [Q, θ] (reactive power, angle)
    - Slack buses: [P, Q] (active power, reactive power)
    
    Args:
        predictions: [n_samples, n_buses, 2] - predicted unknowns
        targets: [n_samples, n_buses, 2] - true unknowns
        bus_types: [n_samples, n_buses] - bus type codes [0=PQ, 1=PV, 2=Slack]
        case_name: Name of the test case
        output_dir: Directory to save plots
        model_name: Optional model name for title
    """
    if bus_types is None:
        raise ValueError("bus_types is required for OPF mode predicted vs actual plots")
    
    # Create 2x3 grid: 2 rows (var1, var2) x 3 columns (PQ, PV, Slack)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Predicted vs. Actual - {case_name.upper()}' + (f' - {model_name}' if model_name else ''), 
                 fontsize=16, fontweight='bold')
    
    # Variable names by bus type
    var_names = {
        0: {'var1': 'Voltage (p.u.)', 'var2': 'Angle (rad)'},      # PQ
        1: {'var1': 'Reactive Power (p.u.)', 'var2': 'Angle (rad)'},  # PV
        2: {'var1': 'Active Power (p.u.)', 'var2': 'Reactive Power (p.u.)'}  # Slack
    }
    
    for bus_type_code, bus_type_name in [(0, 'PQ'), (1, 'PV'), (2, 'Slack')]:
        col = bus_type_code
        
        # Create mask for this bus type
        bus_type_mask = (bus_types == bus_type_code)  # [n_samples, n_buses]
        
        if not np.any(bus_type_mask):
            # No buses of this type
            axes[0, col].text(0.5, 0.5, f'No {bus_type_name} buses', 
                            ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[0, col].axis('off')
            axes[1, col].text(0.5, 0.5, f'No {bus_type_name} buses', 
                            ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[1, col].axis('off')
            continue
        
        # Extract predictions and targets for this bus type
        pred_var1_list = []
        pred_var2_list = []
        targ_var1_list = []
        targ_var2_list = []
        
        for sample_idx in range(predictions.shape[0]):
            sample_bus_mask = bus_type_mask[sample_idx]
            if np.any(sample_bus_mask):
                pred_var1_list.append(predictions[sample_idx, sample_bus_mask, 0])
                pred_var2_list.append(predictions[sample_idx, sample_bus_mask, 1])
                targ_var1_list.append(targets[sample_idx, sample_bus_mask, 0])
                targ_var2_list.append(targets[sample_idx, sample_bus_mask, 1])
        
        if len(pred_var1_list) == 0:
            axes[0, col].text(0.5, 0.5, f'No {bus_type_name} data', 
                            ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[0, col].axis('off')
            axes[1, col].text(0.5, 0.5, f'No {bus_type_name} data', 
                            ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[1, col].axis('off')
            continue
        
        # Concatenate all values
        pred_var1 = np.concatenate(pred_var1_list)
        pred_var2 = np.concatenate(pred_var2_list)
        targ_var1 = np.concatenate(targ_var1_list)
        targ_var2 = np.concatenate(targ_var2_list)
        
        # Row 0: Variable 1
        ax = axes[0, col]
        ax.scatter(targ_var1, pred_var1, alpha=0.5, s=10)
        # Perfect prediction line (y=x)
        min_val = min(targ_var1.min(), pred_var1.min())
        max_val = max(targ_var1.max(), pred_var1.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel(f'Actual {var_names[bus_type_code]["var1"]}', fontsize=11)
        ax.set_ylabel(f'Predicted {var_names[bus_type_code]["var1"]}', fontsize=11)
        ax.set_title(f'{bus_type_name} Buses: {var_names[bus_type_code]["var1"]}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate and display R²
        try:
            slope, intercept, r_value, p_value, std_err = linregress(targ_var1, pred_var1)
            r_squared = r_value**2
            ax.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax.transAxes,
                   fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except:
            pass
        
        # Row 1: Variable 2
        ax = axes[1, col]
        ax.scatter(targ_var2, pred_var2, alpha=0.5, s=10)
        # Perfect prediction line (y=x)
        min_val = min(targ_var2.min(), pred_var2.min())
        max_val = max(targ_var2.max(), pred_var2.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel(f'Actual {var_names[bus_type_code]["var2"]}', fontsize=11)
        ax.set_ylabel(f'Predicted {var_names[bus_type_code]["var2"]}', fontsize=11)
        ax.set_title(f'{bus_type_name} Buses: {var_names[bus_type_code]["var2"]}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate and display R²
        try:
            slope, intercept, r_value, p_value, std_err = linregress(targ_var2, pred_var2)
            r_squared = r_value**2
            ax.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax.transAxes,
                   fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except:
            pass
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'predicted_vs_actual.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_distributions(predictions: np.ndarray, targets: np.ndarray,
                            bus_types: np.ndarray, case_name: str,
                            output_dir: str, model_name: str = ""):
    """
    Generate error distribution histograms, separated by bus type.
    
    Args:
        predictions: [n_samples, n_buses, 2] - predicted unknowns
        targets: [n_samples, n_buses, 2] - true unknowns
        bus_types: [n_samples, n_buses] - bus type codes [0=PQ, 1=PV, 2=Slack]
        case_name: Name of the test case
        output_dir: Directory to save plots
        model_name: Optional model name for title
    """
    if bus_types is None:
        raise ValueError("bus_types is required for OPF mode error distribution plots")
    
    # Create 2x3 grid: 2 rows (var1, var2) x 3 columns (PQ, PV, Slack)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Error Distributions - {case_name.upper()}' + (f' - {model_name}' if model_name else ''), 
                 fontsize=16, fontweight='bold')
    
    # Variable names by bus type
    var_names = {
        0: {'var1': 'Voltage Error (p.u.)', 'var2': 'Angle Error (rad)'},      # PQ
        1: {'var1': 'Reactive Power Error (p.u.)', 'var2': 'Angle Error (rad)'},  # PV
        2: {'var1': 'Active Power Error (p.u.)', 'var2': 'Reactive Power Error (p.u.)'}  # Slack
    }
    
    for bus_type_code, bus_type_name in [(0, 'PQ'), (1, 'PV'), (2, 'Slack')]:
        col = bus_type_code
        
        # Create mask for this bus type
        bus_type_mask = (bus_types == bus_type_code)
        
        if not np.any(bus_type_mask):
            axes[0, col].text(0.5, 0.5, f'No {bus_type_name} buses', 
                            ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[0, col].axis('off')
            axes[1, col].text(0.5, 0.5, f'No {bus_type_name} buses', 
                            ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[1, col].axis('off')
            continue
        
        # Extract errors for this bus type
        errors_var1_list = []
        errors_var2_list = []
        
        for sample_idx in range(predictions.shape[0]):
            sample_bus_mask = bus_type_mask[sample_idx]
            if np.any(sample_bus_mask):
                errors_var1 = predictions[sample_idx, sample_bus_mask, 0] - targets[sample_idx, sample_bus_mask, 0]
                errors_var2 = predictions[sample_idx, sample_bus_mask, 1] - targets[sample_idx, sample_bus_mask, 1]
                errors_var1_list.append(errors_var1)
                errors_var2_list.append(errors_var2)
        
        if len(errors_var1_list) == 0:
            axes[0, col].text(0.5, 0.5, f'No {bus_type_name} data', 
                            ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[0, col].axis('off')
            axes[1, col].text(0.5, 0.5, f'No {bus_type_name} data', 
                            ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[1, col].axis('off')
            continue
        
        # Concatenate all errors
        errors_var1 = np.concatenate(errors_var1_list)
        errors_var2 = np.concatenate(errors_var2_list)
        
        # Row 0: Variable 1 errors
        ax = axes[0, col]
        ax.hist(errors_var1, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax.axvline(x=np.mean(errors_var1), color='g', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors_var1):.6f}')
        ax.set_xlabel(var_names[bus_type_code]['var1'], fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{bus_type_name} Buses: {var_names[bus_type_code]["var1"]}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        std_err = np.std(errors_var1)
        ax.text(0.05, 0.95, f'Mean: {np.mean(errors_var1):.6f}\nStd: {std_err:.6f}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Row 1: Variable 2 errors
        ax = axes[1, col]
        ax.hist(errors_var2, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax.axvline(x=np.mean(errors_var2), color='g', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors_var2):.6f}')
        ax.set_xlabel(var_names[bus_type_code]['var2'], fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{bus_type_name} Buses: {var_names[bus_type_code]["var2"]}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        std_err = np.std(errors_var2)
        ax.text(0.05, 0.95, f'Mean: {np.mean(errors_var2):.6f}\nStd: {std_err:.6f}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'error_distributions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_calibration_diagram(model_outputs: np.ndarray, targets: np.ndarray,
                            bus_types: np.ndarray, case_name: str,
                            output_dir: str, model_name: str = "", config: Any = None):
    """
    Generate calibration plot (reliability diagram) for heteroscedastic uncertainty.
    
    Validates that predicted confidence intervals match actual coverage.
    A well-calibrated model will have points close to the y=x line.
    
    Args:
        model_outputs: [n_samples, n_buses, 4] - natural parameters [η1_var1, η1_var2, f2_var1, f2_var2]
        targets: [n_samples, n_buses, 2] - true unknowns
        bus_types: [n_samples, n_buses] - bus type codes [0=PQ, 1=PV, 2=Slack]
        case_name: Name of the test case
        output_dir: Directory to save plots
        model_name: Optional model name for title
        config: Config object for softplus beta
    """
    if bus_types is None:
        raise ValueError("bus_types is required for OPF mode calibration plot")
    
    import torch
    import torch.nn.functional as F
    from config import ModelOutputIndices
    
    # Convert to tensors for computation
    outputs_tensor = torch.from_numpy(model_outputs).float()
    targets_tensor = torch.from_numpy(targets).float()
    bus_types_tensor = torch.from_numpy(bus_types).long()
    
    # Extract natural parameters
    eta1_var1 = outputs_tensor[:, :, ModelOutputIndices.ETA1_VAR1]
    eta1_var2 = outputs_tensor[:, :, ModelOutputIndices.ETA1_VAR2]
    f2_var1 = outputs_tensor[:, :, ModelOutputIndices.F2_VAR1]
    f2_var2 = outputs_tensor[:, :, ModelOutputIndices.F2_VAR2]
    
    # Get softplus beta
    softplus_beta = getattr(config, 'HETEROSCEDASTIC_SOFTPLUS_BETA', 1.0) if config else 1.0
    
    # Compute g+ and then eta2
    g_plus_var1 = (1.0 / softplus_beta) * F.softplus(softplus_beta * f2_var1)
    g_plus_var2 = (1.0 / softplus_beta) * F.softplus(softplus_beta * f2_var2)
    eta2_var1 = -g_plus_var1
    eta2_var2 = -g_plus_var2
    
    # Convert to mean and variance
    eps = 1e-8
    mu_var1 = -eta1_var1 / (2.0 * eta2_var1 + eps)
    mu_var2 = -eta1_var2 / (2.0 * eta2_var2 + eps)
    sigma2_var1 = -1.0 / (2.0 * eta2_var1 + eps)
    sigma2_var2 = -1.0 / (2.0 * eta2_var2 + eps)
    sigma_var1 = torch.sqrt(sigma2_var1)
    sigma_var2 = torch.sqrt(sigma2_var2)
    
    # Convert back to numpy
    mu_var1 = mu_var1.numpy()
    mu_var2 = mu_var2.numpy()
    sigma_var1 = sigma_var1.numpy()
    sigma_var2 = sigma_var2.numpy()
    targets_np = targets_tensor.numpy()
    
    # Create 2x3 grid: 2 rows (var1, var2) x 3 columns (PQ, PV, Slack)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Calibration Diagram (Reliability) - {case_name.upper()}' + (f' - {model_name}' if model_name else ''), 
                 fontsize=16, fontweight='bold')
    
    # Variable names by bus type
    var_names = {
        0: {'var1': 'Voltage', 'var2': 'Angle'},      # PQ
        1: {'var1': 'Reactive Power', 'var2': 'Angle'},  # PV
        2: {'var1': 'Active Power', 'var2': 'Reactive Power'}  # Slack
    }
    
    # Confidence levels to test (from 10% to 90%)
    confidence_levels = np.arange(0.1, 1.0, 0.1)
    
    for bus_type_code, bus_type_name in [(0, 'PQ'), (1, 'PV'), (2, 'Slack')]:
        col = bus_type_code
        
        # Create mask for this bus type
        bus_type_mask = (bus_types == bus_type_code)
        
        if not np.any(bus_type_mask):
            axes[0, col].text(0.5, 0.5, f'No {bus_type_name} buses', 
                            ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[0, col].axis('off')
            axes[1, col].text(0.5, 0.5, f'No {bus_type_name} buses', 
                            ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[1, col].axis('off')
            continue
        
        # Extract predictions, uncertainties, and targets for this bus type
        mu_var1_list = []
        mu_var2_list = []
        sigma_var1_list = []
        sigma_var2_list = []
        targ_var1_list = []
        targ_var2_list = []
        
        for sample_idx in range(mu_var1.shape[0]):
            sample_bus_mask = bus_type_mask[sample_idx]
            if np.any(sample_bus_mask):
                mu_var1_list.append(mu_var1[sample_idx, sample_bus_mask])
                mu_var2_list.append(mu_var2[sample_idx, sample_bus_mask])
                sigma_var1_list.append(sigma_var1[sample_idx, sample_bus_mask])
                sigma_var2_list.append(sigma_var2[sample_idx, sample_bus_mask])
                targ_var1_list.append(targets_np[sample_idx, sample_bus_mask, 0])
                targ_var2_list.append(targets_np[sample_idx, sample_bus_mask, 1])
        
        if len(mu_var1_list) == 0:
            axes[0, col].text(0.5, 0.5, f'No {bus_type_name} data', 
                            ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[0, col].axis('off')
            axes[1, col].text(0.5, 0.5, f'No {bus_type_name} data', 
                            ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[1, col].axis('off')
            continue
        
        # Concatenate all values
        mu_var1_flat = np.concatenate(mu_var1_list)
        mu_var2_flat = np.concatenate(mu_var2_list)
        sigma_var1_flat = np.concatenate(sigma_var1_list)
        sigma_var2_flat = np.concatenate(sigma_var2_list)
        targ_var1_flat = np.concatenate(targ_var1_list)
        targ_var2_flat = np.concatenate(targ_var2_list)
        
        # Row 0: Variable 1 calibration
        ax = axes[0, col]
        actual_coverage = []
        for conf_level in confidence_levels:
            # Calculate z-score for this confidence level (two-tailed)
            z_score = norm.ppf(0.5 + conf_level / 2.0)
            
            # Predicted confidence interval
            lower_bound = mu_var1_flat - z_score * sigma_var1_flat
            upper_bound = mu_var1_flat + z_score * sigma_var1_flat
            
            # Actual coverage: fraction of true values within predicted interval
            within_interval = (targ_var1_flat >= lower_bound) & (targ_var1_flat <= upper_bound)
            actual_coverage.append(np.mean(within_interval))
        
        ax.plot(confidence_levels, actual_coverage, 'o-', linewidth=2, markersize=8, label='Actual Coverage')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
        ax.set_xlabel('Predicted Confidence Level', fontsize=11)
        ax.set_ylabel('Actual Coverage', fontsize=11)
        ax.set_title(f'{bus_type_name} Buses: {var_names[bus_type_code]["var1"]}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Row 1: Variable 2 calibration
        ax = axes[1, col]
        actual_coverage = []
        for conf_level in confidence_levels:
            z_score = norm.ppf(0.5 + conf_level / 2.0)
            
            lower_bound = mu_var2_flat - z_score * sigma_var2_flat
            upper_bound = mu_var2_flat + z_score * sigma_var2_flat
            
            within_interval = (targ_var2_flat >= lower_bound) & (targ_var2_flat <= upper_bound)
            actual_coverage.append(np.mean(within_interval))
        
        ax.plot(confidence_levels, actual_coverage, 'o-', linewidth=2, markersize=8, label='Actual Coverage')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
        ax.set_xlabel('Predicted Confidence Level', fontsize=11)
        ax.set_ylabel('Actual Coverage', fontsize=11)
        ax.set_title(f'{bus_type_name} Buses: {var_names[bus_type_code]["var2"]}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'calibration_diagram.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

