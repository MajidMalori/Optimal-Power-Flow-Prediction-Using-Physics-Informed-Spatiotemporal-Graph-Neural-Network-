"""
Critical Evaluation Plots for Model Diagnostics

Provides essential visualizations for diagnosing regression models:
1. Predicted vs. Actual Scatter Plots (for 10-dimensional full state)
2. Error Distribution Histograms (for 10-dimensional full state)
3. Calibration Plot (Reliability Diagram) for MC Dropout uncertainty

NOTE: Updated for Full State Reconstruction (10 outputs), not OPF (2 outputs).
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
    Generate predicted vs. actual scatter plots for Full State Reconstruction.
    
    Plots voltage magnitude (VM) and voltage angle (VA) across all buses.
    
    Args:
        predictions: [n_samples, n_buses, 10] - predicted full state
        targets: [n_samples, n_buses, 10] - true full state
        bus_types: [n_samples, n_buses] - bus type codes (unused, kept for compatibility)
        case_name: Name of the test case
        output_dir: Directory to save plots
        model_name: Optional model name for title
    """
    # Full State Reconstruction: We plot VM (col 8) and VA (col 9) for all buses
    # No bus-type separation needed since we reconstruct the full state
    
    # Extract VM and VA from 10-dimensional predictions
    pred_vm = predictions[:, :, 8].flatten()  # Voltage Magnitude
    pred_va = predictions[:, :, 9].flatten()  # Voltage Angle
    targ_vm = targets[:, :, 8].flatten()
    targ_va = targets[:, :, 9].flatten()
    
    # Create 1x2 subplot: VM and VA
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Predicted vs. Actual - {case_name.upper()}' + (f' - {model_name}' if model_name else ''), 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Voltage Magnitude
    ax = axes[0]
    ax.scatter(targ_vm, pred_vm, alpha=0.5, s=10)
    min_val = min(targ_vm.min(), pred_vm.min())
    max_val = max(targ_vm.max(), pred_vm.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Voltage Magnitude (p.u.)', fontsize=11)
    ax.set_ylabel('Predicted Voltage Magnitude (p.u.)', fontsize=11)
    ax.set_title('Voltage Magnitude', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    try:
        slope, intercept, r_value, p_value, std_err = linregress(targ_vm, pred_vm)
        r_squared = r_value**2
        ax.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax.transAxes,
               fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    except:
        pass
    
    # Plot 2: Voltage Angle
    ax = axes[1]
    ax.scatter(targ_va, pred_va, alpha=0.5, s=10)
    min_val = min(targ_va.min(), pred_va.min())
    max_val = max(targ_va.max(), pred_va.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Voltage Angle (rad)', fontsize=11)
    ax.set_ylabel('Predicted Voltage Angle (rad)', fontsize=11)
    ax.set_title('Voltage Angle', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    try:
        slope, intercept, r_value, p_value, std_err = linregress(targ_va, pred_va)
        r_squared = r_value**2
        ax.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax.transAxes,
               fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    except:
        pass
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{model_name}_predicted_vs_actual.png' if model_name else 'predicted_vs_actual.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_distributions(predictions: np.ndarray, targets: np.ndarray,
                            bus_types: np.ndarray, case_name: str,
                            output_dir: str, model_name: str = ""):
    """
    Generate error distribution histograms for Full State Reconstruction.
    
    Args:
        predictions: [n_samples, n_buses, 10] - predicted full state
        targets: [n_samples, n_buses, 10] - true full state
        bus_types: [n_samples, n_buses] - bus type codes (unused, kept for compatibility)
        case_name: Name of the test case
        output_dir: Directory to save plots
        model_name: Optional model name for title
    """
    
    # Extract VM and VA errors from 10-dimensional predictions
    errors_vm = predictions[:, :, 8].flatten() - targets[:, :, 8].flatten()
    errors_va = predictions[:, :, 9].flatten() - targets[:, :, 9].flatten()
    
    # Create 1x2 subplot: VM and VA error distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Error Distributions - {case_name.upper()}' + (f' - {model_name}' if model_name else ''), 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Voltage Magnitude errors
    ax = axes[0]
    ax.hist(errors_vm, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(x=np.mean(errors_vm), color='g', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors_vm):.6f}')
    ax.set_xlabel('Voltage Magnitude Error (p.u.)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Voltage Magnitude Error Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    std_err_vm = np.std(errors_vm)
    ax.text(0.05, 0.95, f'Mean: {np.mean(errors_vm):.6f}\nStd: {std_err_vm:.6f}', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Voltage Angle errors
    ax = axes[1]
    ax.hist(errors_va, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(x=np.mean(errors_va), color='g', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors_va):.6f}')
    ax.set_xlabel('Voltage Angle Error (rad)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Voltage Angle Error Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    std_err_va = np.std(errors_va)
    ax.text(0.05, 0.95, f'Mean: {np.mean(errors_va):.6f}\nStd: {std_err_va:.6f}', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{model_name}_error_distributions.png' if model_name else 'error_distributions.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_calibration_diagram(model_outputs: np.ndarray, targets: np.ndarray,
                            bus_types: np.ndarray, case_name: str,
                            output_dir: str, model_name: str = "", config: Any = None):
    """
    Generate calibration plot (reliability diagram) for MC Dropout uncertainty.
    
    Validates that predicted confidence intervals match actual coverage.
    A well-calibrated model will have points close to the y=x line.
    
    Args:
        model_outputs: [n_samples, n_buses, 10] - MC Dropout predictions (mean)
        targets: [n_samples, n_buses, 10] - true full state
        bus_types: [n_samples, n_buses] - bus type codes (unused, kept for compatibility)
        case_name: Name of the test case
        output_dir: Directory to save plots
        model_name: Optional model name for title
        config: Config object (unused, kept for compatibility)
    """
    # MC Dropout: We don't have explicit uncertainty estimates in model_outputs
    # This function is now a placeholder for compatibility
    # Real calibration would require multiple forward passes with dropout enabled
    
    print(f"[WARNING] plot_calibration_diagram is not yet implemented for MC Dropout uncertainty.")
    print(f"          Skipping calibration plot for {case_name}.")
    
    # Create a placeholder figure to avoid breaking the pipeline
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.text(0.5, 0.5, 'Calibration Plot\n(Not yet implemented for MC Dropout)', 
           transform=ax.transAxes, ha='center', va='center',
           fontsize=14, fontweight='bold')
    ax.set_title(f'Calibration Diagram - {case_name.upper()}' + (f' - {model_name}' if model_name else ''), 
                fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'calibration_diagram.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

