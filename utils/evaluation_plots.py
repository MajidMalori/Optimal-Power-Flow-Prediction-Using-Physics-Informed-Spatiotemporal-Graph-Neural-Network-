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
matplotlib.use('Agg')  # Use non-interactive backend for faster plotting
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode (faster)
from typing import Any
import warnings
from scipy.stats import linregress, norm

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Creating legend with loc="best".*')


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
    
    # Create 1x2 subplot: VM and VA (reduced size for faster rendering)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Reduced from 14x6
    fig.suptitle(f'Predicted vs. Actual - {case_name.upper()}' + (f' - {model_name}' if model_name else ''), 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Voltage Magnitude
    ax = axes[0]
    # OPTIMIZED: Sample if too many points for faster rendering
    n_points = len(targ_vm)
    if n_points > 10000:
        indices = np.random.choice(n_points, 10000, replace=False)
        ax.scatter(targ_vm[indices], pred_vm[indices], alpha=0.4, s=8)  # Reduced alpha and size
    else:
        ax.scatter(targ_vm, pred_vm, alpha=0.4, s=8)  # Reduced alpha and size
    min_val = min(targ_vm.min(), pred_vm.min())
    max_val = max(targ_vm.max(), pred_vm.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Voltage Magnitude (p.u.)', fontsize=11)
    ax.set_ylabel('Predicted Voltage Magnitude (p.u.)', fontsize=11)
    ax.set_title('Voltage Magnitude', fontweight='bold')
    ax.legend(loc='lower right')  # Lower right to avoid overlap with R² text box
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
    # OPTIMIZED: Sample if too many points for faster rendering
    n_points = len(targ_va)
    if n_points > 10000:
        indices = np.random.choice(n_points, 10000, replace=False)
        ax.scatter(targ_va[indices], pred_va[indices], alpha=0.4, s=8)  # Reduced alpha and size
    else:
        ax.scatter(targ_va, pred_va, alpha=0.4, s=8)  # Reduced alpha and size
    min_val = min(targ_va.min(), pred_va.min())
    max_val = max(targ_va.max(), pred_va.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Voltage Angle (rad)', fontsize=11)
    ax.set_ylabel('Predicted Voltage Angle (rad)', fontsize=11)
    ax.set_title('Voltage Angle', fontweight='bold')
    ax.legend(loc='lower right')  # Lower right to avoid overlap with R² text box
    ax.grid(True, alpha=0.3)
    
    try:
        slope, intercept, r_value, p_value, std_err = linregress(targ_va, pred_va)
        r_squared = r_value**2
        ax.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax.transAxes,
               fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    except:
        pass
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot with optimized DPI (150 is high quality but 4x faster than 300)
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{model_name}_predicted_vs_actual.png' if model_name else 'predicted_vs_actual.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close('all')  # Close all figures to free memory


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
    
    # Create 1x2 subplot: VM and VA error distributions (reduced size for faster rendering)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Reduced from 14x6
    fig.suptitle(f'Error Distributions - {case_name.upper()}' + (f' - {model_name}' if model_name else ''), 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Voltage Magnitude errors
    ax = axes[0]
    # OPTIMIZED: Reduce bins for faster rendering
    ax.hist(errors_vm, bins=40, alpha=0.7, edgecolor='black')  # Reduced from 50 bins
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(x=np.mean(errors_vm), color='g', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors_vm):.6f}')
    ax.set_xlabel('Voltage Magnitude Error (p.u.)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Voltage Magnitude Error Distribution', fontweight='bold')
    ax.legend(loc='upper right')  # Upper right to avoid overlap with stats text box in upper left
    ax.grid(True, alpha=0.3)
    
    std_err_vm = np.std(errors_vm)
    ax.text(0.05, 0.95, f'Mean: {np.mean(errors_vm):.6f}\nStd: {std_err_vm:.6f}', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Voltage Angle errors
    ax = axes[1]
    # OPTIMIZED: Reduce bins for faster rendering
    ax.hist(errors_va, bins=40, alpha=0.7, edgecolor='black')  # Reduced from 50 bins
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(x=np.mean(errors_va), color='g', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors_va):.6f}')
    ax.set_xlabel('Voltage Angle Error (rad)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Voltage Angle Error Distribution', fontweight='bold')
    ax.legend(loc='upper right')  # Upper right to avoid overlap with stats text box in upper left
    ax.grid(True, alpha=0.3)
    
    std_err_va = np.std(errors_va)
    ax.text(0.05, 0.95, f'Mean: {np.mean(errors_va):.6f}\nStd: {std_err_va:.6f}', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot with optimized DPI (150 is high quality but 4x faster than 300)
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{model_name}_error_distributions.png' if model_name else 'error_distributions.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close('all')  # Close all figures to free memory


def plot_calibration_diagram(model_outputs: np.ndarray, targets: np.ndarray,
                            bus_types: np.ndarray, case_name: str,
                            output_dir: str, model_name: str = "", config: Any = None,
                            uncertainties: np.ndarray = None, targets_norm: np.ndarray = None):
    """
    Generate proper calibration plot (reliability diagram) for uncertainty quantification.
    
    Shows:
    1. Reliability diagram: Predicted confidence intervals vs. actual coverage
    2. Uncertainty vs. Error scatter: Whether high uncertainty correlates with high error
    
    Args:
        model_outputs: [n_samples, n_buses, 10] - Model predictions (mean) in NORMALIZED space
        targets: [n_samples, n_buses, 10] - True full state (unused, kept for compatibility)
        bus_types: [n_samples, n_buses] - Bus type codes (unused, kept for compatibility)
        case_name: Name of the test case
        output_dir: Directory to save plots
        model_name: Optional model name for title
        config: Config object (unused, kept for compatibility)
        uncertainties: [n_samples, n_buses, 10] - Prediction uncertainties (std from MC Dropout) in NORMALIZED space
                       REQUIRED - function will fail if None
        targets_norm: [n_samples, n_buses, 10] - True full state in NORMALIZED space
                      REQUIRED - function will fail if None (needed for proper scale matching with uncertainties)
    
    Raises:
        ValueError: If uncertainties is None or has wrong shape
    """
    if uncertainties is None:
        raise ValueError("uncertainties is required for calibration diagram. Cannot plot without uncertainty data.")
    
    if uncertainties.shape != model_outputs.shape:
        raise ValueError(f"uncertainties shape {uncertainties.shape} must match model_outputs shape {model_outputs.shape}")
    
    if targets_norm is None:
        raise ValueError("targets_norm is required for calibration diagram. Cannot compare normalized uncertainties with denormalized targets.")
    
    if targets_norm.shape != model_outputs.shape:
        raise ValueError(f"targets_norm shape {targets_norm.shape} must match model_outputs shape {model_outputs.shape}")
    
    # Calculate prediction errors in NORMALIZED space (matches uncertainties)
    errors = model_outputs - targets_norm  # [n_samples, n_buses, 10]
    abs_errors = np.abs(errors)  # Absolute errors
    
    # Flatten for analysis
    errors_flat = abs_errors.flatten()
    uncertainties_flat = uncertainties.flatten()
    
    # Create figure with 2 subplots (reduced size for faster rendering)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Reduced from 14x6
    
    # Subplot 1: Reliability Diagram (Calibration Curve)
    ax1 = axes[0]
    
    # OPTIMIZED: Create reliability diagram with fewer confidence levels for faster computation
    confidence_levels = np.linspace(0.1, 0.99, 15)  # Reduced from 20 to 15 levels
    actual_coverage = []
    predicted_coverage = []
    
    for conf_level in confidence_levels:
        # For Gaussian assumption: z-score for confidence level
        z_score = norm.ppf(0.5 + conf_level / 2.0)  # Two-tailed
        
        # Predicted coverage: fraction of predictions within z_score * uncertainty
        predicted_interval = z_score * uncertainties_flat
        within_interval = errors_flat <= predicted_interval
        actual_frac = np.mean(within_interval)
        
        actual_coverage.append(actual_frac)
        predicted_coverage.append(conf_level)
    
    # Plot reliability diagram
    ax1.plot(predicted_coverage, actual_coverage, 'o-', linewidth=2, markersize=6, 
            label='Model Calibration', color='steelblue')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
    ax1.set_xlabel('Predicted Confidence Level', fontsize=11)
    ax1.set_ylabel('Actual Coverage', fontsize=11)
    ax1.set_title('Reliability Diagram (Calibration Curve)', fontweight='bold')
    ax1.legend(loc='lower right')  # Lower right to avoid overlap with ECE text box in upper left
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Calculate Expected Calibration Error (ECE)
    ece = np.mean(np.abs(np.array(actual_coverage) - np.array(predicted_coverage)))
    ax1.text(0.05, 0.95, f'ECE: {ece:.4f}\n(Lower is better)', 
            transform=ax1.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Subplot 2: Uncertainty vs. Error Scatter
    ax2 = axes[1]
    
    # Sample for visualization if too many points
    n_points = len(errors_flat)
    if n_points > 10000:
        indices = np.random.choice(n_points, 10000, replace=False)
        errors_flat = errors_flat[indices]
        uncertainties_flat = uncertainties_flat[indices]
    
    ax2.scatter(uncertainties_flat, errors_flat, alpha=0.3, s=5, color='steelblue')
    ax2.set_xlabel('Predicted Uncertainty (Std)', fontsize=11)
    ax2.set_ylabel('Absolute Error', fontsize=11)
    ax2.set_title('Uncertainty vs. Error', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    if len(errors_flat) > 1:
        correlation = np.corrcoef(uncertainties_flat, errors_flat)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {correlation:.4f}\n(Positive = Good)', 
                transform=ax2.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'Uncertainty Calibration - {case_name.upper()}' + (f' - {model_name}' if model_name else ''), 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save plot with optimized DPI (150 is high quality but 4x faster than 300)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'calibration_diagram.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close('all')  # Close all figures to free memory

