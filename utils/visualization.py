"""
Visualization utilities for power system machine learning models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Any
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='matplotlib')

def safe_plot_operation(plot_func, *args, **kwargs):
    try:
        plt.clf()
        plt.close('all')
        result = plot_func(*args, **kwargs)
        plt.close('all')
        return result
    except Exception as e:
        plt.close('all')
        print(f"Warning: Plotting operation failed: {e}")
        return None

def plot_training_history(history: Dict[str, list], model_name: str, config: Any, 
                         num_buses: int, is_physics_informed: bool = True):
    return safe_plot_operation(_plot_training_history_impl, history, model_name, config, num_buses, is_physics_informed)

def _plot_training_history_impl(history: Dict[str, list], model_name: str, config: Any, 
                               num_buses: int, is_physics_informed: bool = True):
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training History - {model_name}', fontsize=16, fontweight='bold')
    
    # Row 1, Col 1: MSE
    if 'train_mse' in history:
        axes[0, 0].plot(history['train_mse'], label='Train', linewidth=2)
        axes[0, 0].plot(history['val_mse'], label='Validation', linewidth=2)
        axes[0, 0].set_title('MSE (L1 Loss)', fontweight='bold')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Row 1, Col 2: Physics Loss (L2)
    if is_physics_informed and 'train_physics_loss' in history:
        axes[0, 1].plot(history['train_physics_loss'], label='Train', linewidth=2, color='red')
        axes[0, 1].set_title('Physics Loss (L2)', fontweight='bold')
        axes[0, 1].set_ylabel('Power Mismatch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].axis('off')
        
    # Row 1, Col 3: Safety Loss (L3)
    if is_physics_informed and 'train_safety_loss' in history:
        axes[0, 2].plot(history['train_safety_loss'], label='Train', linewidth=2, color='blue')
        axes[0, 2].set_title('Safety Loss (L3)', fontweight='bold')
        axes[0, 2].set_ylabel('Voltage Violation')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].axis('off')
        
    # Row 2, Col 1: Total Loss
    if 'train_total_loss' in history:
        axes[1, 0].plot(history['train_total_loss'], label='Train', linewidth=2, color='black')
        axes[1, 0].set_title('Total Weighted Loss', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
    # Row 2, Col 2: Learned Sigmas (Uncertainty Weights)
    if is_physics_informed and 'train_sigmas' in history and history['train_sigmas']:
        # sigmas is list of [s1, s2, s3]
        sigmas = np.array(history['train_sigmas'])
        if sigmas.ndim == 2 and sigmas.shape[1] == 3:
            axes[1, 1].plot(sigmas[:, 0], label='σ_data (L1)', linewidth=2)
            axes[1, 1].plot(sigmas[:, 1], label='σ_phys (L2)', linewidth=2)
            axes[1, 1].plot(sigmas[:, 2], label='σ_safe (L3)', linewidth=2)
            axes[1, 1].set_title('Learned Uncertainties (σ)', fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
    # Row 2, Col 3: Generalization Gap
    if 'train_mse' in history and 'val_mse' in history:
        gap = np.abs(np.array(history['train_mse']) - np.array(history['val_mse']))
        axes[1, 2].plot(gap, color='purple', linewidth=2)
        axes[1, 2].set_title('Generalization Gap', fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = config.get_training_history_path(num_buses, model_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_convergence(history: list, model_name: str, config: Any, num_buses: int):
    return safe_plot_operation(_plot_convergence_impl, history, model_name, config, num_buses)

def _plot_convergence_impl(history: list, model_name: str, config: Any, num_buses: int):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history) + 1), history, 'b-', marker='o')
    plt.title(f'Convergence - {model_name}', fontsize=14)
    plt.xlabel('Iteration')
    plt.ylabel('Best Loss')
    plt.grid(True)
    
    save_path = config.get_convergence_plot_path(num_buses, model_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_all_renewable_impacts(renewable_impact_data: pd.DataFrame, config: Any, 
                               num_buses: int, model_name: str):
    return safe_plot_operation(_plot_all_renewable_impacts_impl, renewable_impact_data, config, num_buses, model_name)

def _plot_all_renewable_impacts_impl(df: pd.DataFrame, config: Any, num_buses: int, model_name: str):
    if df.empty: return
    
    metrics = {
        'carbon_emissions': 'Carbon Emissions',
        'voltage_deviation': 'Voltage Deviation',
        'power_loss': 'Power Loss'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Renewable Impact Analysis - {model_name}', fontsize=16, fontweight='bold')
    
    unique_fracs = sorted(df['renewable_fraction'].unique())
    
    for idx, (col, label) in enumerate(metrics.items()):
        ax = axes[idx]
        if col in df.columns:
            data = [df[df['renewable_fraction'] == f][col].values for f in unique_fracs]
            ax.boxplot(data, positions=range(len(unique_fracs)))
            ax.set_title(label)
            ax.set_xticklabels([f'{f:.1f}' for f in unique_fracs])
            ax.set_xlabel('Renewable Fraction')
    
    plt.tight_layout()
    model_dir = config.get_model_eval_dir(num_buses, model_name)
    os.makedirs(model_dir, exist_ok=True)
    plt.savefig(os.path.join(model_dir, 'ri_combined.png'), dpi=300)
    plt.close()

def create_model_comparison_plot(all_results: list, save_path: str = None):
    # Simplified implementation
    pass
