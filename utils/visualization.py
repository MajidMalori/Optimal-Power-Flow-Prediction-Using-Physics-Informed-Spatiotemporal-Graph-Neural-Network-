"""
Visualization utilities for power system machine learning models.
Contains all plotting and visualization functions used in training and evaluation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent threading issues
import matplotlib.pyplot as plt
from typing import Dict, Any
import warnings

# Suppress matplotlib warnings that can cause threading issues
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='matplotlib')

def safe_plot_operation(plot_func, *args, **kwargs):
    """
    Safely execute plotting operations to prevent threading issues.
    Ensures all figures are properly closed and matplotlib state is clean.
    """
    try:
        # Clear any existing figures to prevent memory leaks
        plt.clf()
        plt.close('all')
        
        # Execute the plotting function
        result = plot_func(*args, **kwargs)
        
        # Ensure all figures are closed
        plt.close('all')
        
        return result
    except Exception as e:
        # Clean up on error
        plt.close('all')
        print(f"Warning: Plotting operation failed: {e}")
        return None


def plot_training_history(history: Dict[str, list], model_name: str, config: Any, 
                         num_buses: int, is_physics_informed: bool = True):
    """Plots and saves the training history for the best model."""
    return safe_plot_operation(_plot_training_history_impl, history, model_name, config, num_buses, is_physics_informed)

def _plot_training_history_impl(history: Dict[str, list], model_name: str, config: Any, 
                               num_buses: int, is_physics_informed: bool = True):
    """
    Internal implementation of plot_training_history - creates single consolidated 2x3 grid plot.
    Shows: RMSE, MSE, Power Violation, Voltage Violation, Effective Loss Weights (λ), Generalization Gap.
    """
    
    # Single consolidated plot: 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training History - {model_name}', fontsize=16, fontweight='bold')
    
    # Row 1, Col 1: RMSE
    train_rmse = [mse**0.5 for mse in history['train_mse']]
    val_rmse = [mse**0.5 for mse in history['val_mse']]
    axes[0, 0].plot(train_rmse, label='Train', linewidth=2)
    axes[0, 0].plot(val_rmse, label='Validation', linewidth=2)
    axes[0, 0].set_title('RMSE', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Row 1, Col 2: MSE (Prediction Error)
    axes[0, 1].plot(history['train_mse'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_mse'], label='Validation', linewidth=2)
    axes[0, 1].set_title('MSE (Prediction Error)', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Row 1, Col 3: Power Balance Violation (only for physics-informed)
    if is_physics_informed and 'train_power_violation' in history:
        axes[0, 2].plot(history['train_power_violation'], label='Train', linewidth=2, color='red')
        axes[0, 2].plot(history['val_power_violation'], label='Validation', linewidth=2, color='darkred')
        axes[0, 2].set_title('Power Balance Violation', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Violation')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'Power violation\nnot available', 
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightgray'))
        axes[0, 2].axis('off')
    
    # Row 2, Col 1: Voltage Limit Violation (only for physics-informed)
    if is_physics_informed and 'train_voltage_violation' in history:
        axes[1, 0].plot(history['train_voltage_violation'], label='Train', linewidth=2, color='blue')
        axes[1, 0].plot(history['val_voltage_violation'], label='Validation', linewidth=2, color='darkblue')
        axes[1, 0].set_title('Voltage Limit Violation', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Violation')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Voltage violation\nnot available', 
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightgray'))
        axes[1, 0].axis('off')
    
    # Row 2, Col 2: Effective Loss Weights (λ) - more interpretable than raw σ
    has_learnable_uncertainty = ('sigma_data' in history and len(history['sigma_data']) > 0)
    if has_learnable_uncertainty and is_physics_informed:
        axes[1, 1].plot(history['effective_lambda_p'], label='λ_p (power)', linewidth=2, color='darkred')
        axes[1, 1].plot(history['effective_lambda_v'], label='λ_v (voltage)', linewidth=2, color='darkblue')
        axes[1, 1].set_title('Effective Loss Weights (λ = 1/(2σ²))', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Effective Weight')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    else:
        axes[1, 1].text(0.5, 0.5, 'Effective weights\nnot available', 
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightgray'))
        axes[1, 1].axis('off')
    
    # Row 2, Col 3: Generalization Gap
    epochs = list(range(1, len(history['train_mse']) + 1))
    train_val_gap = [abs(t - v) for t, v in zip(history['train_mse'], history['val_mse'])]
    axes[1, 2].plot(epochs, train_val_gap, color='red', label='Train-Val Gap', linewidth=2)
    axes[1, 2].set_title('Generalization Gap', fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('|Train MSE - Val MSE|')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save consolidated Training History plot
    save_path_history = config.get_training_history_path(num_buses, model_name)
    os.makedirs(os.path.dirname(save_path_history), exist_ok=True)
    plt.savefig(save_path_history, dpi=300)
    plt.close()


def plot_convergence(history: list, model_name: str, config: Any, num_buses: int):
    """Plots the convergence curve of the MoSOA algorithm."""
    return safe_plot_operation(_plot_convergence_impl, history, model_name, config, num_buses)

def _plot_convergence_impl(history: list, model_name: str, config: Any, num_buses: int):
    """Internal implementation of plot_convergence."""
    plt.figure(figsize=(10, 6))
    # Create explicit iteration numbers for x-axis (1-based indexing for readability)
    iterations = list(range(1, len(history) + 1))
    plt.plot(iterations, history, 'b-', label='Convergence curve')
    plt.title(f'MoSOA Convergence for {model_name}', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    
    # Set appropriate Y-axis label based on model type
    is_physics_informed = 'PI' in model_name  # Models with 'PI' are physics-informed
    if is_physics_informed:
        plt.ylabel('Best MSE + Physics-Informed Loss', fontsize=12)
    else:
        plt.ylabel('Best MSE Loss', fontsize=12)
    
    plt.grid(True)
    plt.legend()
    
    save_path = config.get_convergence_plot_path(num_buses, model_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_all_renewable_impacts(renewable_impact_data: pd.DataFrame, config: Any, 
                               num_buses: int, model_name: str):
    return safe_plot_operation(_plot_all_renewable_impacts_impl, renewable_impact_data, config, num_buses, model_name)

def _plot_all_renewable_impacts_impl(renewable_impact_data: pd.DataFrame, config: Any, 
                                    num_buses: int, model_name: str):
    """
    Plot all renewable impact metrics for a physics-informed model in a single 2x2 grid.
    Uses professional box plots matching data_profile_story quality.
    """
    if renewable_impact_data.empty:
        print(f"No renewable impact data to plot for {model_name}")
        return
        
    # Update metrics dictionary to match column names in renewable_impact_data
    metrics = {
        'normalized_carbon_emissions': 'Carbon Emissions',
        'voltage_deviation': 'Voltage Deviation',
        'power_loss': 'Power Loss',
        'power_flow': 'Power Flow'
    }
    
    # Get unique renewable fractions (discrete values)
    x_col = 'renewable_fraction'
    unique_fracs = sorted(renewable_impact_data[x_col].unique())
    
    # Professional color scheme (matching data_profile_story style)
    primary_color = 'steelblue'
    trend_color = '#d62728'  # Professional red for trends
    
    # Create 2x2 subplot grid with professional styling
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle(f'Renewable Impact Analysis - {model_name}', fontsize=18, fontweight='bold', y=0.995)
    
    for idx, (metric, label) in enumerate(metrics.items()):
        ax = axes[idx]
        
        try:
            if metric not in renewable_impact_data.columns:
                # If metric not available, show placeholder
                ax.text(0.5, 0.5, f'{label}\nnot available', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", edgecolor='black', linewidth=1.5))
                ax.set_title(f'{label}', fontsize=14, fontweight='bold')
                ax.axis('off')
                continue
            
            # Prepare data for box plots (group by renewable fraction)
            plot_data = []
            plot_positions = []
            
            for frac_idx, frac in enumerate(unique_fracs):
                mask = (renewable_impact_data[x_col] == frac)
                values = renewable_impact_data.loc[mask, metric].dropna().values
                
                if len(values) > 0:
                    plot_data.append(values)
                    plot_positions.append(frac_idx)
            
            # Create professional box plots
            if plot_data:
                bp = ax.boxplot(plot_data, positions=plot_positions, widths=0.6,
                               patch_artist=True, showmeans=True, meanline=False,
                               boxprops=dict(facecolor=primary_color, alpha=0.7, linewidth=1.5),
                               medianprops=dict(color='white', linewidth=2),
                               meanprops=dict(marker='D', markerfacecolor='gold', 
                                            markeredgecolor='black', markersize=8, markeredgewidth=1),
                               whiskerprops=dict(linewidth=1.5),
                               capprops=dict(linewidth=1.5),
                               flierprops=dict(marker='o', markersize=6, alpha=0.6, 
                                             markerfacecolor='darkred', markeredgecolor='black'))
            
            # Calculate and plot trend line (using median values)
            medians = []
            x_vals = []
            for frac in unique_fracs:
                mask = (renewable_impact_data[x_col] == frac)
                values = renewable_impact_data.loc[mask, metric].dropna().values
                if len(values) > 0:
                    medians.append(np.median(values))
                    x_vals.append(frac)
            
            if len(x_vals) > 1:
                try:
                    z = np.polyfit(x_vals, medians, 1)
                    p = np.poly1d(z)
                    x_sorted = np.sort(x_vals)
                    ax.plot(x_sorted, p(x_sorted), '--', linewidth=3, 
                           color=trend_color, alpha=0.9, label=f'Trend: y={z[0]:.4g}x + {z[1]:.4g}',
                           zorder=10)
                except:
                    pass
            
            # Professional formatting
            ax.set_title(f'{label}', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Renewable Energy Fraction', fontsize=12, fontweight='bold')
            ax.set_ylabel(label, fontsize=12, fontweight='bold')
            ax.set_xticks(unique_fracs)
            ax.set_xticklabels([f'{f:.1f}' for f in unique_fracs], fontsize=11)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.legend(fontsize=10, loc='best', framealpha=0.95, edgecolor='black', frameon=True)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
            ax.set_axisbelow(True)  # Grid behind plots
            
        except KeyError as e:
            # If metric not available, show placeholder
            ax.text(0.5, 0.5, f'{label}\nnot available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", edgecolor='black', linewidth=1.5))
            ax.set_title(f'{label}', fontsize=14, fontweight='bold')
            ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save combined plot directly in model folder
    model_dir = config.get_model_eval_dir(num_buses, model_name)
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, 'ri_combined.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def create_model_comparison_plot(all_results: list, save_path: str = None):
    """Create a comprehensive comparison plot of all model performances."""
    if not all_results:
        print("No results to plot")
        return
    
    # Prepare data for plotting
    models = [r['model_name'] for r in all_results]
    bus_systems = [r['num_buses'] for r in all_results]
    scores = [r['final_test_score'] for r in all_results if r['final_test_score'] != float('inf')]
    
    if not scores:
        print("No valid scores to plot")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Performance by model type
    model_scores = {}
    for r in all_results:
        if r['final_test_score'] != float('inf'):
            model_type = 'Physics' if r['is_physics_informed'] else 'Non-Physics'
            if model_type not in model_scores:
                model_scores[model_type] = []
            model_scores[model_type].append(r['final_test_score'])
    
    types = list(model_scores.keys())
    avg_scores = [np.mean(model_scores[t]) for t in types]
    ax1.bar(types, avg_scores)
    ax1.set_title('Average Performance by Model Type')
    ax1.set_ylabel('Test Score')
    
    # Plot 2: Performance by bus system
    bus_scores = {}
    for r in all_results:
        if r['final_test_score'] != float('inf'):
            bus_sys = f"{r['num_buses']}-bus"
            if bus_sys not in bus_scores:
                bus_scores[bus_sys] = []
            bus_scores[bus_sys].append(r['final_test_score'])
    
    bus_systems_unique = list(bus_scores.keys())
    avg_bus_scores = [np.mean(bus_scores[b]) for b in bus_systems_unique]
    ax2.bar(bus_systems_unique, avg_bus_scores)
    ax2.set_title('Average Performance by Bus System')
    ax2.set_ylabel('Test Score')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    plt.close()


def create_comparative_renewable_plots(all_renewable_data: Dict[str, pd.DataFrame], 
                                     config: Any, num_buses: int, all_tested_models: list = None):
    return safe_plot_operation(_create_comparative_renewable_plots_impl, all_renewable_data, config, num_buses, all_tested_models)

def _create_comparative_renewable_plots_impl(all_renewable_data: Dict[str, pd.DataFrame], 
                                           config: Any, num_buses: int, all_tested_models: list = None):
    """
    Create IMPROVED comparative renewable impact plots using box plots for discrete renewable fractions.
    All models shown on same plot for direct comparison with consistent Y-axis.
    
    Creates two types of plots:
    1. Individual metric plots (all models together with box plots)
    2. Combined overview (2x2 grid of all MOOPF metrics)
    
    Args:
        all_renewable_data: Dictionary mapping model_name -> renewable_impact_dataframe
        config: Configuration object
        num_buses: Number of buses in the system
        all_tested_models: List of all models tested (used for filtering physics models)
    """
    # Only include physics-informed models that have renewable data
    if not all_renewable_data:
        print(f"No physics-informed models with renewable data for {num_buses}-bus system")
        return
    
    # Filter to only physics-informed models
    from config import Config
    physics_models = [model for model in all_renewable_data.keys() 
                     if Config.is_physics_informed(model)]
    
    if not physics_models:
        print(f"No physics-informed models to plot renewable impacts for {num_buses}-bus system")
        return
    
    # Create comparison directory at bus level
    bus_dir = os.path.join(config.EVALUATION_DIR, f"{num_buses}bus")
    os.makedirs(bus_dir, exist_ok=True)
    
    # Metrics to compare - each will be a separate plot
    metrics = {
        'normalized_carbon_emissions': 'Carbon Emissions',
        'voltage_deviation': 'Voltage Deviation',
        'power_loss': 'Power Loss', 
        'power_flow': 'Power Flow'
    }
    
    # Get unique renewable fractions (discrete values like 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    all_fracs = set()
    for data_df in all_renewable_data.values():
        if 'renewable_fraction' in data_df.columns:
            all_fracs.update(data_df['renewable_fraction'].unique())
    unique_fracs = sorted(list(all_fracs))
    
    # Professional color palette for models (matching data_profile_story quality)
    colors = plt.cm.Set2(np.linspace(0, 1, len(physics_models)))
    trend_color = '#d62728'  # Professional red for trends
    
    # ============================================================================
    # TYPE 1: Individual metric plots (all models together with box plots)
    # ============================================================================
    for metric, metric_label in metrics.items():
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data for box plots: group by renewable fraction and model
        plot_data = []
        plot_positions = []
        plot_labels = []
        plot_colors_list = []
        
        # Position offset for each model at each renewable fraction
        width = 0.15  # Width of each box
        model_offset = np.linspace(-width * (len(physics_models) - 1) / 2, 
                                   width * (len(physics_models) - 1) / 2, 
                                   len(physics_models))
        
        for frac_idx, frac in enumerate(unique_fracs):
            for model_idx, model_name in enumerate(physics_models):
                data_df = all_renewable_data[model_name]
                if metric in data_df.columns:
                    # Get all values for this model at this renewable fraction
                    mask = (data_df['renewable_fraction'] == frac)
                    values = data_df.loc[mask, metric].dropna().values
                    
                    if len(values) > 0:
                        plot_data.append(values)
                        plot_positions.append(frac_idx + model_offset[model_idx])
                        plot_labels.append(model_name if frac_idx == 0 else '')  # Label only first time
                        plot_colors_list.append(colors[model_idx])
        
        # Create professional box plots
        if plot_data:
            bp = ax.boxplot(plot_data, positions=plot_positions, widths=width*0.8,
                           patch_artist=True, showmeans=True, meanline=False,
                           medianprops=dict(color='white', linewidth=2),
                           meanprops=dict(marker='D', markerfacecolor='gold', 
                                        markeredgecolor='black', markersize=7, markeredgewidth=1),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5),
                           flierprops=dict(marker='o', markersize=5, alpha=0.5))
            
            # Color the boxes professionally
            for patch, color in zip(bp['boxes'], plot_colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.5)
        
        # Add trend lines for each model (using median values)
        for model_idx, model_name in enumerate(physics_models):
            data_df = all_renewable_data[model_name]
            if metric in data_df.columns:
                medians = []
                x_vals = []
                for frac in unique_fracs:
                    mask = (data_df['renewable_fraction'] == frac)
                    values = data_df.loc[mask, metric].dropna().values
                    if len(values) > 0:
                        medians.append(np.median(values))
                        x_vals.append(frac)
                
                if len(x_vals) > 1:
                    try:
                        z = np.polyfit(x_vals, medians, 1)
                        p = np.poly1d(z)
                        x_sorted = np.sort(x_vals)
                        ax.plot(x_sorted, p(x_sorted), '--', linewidth=3, 
                               color=colors[model_idx], alpha=0.9, label=model_name,
                               zorder=10)
                    except:
                        pass
        
        # Professional formatting (matching data_profile_story style)
        ax.set_xlabel('Renewable Energy Fraction', fontsize=13, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=13, fontweight='bold')
        ax.set_title(f'{metric_label} vs Renewable Fraction - {num_buses}-bus System', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(unique_fracs)
        ax.set_xticklabels([f'{f:.1f}' for f in unique_fracs], fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.legend(loc='best', fontsize=10, framealpha=0.95, edgecolor='black', 
                 frameon=True, ncol=2 if len(physics_models) > 3 else 1)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
        ax.set_axisbelow(True)  # Grid behind plots
        
        plt.tight_layout()
        
        # Save individual metric plot with professional quality
        clean_metric_name = metric.replace('normalized_', '').replace('_', '')[:6]
        save_path = os.path.join(bus_dir, f"ri_{clean_metric_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
    
    # ============================================================================
    # TYPE 2: Combined overview (2x2 grid of all MOOPF metrics)
    # ============================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle(f'MOOPF Objectives vs Renewable Fraction - {num_buses}-bus System', 
                fontsize=16, fontweight='bold')
    
    for idx, (metric, metric_label) in enumerate(metrics.items()):
        ax = axes[idx]
        
        # Prepare data for box plots
        plot_data = []
        plot_positions = []
        plot_colors_list = []
        
        width = 0.15
        model_offset = np.linspace(-width * (len(physics_models) - 1) / 2, 
                                   width * (len(physics_models) - 1) / 2, 
                                   len(physics_models))
        
        for frac_idx, frac in enumerate(unique_fracs):
            for model_idx, model_name in enumerate(physics_models):
                data_df = all_renewable_data[model_name]
                if metric in data_df.columns:
                    mask = (data_df['renewable_fraction'] == frac)
                    values = data_df.loc[mask, metric].dropna().values
                    if len(values) > 0:
                        plot_data.append(values)
                        plot_positions.append(frac_idx + model_offset[model_idx])
                        plot_colors_list.append(colors[model_idx])
        
        # Create professional box plots
        if plot_data:
            bp = ax.boxplot(plot_data, positions=plot_positions, widths=width*0.8,
                           patch_artist=True, showmeans=True, meanline=False,
                           medianprops=dict(color='white', linewidth=2),
                           meanprops=dict(marker='D', markerfacecolor='gold', 
                                        markeredgecolor='black', markersize=6, markeredgewidth=1),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5),
                           flierprops=dict(marker='o', markersize=4, alpha=0.5))
            
            # Color the boxes professionally
            for patch, color in zip(bp['boxes'], plot_colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.5)
        
        # Add trend lines for each model
        for model_idx, model_name in enumerate(physics_models):
            data_df = all_renewable_data[model_name]
            if metric in data_df.columns:
                medians = []
                x_vals = []
                for frac in unique_fracs:
                    mask = (data_df['renewable_fraction'] == frac)
                    values = data_df.loc[mask, metric].dropna().values
                    if len(values) > 0:
                        medians.append(np.median(values))
                        x_vals.append(frac)
                
                if len(x_vals) > 1:
                    try:
                        z = np.polyfit(x_vals, medians, 1)
                        p = np.poly1d(z)
                        x_sorted = np.sort(x_vals)
                        ax.plot(x_sorted, p(x_sorted), '--', linewidth=2.5, 
                               color=colors[model_idx], alpha=0.9, zorder=10)
                    except:
                        pass
        
        # Professional formatting
        ax.set_xlabel('Renewable Energy Fraction', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=11, fontweight='bold')
        ax.set_title(metric_label, fontsize=12, fontweight='bold', pad=12)
        ax.set_xticks(unique_fracs)
        ax.set_xticklabels([f'{f:.1f}' for f in unique_fracs], fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
        ax.set_axisbelow(True)  # Grid behind plots
    
    # Add professional legend only once
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], alpha=0.7, edgecolor='black', 
                            linewidth=1.5, label=physics_models[i]) 
                       for i in range(len(physics_models))]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
              ncol=len(physics_models), fontsize=11, framealpha=0.95, 
              edgecolor='black', frameon=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    # Save combined overview with professional quality
    save_path = os.path.join(bus_dir, "ri_moopf_overview.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
        


def create_comparative_convergence_plot(all_convergence_data: Dict[str, list], 
                                      config: Any, num_buses: int):
    return safe_plot_operation(_create_comparative_convergence_plot_impl, all_convergence_data, config, num_buses)

def _create_comparative_convergence_plot_impl(all_convergence_data: Dict[str, list], 
                                            config: Any, num_buses: int):
    """
    Create comparative convergence plot for all models in a bus system.
    Single plot with multiple convergence curves and legend.
    
    Args:
        all_convergence_data: Dictionary mapping model_name -> convergence_history
        config: Configuration object  
        num_buses: Number of buses in the system
    """
    if not all_convergence_data:
        print(f"No convergence data to plot for {num_buses}-bus system")
        return
    
    # Create comparison directory at bus level
    bus_dir = os.path.join(config.EVALUATION_DIR, f"{num_buses}bus")
    os.makedirs(bus_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Plot convergence curve for each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_convergence_data)))
    
    for idx, (model_name, history) in enumerate(all_convergence_data.items()):
        if history:  # Check if history is not empty
            iterations = list(range(1, len(history) + 1))
            plt.plot(iterations, history, marker='o', linewidth=2, markersize=4, 
                    label=f'{model_name}', alpha=0.8, color=colors[idx])
    
    plt.title(f'MoSOA Convergence Comparison - {num_buses}-bus System', fontsize=16)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Objective Score (Lower is Better)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale often helps with convergence visualization
    
    # Save the comparison plot
    save_path = os.path.join(bus_dir, "convergence.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
