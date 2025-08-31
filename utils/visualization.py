"""
Visualization utilities for power system machine learning models.
Contains all plotting and visualization functions used in training and evaluation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any


def plot_training_history(history: Dict[str, list], model_name: str, config: Any, 
                         num_buses: int, is_physics_informed: bool = True):
    """Plots and saves the training history for the best model."""
    
    if is_physics_informed:
        # Physics-informed models: Show 4 plots with physics violations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History for {model_name} (Physics-Informed)', fontsize=16)

        # Plot total loss
        axes[0, 0].plot(history['train_total_loss'], label='Train')
        axes[0, 0].plot(history['val_total_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot MSE
        axes[0, 1].plot(history['train_mse'], label='Train')
        axes[0, 1].plot(history['val_mse'], label='Validation')
        axes[0, 1].set_title('MSE Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot power violation
        axes[1, 0].plot(history['train_power_violation'], label='Train')
        axes[1, 0].plot(history['val_power_violation'], label='Validation')
        axes[1, 0].set_title('Power Balance Violation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Violation')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot voltage violation
        axes[1, 1].plot(history['train_voltage_violation'], label='Train')
        axes[1, 1].plot(history['val_voltage_violation'], label='Validation')
        axes[1, 1].set_title('Voltage Violation')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Violation')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
    else:
        # Non-physics models: Show 4 plots with training metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History for {model_name} (Non-Physics)', fontsize=16)

        # Plot MSE (main metric)
        axes[0, 0].plot(history['train_mse'], label='Train', linewidth=2)
        axes[0, 0].plot(history['val_mse'], label='Validation', linewidth=2)
        axes[0, 0].set_title('MSE Loss (Primary Metric)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot RMSE (derived metric)
        train_rmse = [mse**0.5 for mse in history['train_mse']]
        val_rmse = [mse**0.5 for mse in history['val_mse']]
        axes[0, 1].plot(train_rmse, label='Train')
        axes[0, 1].plot(val_rmse, label='Validation')
        axes[0, 1].set_title('RMSE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot learning rate progression (if available) or loss smoothness
        epochs = list(range(1, len(history['train_mse']) + 1))
        axes[1, 0].plot(epochs, history['train_mse'], alpha=0.7, label='Train MSE')
        axes[1, 0].plot(epochs, history['val_mse'], alpha=0.7, label='Val MSE')
        axes[1, 0].set_title('Loss Progression')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_yscale('log')  # Log scale to better see convergence
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot training vs validation gap
        train_val_gap = [abs(t - v) for t, v in zip(history['train_mse'], history['val_mse'])]
        axes[1, 1].plot(epochs, train_val_gap, color='red', label='Train-Val Gap')
        axes[1, 1].set_title('Generalization Gap (|Train - Val|)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MSE Difference')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save in the new directory structure
    save_path = config.get_training_history_path(num_buses, model_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_renewable_impact(data_df: pd.DataFrame, metric_name: str, y_label: str, 
                         title: str, config: Any, num_buses: int, model_name: str):
    """Plots renewable impact for the best model."""
    x_col = 'renewable_fraction'
    y_col = metric_name
    
    plt.figure(figsize=(12, 8))
    x, y = data_df[x_col], data_df[y_col]
    plt.scatter(x, y, alpha=0.6, label='Test Scenario')

    # Fit trendline
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x.sort_values(), p(x.sort_values()), "r--", linewidth=2, 
             label=f'Trendline (y={z[0]:.2f}x + {z[1]:.2f})')

    plt.title(title, fontsize=16)
    plt.xlabel('Renewable Energy Fraction', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)

    # Save in the new directory structure
    save_dir = config.get_renewable_impacts_dir(num_buses, model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{metric_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_convergence(history: list, model_name: str, config: Any, num_buses: int):
    """Plots the convergence curve of the MoSOA algorithm."""
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
    """Plot all renewable impact metrics for a physics-informed model."""
    if renewable_impact_data.empty:
        print(f"ℹ️  No renewable impact data to plot for {model_name}")
        return
        
    # Update metrics dictionary to match column names in renewable_impact_data
    metrics = {
        'normalized_carbon_emissions': 'Carbon Emissions',
        'voltage_deviation': 'Voltage Deviation',          # Changed from normalized_voltage_deviation
        'power_loss': 'Power Loss',                        # Changed from normalized_power_loss
        'power_flow': 'Power Flow'                         # NEW: Added power flow metric
    }
    
    for metric, label in metrics.items():
        try:
            plot_renewable_impact(
                renewable_impact_data,
                metric_name=metric,
                y_label=label,
                title=f'Impact of Renewable Fraction on {label}',
                config=config,
                num_buses=num_buses,
                model_name=model_name
            )
        except KeyError as e:
            print(f"Warning: Could not plot {metric} due to missing column: {e}")
            continue


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
    """
    Create comparative renewable impact plots for physics-informed models only.
    Non-physics models are excluded from renewable impact analysis as these 
    metrics are not applicable to them. Subplot layout dynamically adjusts 
    to the number of physics-informed models.
    
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
    
    # Use only physics-informed models
    model_names = physics_models
    num_models = len(model_names)
    
    # Calculate optimal subplot layout based on number of physics models
    def calculate_subplot_layout(n_models):
        """Calculate optimal rows and columns for n_models subplots"""
        if n_models == 1:
            return 1, 1
        elif n_models == 2:
            return 1, 2
        elif n_models <= 4:
            return 2, 2
        elif n_models <= 6:
            return 2, 3
        elif n_models <= 9:
            return 3, 3
        else:
            # For more than 9 models, use 3x4 grid and limit to 12
            return 3, 4
    
    nrows, ncols = calculate_subplot_layout(num_models)
    
    # Create a separate plot for each metric
    for metric, metric_label in metrics.items():
        # Dynamic figure size based on layout
        fig_width = ncols * 5  # 5 inches per column
        fig_height = nrows * 4  # 4 inches per row
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
        fig.suptitle(f'{metric_label} vs Renewable Fraction - {num_buses}-bus System', fontsize=16)
        
        # Handle different subplot layouts
        if num_models == 1:
            axes = [axes]  # Make it iterable
        elif nrows == 1 or ncols == 1:
            axes = axes.flatten() if num_models > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for idx, model_name in enumerate(model_names):
            ax = axes[idx]
            data_df = all_renewable_data[model_name]
            
            if metric in data_df.columns:
                x = data_df['renewable_fraction']
                y = data_df[metric]
                
                # Plot scatter points
                ax.scatter(x, y, alpha=0.7, s=50, label='Data Points')
                
                # Fit and plot trendline
                if len(x) > 1:  # Need at least 2 points for trendline
                    try:
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        x_sorted = np.sort(x)
                        ax.plot(x_sorted, p(x_sorted), 'r--', linewidth=2, 
                               label=f'Trend: y={z[0]:.3f}x + {z[1]:.3f}')
                    except np.linalg.LinAlgError:
                        print(f"Warning: Could not fit trendline for {model_name} {metric}")
                
                ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Renewable Energy Fraction', fontsize=10)
                ax.set_ylabel(metric_label, fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                # If metric not available in physics model, show message
                ax.text(0.5, 0.5, f'{metric}\nnot available', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        
        # Remove any unused subplots (only if we have more subplot spaces than models)
        total_subplots = nrows * ncols
        if total_subplots > num_models:
            for idx in range(num_models, total_subplots):
                axes[idx].remove()
        
        plt.tight_layout()
        
        # Save the plot - clean metric name for filename
        clean_metric_name = metric.replace('normalized_', '')
        save_path = os.path.join(bus_dir, f"renewable_impact_{clean_metric_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Renewable impact comparison ({metric_label}) saved to: {save_path}")


def create_comparative_convergence_plot(all_convergence_data: Dict[str, list], 
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
    save_path = os.path.join(bus_dir, "convergence_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Convergence comparison saved to: {save_path}")
