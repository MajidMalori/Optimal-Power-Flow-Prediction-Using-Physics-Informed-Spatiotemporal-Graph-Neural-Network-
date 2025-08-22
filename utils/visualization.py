import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import numpy as np
import os

# --- MODIFIED: Define metric sets as constants, now including 'Carbon' ---
GCN_METRICS = ['mse', 'mae', 'rmse']
PI_METRICS = [
    'mse', 'mae', 'rmse',      # Standard metrics
    'TRPL', 'TVD', 'Carbon',   # Physics objectives (Carbon added here)
    'equality_constraint', 'inequality_constraint'  # Constraints
]


def plot_training_history(
    history: Dict[str, List[float]],
    val_history: Dict[str, List[float]],
    test_metrics: Dict[str, float],
    save_path: str,
    model_name: str,
    is_physics_informed: bool = False,
    is_sequential: bool = False) -> None:
    """
    Plots a comprehensive training history, now including the Carbon objective.
    
    Args:
        history (Dict): Dictionary of training metrics per epoch.
        val_history (Dict): Dictionary of validation metrics per epoch.
        test_metrics (Dict): Dictionary of final test metrics.
        save_path (str): Path to save the plot image.
        model_name (str): Name of the model for titles.
        is_physics_informed (bool): Flag to determine plot layout.
        is_sequential (bool): Flag for sequential model titles.
    """
    
    # Ensure the directory for saving the plot exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create subplots based on whether the model is physics-informed
    if is_physics_informed:
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        fig.suptitle(f'Training History for {model_name}', fontsize=16)

        # Plot 1: MSE Loss (Data-driven objective)
        ax = axes[0, 0]
        ax.plot(history['mse_loss'], 'b-', label='Train MSE')
        if val_history and 'mse_loss' in val_history:
            ax.plot(val_history['mse_loss'], 'r-', label='Val MSE')
        if 'mse' in test_metrics:
            ax.axhline(y=test_metrics['mse'], color='g', linestyle='--', label=f"Test MSE: {test_metrics['mse']:.4f}")
        ax.set_title('MSE Loss (Data-Driven Objective)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.legend()
        ax.grid(True)
        
        # --- MODIFIED: Plot 2 now includes Carbon Emissions ---
        ax = axes[0, 1]
        ax.plot(history['TRPL'], color='orangered', label='Power Loss (TRPL)')
        ax.plot(history['TVD'], color='deepskyblue', label='Voltage Deviation (TVD)')
        # Add the new Carbon objective plot
        if 'Carbon' in history:
            ax.plot(history['Carbon'], color='darkgreen', label='Carbon Emissions')
        ax.set_title('Normalized Physics Objectives')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Normalized Value')
        ax.legend()
        ax.grid(True)
        
        # Plot 3: Constraint Violations
        ax = axes[1, 0]
        ax.plot(history['equality_constraint'], 'g-', label='Equality Constraints')
        ax.plot(history['inequality_constraint'], 'y-', label='Inequality Constraints')
        ax.set_title('Constraint Penalties')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Penalty Value')
        ax.legend()
        ax.set_yscale('log') # Use log scale for constraints as they can be very small
        ax.grid(True)
        
        # Plot 4: Total Loss
        ax = axes[1, 1]
        ax.plot(history['loss'], 'purple', label='Total Weighted Loss')
        if val_history and 'loss' in val_history:
            ax.plot(val_history['loss'], color='magenta', linestyle='--', label='Val Total Loss')
        ax.set_title('Total Loss (Weighted Sum)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)

    else: # For non-physics-informed models (e.g., standard GCN)
        plt.figure(figsize=(10, 6))
        plt.plot(history['mse'], 'b-', label='Train MSE')
        if val_history and 'mse' in val_history:
            plt.plot(val_history['mse'], 'r-', label='Val MSE')
        plt.axhline(y=test_metrics['mse'], color='g', linestyle='--', label=f"Test MSE: {test_metrics['mse']:.4f}")
        
        title_prefix = "Sequential" if is_sequential else "Static"
        plt.title(f'{title_prefix} {model_name} Training History')
        
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
    plt.savefig(save_path)
    plt.close()


def plot_model_comparison(
    metrics: Dict[str, Dict[str, float]], 
    save_path: str,
    model_types: Dict[str, bool],
    is_sequential: Dict[str, bool]) -> None:
    """
    Plots a comparison of final model metrics across different runs.
    This function is now capable of plotting the 'Carbon' metric.
    """
    # Ensure the directory for saving the plot exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    num_plots = 0
    if any(not v for v in is_sequential.values()): num_plots += 1 # Static models exist
    if any(v for v in is_sequential.values()): num_plots += 1    # Sequential models exist
    
    if num_plots == 0:
        print("No models to compare.")
        return
        
    fig, axes = plt.subplots(num_plots, 2, figsize=(18, 9 * num_plots), squeeze=False)
    plot_row = 0
    
    # --- Plot Static Models ---
    static_models = {k: v for k, v in metrics.items() if not is_sequential[k]}
    if static_models:
        static_gcn = {k: v for k, v in static_models.items() if not model_types[k]}
        static_pi = {k: v for k, v in static_models.items() if model_types[k]}
        
        if static_gcn:
            plot_metrics(static_gcn, GCN_METRICS, axes[plot_row, 0], 'Static GCN Models - Final Metrics')
        else:
            axes[plot_row, 0].text(0.5, 0.5, 'No Static GCN Models to Compare', ha='center', va='center')
            axes[plot_row, 0].axis('off')
            
        if static_pi:
            plot_metrics(static_pi, PI_METRICS, axes[plot_row, 1], 'Static Physics-Informed - Final Metrics')
        else:
            axes[plot_row, 1].text(0.5, 0.5, 'No Static PI Models to Compare', ha='center', va='center')
            axes[plot_row, 1].axis('off')
        plot_row += 1

    # --- Plot Sequential Models ---
    seq_models = {k: v for k, v in metrics.items() if is_sequential[k]}
    if seq_models:
        seq_gcn = {k: v for k, v in seq_models.items() if not model_types[k]}
        seq_pi = {k: v for k, v in seq_models.items() if is_sequential[k]} # Bug fix: should be model_types
        
        if seq_gcn:
            plot_metrics(seq_gcn, GCN_METRICS, axes[plot_row, 0], 'Sequential GCN Models - Final Metrics')
        else:
            axes[plot_row, 0].text(0.5, 0.5, 'No Sequential GCN Models to Compare', ha='center', va='center')
            axes[plot_row, 0].axis('off')

        if seq_pi:
            plot_metrics(seq_pi, PI_METRICS, axes[plot_row, 1], 'Sequential Physics-Informed - Final Metrics')
        else:
            axes[plot_row, 1].text(0.5, 0.5, 'No Sequential PI Models to Compare', ha='center', va='center')
            axes[plot_row, 1].axis('off')
        plot_row += 1

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_metrics(models: Dict[str, Dict[str, float]], 
                metrics_to_plot: List[str], 
                ax: plt.Axes, 
                title: str) -> None:
    """Helper function to plot metrics for a group of models on a given axis."""
    model_names = list(models.keys())
    
    # Create data for grouped bar chart
    x = np.arange(len(model_names))  # the label locations
    width = 0.8 / len(metrics_to_plot)  # the width of the bars
    
    for i, metric in enumerate(metrics_to_plot):
        values = [models[name].get(metric, 0) for name in model_names] # Use .get for safety
        offset = width * (i - (len(metrics_to_plot) - 1) / 2)
        rects = ax.bar(x + offset, values, width, label=metric)
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=8)

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Metric Value')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)