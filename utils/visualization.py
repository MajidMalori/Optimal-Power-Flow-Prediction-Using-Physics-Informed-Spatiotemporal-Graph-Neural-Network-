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

def plot_training_history(history: Dict[str, list], model_name: str, config: Any, 
                         num_buses: int, is_physics_informed: bool = True):
    """Plot training history with automatic error handling."""
    try:
        plt.clf()
        plt.close('all')
        
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
            # Convert to numpy array (handles tensors with gradients)
            train_loss = np.array([x.detach().item() if hasattr(x, 'detach') else (x.item() if hasattr(x, 'item') else x) 
                                  for x in history['train_total_loss']])
            axes[1, 0].plot(train_loss, label='Train', linewidth=2, color='black')
            axes[1, 0].set_title('Total Weighted Loss', fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
        # Row 2, Col 2: Learned Task Weights (Kendall's Method)
        if is_physics_informed and 'train_weights' in history and history['train_weights']:
            # Filter out None values and convert to numpy array
            weights = np.array([w for w in history['train_weights'] if w is not None])
            if weights.ndim == 2 and weights.shape[1] == 3:
                axes[1, 1].plot(weights[:, 0], label='w_data (L1)', linewidth=2)
                axes[1, 1].plot(weights[:, 1], label='w_phys (L2)', linewidth=2)
                axes[1, 1].plot(weights[:, 2], label='w_safe (L3)', linewidth=2)
                axes[1, 1].set_title('Learned Task Weights (Kendall)', fontweight='bold')
                axes[1, 1].set_ylabel('Precision Weight (w)')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')
                
        # Row 2, Col 3: Generalization Gap
        if 'train_mse' in history and 'val_mse' in history:
            # Convert to numpy arrays (handles tensors with gradients)
            train_mse = np.array([x.detach().item() if hasattr(x, 'detach') else (x.item() if hasattr(x, 'item') else x) 
                                 for x in history['train_mse']])
            val_mse = np.array([x.detach().item() if hasattr(x, 'detach') else (x.item() if hasattr(x, 'item') else x) 
                               for x in history['val_mse']])
            gap = np.abs(train_mse - val_mse)
            axes[1, 2].plot(gap, color='purple', linewidth=2)
            axes[1, 2].set_title('Generalization Gap', fontweight='bold')
            axes[1, 2].grid(True, alpha=0.3)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = config.get_training_history_path(num_buses, model_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close('all')
        
    except Exception as e:
        plt.close('all')
        print(f"Warning: Training history plotting failed: {e}")

def plot_convergence(history: list, model_name: str, config: Any, num_buses: int):
    """Plot convergence curve with automatic error handling."""
    try:
        plt.clf()
        plt.close('all')
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(history) + 1), history, 'b-', marker='o')
        plt.title(f'Convergence - {model_name}', fontsize=14)
        plt.xlabel('Iteration')
        plt.ylabel('Best Loss')
        plt.grid(True)
        
        save_path = config.get_convergence_plot_path(num_buses, model_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close('all')
        
    except Exception as e:
        plt.close('all')
        print(f"Warning: Convergence plotting failed: {e}")

def plot_all_renewable_impacts(renewable_impact_data: pd.DataFrame, config: Any, 
                               num_buses: int, model_name: str):
    """Plot renewable impacts with automatic error handling."""
    try:
        plt.clf()
        plt.close('all')
        
        if renewable_impact_data.empty:
            return
        
        metrics = {
            'carbon_emissions': 'Carbon Emissions',
            'voltage_deviation': 'Voltage Deviation',
            'power_loss': 'Power Loss'
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Renewable Impact Analysis - {model_name}', fontsize=16, fontweight='bold')
        
        unique_fracs = sorted(renewable_impact_data['renewable_fraction'].unique())
        
        for idx, (col, label) in enumerate(metrics.items()):
            ax = axes[idx]
            if col in renewable_impact_data.columns:
                data = [renewable_impact_data[renewable_impact_data['renewable_fraction'] == f][col].values for f in unique_fracs]
                ax.boxplot(data, positions=range(len(unique_fracs)))
                ax.set_title(label)
                ax.set_xticklabels([f'{f:.1f}' for f in unique_fracs])
                ax.set_xlabel('Renewable Fraction')
        
        plt.tight_layout()
        model_dir = config.get_model_eval_dir(num_buses, model_name)
        os.makedirs(model_dir, exist_ok=True)
        plt.savefig(os.path.join(model_dir, 'ri_combined.png'), dpi=300)
        plt.close('all')
        
    except Exception as e:
        plt.close('all')
        print(f"Warning: Renewable impacts plotting failed: {e}")

def create_model_comparison_plot(all_results: list, save_path: str = None):
    """Create comprehensive model comparison plot."""
    try:
        plt.clf()
        plt.close('all')
        
        if not all_results:
            print("No results to plot")
            return
        
        # Filter successful results
        successful_results = [r for r in all_results if r['final_test_score'] != float('inf')]
        
        if not successful_results:
            print("No successful results to plot")
            return
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract data
        model_names = [r['model_name'] for r in successful_results]
        bus_systems = [r['num_buses'] for r in successful_results]
        test_scores = [r['final_test_score'] for r in successful_results]
        is_physics = [r['is_physics_informed'] for r in successful_results]
        
        # Plot 1: Test Score by Model and Bus System
        unique_buses = sorted(set(bus_systems))
        x_pos = np.arange(len(successful_results))
        colors = ['#2E86AB' if pi else '#A23B72' for pi in is_physics]
        
        bars = ax1.bar(x_pos, test_scores, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Model Configuration', fontsize=11)
        ax1.set_ylabel('Test Score (MSE)', fontsize=11)
        ax1.set_title('Test Performance by Model', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"{m}\n{b}-bus" for m, b in zip(model_names, bus_systems)], 
                           rotation=45, ha='right', fontsize=8)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E86AB', edgecolor='black', label='Physics-Informed'),
            Patch(facecolor='#A23B72', edgecolor='black', label='Non-Physics')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # Plot 2: Performance vs Bus System Size
        bus_perf = {}
        for bus, score in zip(bus_systems, test_scores):
            if bus not in bus_perf:
                bus_perf[bus] = []
            bus_perf[bus].append(score)
        
        bus_means = [np.mean(bus_perf[b]) for b in sorted(bus_perf.keys())]
        bus_stds = [np.std(bus_perf[b]) if len(bus_perf[b]) > 1 else 0 for b in sorted(bus_perf.keys())]
        
        ax2.errorbar(sorted(bus_perf.keys()), bus_means, yerr=bus_stds, 
                    marker='o', markersize=10, linewidth=2, capsize=5, color='#F18F01')
        ax2.set_xlabel('Bus System Size', fontsize=11)
        ax2.set_ylabel('Average Test Score (MSE)', fontsize=11)
        ax2.set_title('Performance vs System Size', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(sorted(bus_perf.keys()))
        ax2.set_xticklabels([f'{b}-bus' for b in sorted(bus_perf.keys())])
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved: {save_path}")
        
        plt.close('all')
        
    except Exception as e:
        plt.close('all')
        print(f"Warning: Model comparison plotting failed: {e}")

def create_comparative_renewable_plots(renewable_data: dict, config: Any, num_buses: int, tested_models: list):
    """Create comparative renewable impact plots across all tested models.
    
    Generates a 2x2 subplot figure comparing how all models perform across different
    renewable penetration levels (0.0 to 1.0) for key metrics:
    - Carbon emissions (conventional generation)
    - Voltage deviation from nominal
    - System power losses
    - Prediction uncertainty
    
    Args:
        renewable_data: Dict mapping model_name to DataFrame with renewable impact data
        config: Config object with RESULTS_BASE_DIR
        num_buses: Number of buses in the system
        tested_models: List of all tested model names (for consistent ordering)
    """
    try:
        # Filter to only models with non-empty data
        models_with_data = {m: df for m, df in renewable_data.items() if not df.empty}
        
        if not models_with_data:
            return
        
        plt.clf()
        plt.close('all')
        
        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Define metrics to plot
        metrics = [
            ('carbon_proxy', 'Carbon Emissions (MW)', axes[0, 0]),
            ('voltage_deviation', 'Voltage Deviation (p.u.)', axes[0, 1]),
            ('power_loss', 'Power Loss (p.u.)', axes[1, 0]),
            ('uncertainty', 'Prediction Uncertainty', axes[1, 1])
        ]
        
        # Assign unique colors to each model
        colors = plt.cm.tab10(np.linspace(0, 1, len(models_with_data)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        for (metric_col, ylabel, ax) in metrics:
            for idx, (model_name, df) in enumerate(models_with_data.items()):
                if metric_col not in df.columns:
                    continue
                
                # Group by renewable_fraction and compute statistics
                grouped = df.groupby('renewable_fraction')[metric_col].agg(['mean', 'std', 'count'])
                fractions = grouped.index.values
                means = grouped['mean'].values
                stds = grouped['std'].values
                counts = grouped['count'].values
                
                # Calculate standard error
                stderr = stds / np.sqrt(counts)
                
                # Plot with error bars
                marker_style = markers[idx % len(markers)]
                ax.errorbar(fractions, means, yerr=stderr, 
                           label=model_name, marker=marker_style, 
                           color=colors[idx], linewidth=2.5, 
                           capsize=5, capthick=1.5, markersize=8,
                           alpha=0.8)
            
            # Customize subplot
            ax.set_xlabel('Renewable Penetration Fraction', fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(f'{ylabel.split("(")[0].strip()} Comparison', 
                        fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(-0.05, 1.05)
            
            # Add tick marks
            ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
        # Overall title
        plt.suptitle(f'Comparative Renewable Impact Analysis - {num_buses}-bus System', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save to bus-level directory
        output_dir = os.path.join(config.RESULTS_BASE_DIR, f"{num_buses}bus")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'comparative_renewable_impacts.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        
    except Exception as e:
        plt.close('all')
        print(f"Warning: Comparative renewable plots failed: {e}")

def create_comparative_convergence_plot(convergence_data: dict, config: Any, num_buses: int):
    """Create comparative convergence plot across all tested models."""
    try:
        if not convergence_data:
            return
        
        plt.clf()
        plt.close('all')
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for model_name, history in convergence_data.items():
            if history:
                ax.plot(range(1, len(history) + 1), history, marker='o', label=model_name, linewidth=2)
        
        ax.set_title(f'Convergence Comparison - {num_buses}-bus System', fontsize=14, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Best Loss', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = os.path.join(config.RESULTS_BASE_DIR, f"{num_buses}bus")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'convergence_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        
    except Exception as e:
        plt.close('all')
        print(f"Warning: Comparative convergence plot failed: {e}")
