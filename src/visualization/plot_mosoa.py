import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import yaml

def load_viz_config():
    """Load visualization settings from mosoa.yaml if exists."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "mosoa.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            return full_config.get('visualization', {})
    return {}

def set_premium_mosoa_aesthetics():
    """Set the aesthetic parameters for MoSOA plots."""
    viz_config = load_viz_config()
    dpi = viz_config.get('dpi', 300)
    
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': dpi
    })

def plot_mosoa_ranks(avg_ranks, output_path):
    """
    Generate the bar plot of average ranks across benchmark functions.
    
    Args:
        avg_ranks: pd.Series with algorithm names as index and ranks as values.
        output_path: Full path including filename to save the plot.
    """
    set_premium_mosoa_aesthetics()
    
    avg_ranks_df = avg_ranks.reset_index()
    avg_ranks_df.columns = ['Algorithm', 'Avg Rank']
    
    plt.figure(figsize=(10, 6))
    colors = ['#2ecc71' if a == 'MoSOA' else '#3498db' for a in avg_ranks_df['Algorithm']]
    
    sns.barplot(data=avg_ranks_df, x='Algorithm', y='Avg Rank', palette=colors)
    plt.title('Algorithm Comparison Ranks (F1-F23)', fontweight='bold', pad=20)
    plt.ylabel('Average Rank (Lower is Better)', fontweight='bold')
    plt.xlabel('Algorithm', fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_convergence_curves(history_dict, func_name, output_path, num_runs=10, dim=30):
    """
    Generate convergence curves for a specific benchmark function.
    
    Args:
        history_dict: Dict mapping algo_name -> list of fitness values over iterations.
        func_name: Name of the benchmark function (e.g., 'F1').
        output_path: Full path including filename to save the plot.
        num_runs: Number of runs.
        dim: Problem dimensionality.
    """
    set_premium_mosoa_aesthetics()
    
    plt.figure(figsize=(12, 7))
    
    # Load color map from config or use defaults
    viz_config = load_viz_config()
    color_map = viz_config.get('colors', {
        'MoSOA': '#E91E63',  # Magenta (Publication Style)
        'PSO':   '#FF9800',  # Bright Orange
        'GWO':   '#2980b9',  # Blue
        'GA':    '#8e44ad',  # Purple
        'TSA':   '#f1c40f',  # Yellow
        'SOA*':  '#d35400',  # Brownish Orange
        'ESOA':  '#2ecc71',  # Strong Green
        'HGSO':  '#2c3e50',  # Dark Blue-Grey
    })
    
    default_linewidth = viz_config.get('linewidth', 2.0)
    
    min_val = np.inf
    has_zero_or_negative = False

    for i, (algo, history) in enumerate(history_dict.items()):
        # Thinner linewidth for a professional publication look
        linewidth = default_linewidth
        alpha = 1.0 if algo == 'MoSOA' else 0.8
        
        # Get color from map or fallback to palette
        color = color_map.get(algo, sns.color_palette("tab10")[i % 10])
        
        history_arr = np.array(history)
        
        # Standard plot (no staircase) as requested for higher iterations
        plt.plot(range(len(history_arr)), history_arr, 
                 label=algo, linewidth=linewidth, color=color, alpha=alpha,
                 marker='o', markersize=4, markevery=max(1, len(history_arr)//20))
        
        if np.any(history_arr <= 1e-12):
            has_zero_or_negative = True
        min_val = min(min_val, np.min(history_arr))

    # Safely apply log scale
    if not has_zero_or_negative and min_val > 0:
        plt.yscale('log')
        plt.ylabel('Average Best Score (Log Scale)', fontweight='bold')
    else:
        plt.ylabel('Average Best Score (Linear Scale)', fontweight='bold')

    plt.title(f'Avg Convergence Curves for {func_name} ({num_runs} Runs, Dim={dim})', 
              fontweight='bold', pad=25)
    plt.xlabel('Iterations', fontweight='bold')
    
    # Place legend outside to the right
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', frameon=True, shadow=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_categorical_convergence(category_histories, category_name, output_path, num_runs=10, default_dim=30):
    """
    Generate a grid of convergence curves for a specific category of mathematical functions.
    category_histories: {fn_name: {algo_name: list_of_fitness_values}}
    """
    set_premium_mosoa_aesthetics()
    
    num_plots = len(category_histories)
    if num_plots == 0: return
    
    cols = 3 if num_plots > 4 else min(num_plots, 3)
    rows = int(np.ceil(num_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    viz_config = load_viz_config()
    color_map = viz_config.get('colors', {
        'MoSOA': '#E91E63', 'PSO': '#FF9800', 'GWO': '#2980b9', 
        'GA': '#8e44ad', 'TSA': '#f1c40f', 'SOA*': '#d35400', 
        'ESOA': '#2ecc71', 'HGSO': '#2c3e50'
    })
    
    # Inject Perturbation Strategy Colors
    color_map.update({
        'linear': '#34495e',
        'cosine': '#3498db',
        'quadratic': '#9b59b6',
        'exponential': '#E91E63'  # Match MoSOA Magenta
    })
    
    default_linewidth = viz_config.get('linewidth', 2.0)
    
    algorithms_seen = set()
    
    for idx, (func_name, history_dict) in enumerate(category_histories.items()):
        ax = axes[idx]
        min_val = np.inf
        has_zero_or_negative = False
        
        for i, (algo, history) in enumerate(history_dict.items()):
            algorithms_seen.add(algo)
            linewidth = 2.5 if algo in ['MoSOA', 'exponential'] else default_linewidth
            alpha = 1.0 if algo in ['MoSOA', 'exponential'] else 0.8
            color = color_map.get(algo, sns.color_palette("tab10")[i % 10])
            
            history_arr = np.array(history)
            ax.plot(range(len(history_arr)), history_arr, 
                    linewidth=linewidth, color=color, alpha=alpha,
                    marker='o', markersize=3, markevery=max(1, len(history_arr)//20))
                    
            if np.any(history_arr <= 1e-12):
                has_zero_or_negative = True
            min_val = min(min_val, np.min(history_arr))
            
        if not has_zero_or_negative and min_val > 0:
            ax.set_yscale('log')
            
        ax.set_title(func_name, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        if idx % cols == 0:
            ax.set_ylabel('Fitness')
        if idx >= num_plots - cols:
            ax.set_xlabel('Iterations')
            
    for idx in range(num_plots, len(axes)):
        fig.delaxes(axes[idx])
        
    fig.suptitle(f'{category_name} ({num_runs} Runs)', fontweight='bold', fontsize=18, y=1.02)
    
    handles = []
    labels = []
    for algo in sorted(list(algorithms_seen)):
        color = color_map.get(algo, 'black')
        alpha = 1.0 if algo in ['MoSOA', 'exponential'] else 0.8
        linewidth = 2.5 if algo in ['MoSOA', 'exponential'] else default_linewidth
        line = plt.Line2D([0], [0], color=color, linewidth=linewidth, alpha=alpha, marker='o', markersize=3)
        handles.append(line)
        labels.append(algo)
        
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -max(0.05, 0.1/rows)), ncol=min(8, len(labels)), frameon=True, shadow=True)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_perturbation_ablation(df, output_path, title_suffix=""):
    """
    Plot bar comparison of different perturbation strategies.
    
    Args:
        df: DataFrame with columns ['Strategy', 'Function', 'Mean', 'Std']
        output_path: Path to save the plot.
        title_suffix: Optional suffix for the title (e.g., "Unimodal").
    """
    set_premium_mosoa_aesthetics()
    
    plt.figure(figsize=(15, 8))
    
    # We use a human-selected palette for strategy distinctness
    strategy_colors = {
        'linear': '#34495e',
        'cosine': '#3498db',
        'quadratic': '#9b59b6',
        'exponential': '#2ecc71'  # Our champion
    }
    
    sns.barplot(data=df, x='Function', y='Mean', hue='Strategy', palette=strategy_colors)
    
    # Check if any values are too close to zero for Log Scale
    # We use a small epsilon for values that are "conceptually" zero but found by optimizer
    min_val = df['Mean'].min()
    if min_val > 1e-50:
        plt.yscale('log')
        plt.ylabel('Mean Fitness (Log Scale)', fontweight='bold')
    else:
        plt.ylabel('Mean Fitness (Linear Scale)', fontweight='bold')

    title = 'Perturbation Strategy Comparison (Ablation Study)'
    if title_suffix:
        title += f' - {title_suffix}'
    
    plt.title(title, fontweight='bold', pad=25)
    plt.xlabel('Benchmark Function', fontweight='bold')
    
    # Legend outside to the right
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', title='Strategy', frameon=True, shadow=True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_perturbation_convergence(history_dict, func_name, output_path, num_runs=10):
    """
    Generate convergence curves comparing different perturbation strategies for MoSOA.
    """
    set_premium_mosoa_aesthetics()
    
    plt.figure(figsize=(10, 6))
    
    strategy_colors = {
        'linear': '#34495e',
        'cosine': '#3498db',
        'quadratic': '#9b59b6',
        'exponential': '#E91E63'  # Match MoSOA Magenta
    }
    
    min_val = np.inf
    has_zero_or_negative = False
    
    for strat, history in history_dict.items():
        linewidth = 2.5 if strat == 'exponential' else 2.0
        color = strategy_colors.get(strat, '#7f8c8d')
        
        history_arr = np.array(history)
        plt.plot(range(len(history_arr)), history_arr, 
                 label=f"MoSOA ({strat})", linewidth=linewidth, color=color,
                 marker='o', markersize=4, markevery=max(1, len(history_arr)//20))
        
        if np.any(history_arr <= 1e-12):
            has_zero_or_negative = True
        min_val = min(min_val, np.min(history_arr))

    if not has_zero_or_negative and min_val > 0:
        plt.yscale('log')
        plt.ylabel('Avg Best Score (Log Scale)', fontweight='bold')
    else:
        plt.ylabel('Avg Best Score (Linear Scale)', fontweight='bold')

    plt.title(f'Perturbation Strategy Convergence: {func_name} ({num_runs} Runs)', 
              fontweight='bold', pad=20)
    plt.xlabel('Iterations', fontweight='bold')
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
def plot_3d_landscape(fn_lambda, bounds, title, output_path, resolution=100, target_dim=2):
    """
    Generates a high-fidelity 3D surface plot for a given function.
    target_dim: Total dimensions the function expects (pads with zeros if > 2)
    """
    set_premium_mosoa_aesthetics()
    
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            # Create a point of target_dim size
            point = np.zeros(target_dim)
            point[0] = X[i, j]
            point[1] = Y[i, j]
            Z[i, j] = fn_lambda(point)
            
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use viridis with a slight transparency for better mesh visibility
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                          antialiased=True, alpha=0.9,
                          linewidth=0.1, edgecolors='gray', shade=True)
    
    # Add contours on the floor for better depth perception
    offset = np.min(Z) - (np.max(Z)-np.min(Z))*0.15
    ax.contourf(X, Y, Z, zdir='z', offset=offset, cmap='viridis', alpha=0.3)
    
    ax.set_title(title, fontweight='bold', pad=30, fontsize=16)
    ax.set_xlabel('X1', fontweight='bold', labelpad=15)
    ax.set_ylabel('X2', fontweight='bold', labelpad=15)
    ax.set_zlabel('Objective/Loss', fontweight='bold', labelpad=15)
    
    # Premium lighting and view angle
    ax.view_init(elev=35, azim=-45)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, label='Objective Value')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_categorical_landscapes_grid(fn_list, names, bounds_list, title, output_path, target_dims, resolution=60):
    """
    Groups 3D landscapes into a grid image (max 4-6 panels).
    fn_list: List of function lambdas
    names: List of strings (F1, F2...)
    bounds_list: List of tuples (min, max)
    target_dims: List of integers (2, 30...)
    """
    set_premium_mosoa_aesthetics()
    
    num_fns = len(fn_list)
    cols = 2
    rows = (num_fns + 1) // 2
    
    fig = plt.figure(figsize=(14, 6 * rows))
    plt.suptitle(title, fontweight='bold', fontsize=20, y=0.98)
    
    for idx, (fn, name, bounds, t_dim) in enumerate(zip(fn_list, names, bounds_list, target_dims)):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        
        x = np.linspace(bounds[0], bounds[1], resolution)
        y = np.linspace(bounds[0], bounds[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(resolution):
            for j in range(resolution):
                p = np.zeros(t_dim)
                p[0] = X[i, j]
                p[1] = Y[i, j]
                Z[i, j] = fn(p)
                
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', antialiased=True, alpha=0.9, linewidth=0.1, edgecolors='gray')
        
        # Consistent depth cue
        offset = np.min(Z) - (np.max(Z)-np.min(Z))*0.15
        ax.contourf(X, Y, Z, zdir='z', offset=offset, cmap='viridis', alpha=0.3)
        
        ax.set_title(f"Landscape: {name}", fontweight='bold', pad=10, fontsize=14)
        ax.set_xlabel('X1', fontsize=10)
        ax.set_ylabel('X2', fontsize=10)
        ax.view_init(elev=30, azim=-45)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight') # Lower DPI for grids to save memory/space
    plt.close()

