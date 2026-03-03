"""
Professional Topology Event Visualization
Shows when and where contingencies (line trips) occurred in the dataset.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

# Professional style
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
})

def plot_topology_events(data_dir: str, case_name: str, output_path: str, config: dict = None) -> str:
    """
    Visualize topology changes (contingencies) over time.
    """
    import glob
    from constants import FeatureIndices
    
    # Load feature files to detect degree changes (contingencies)
    pattern = os.path.join(data_dir, f'{case_name}_features_frac*.npy')
    files = sorted(glob.glob(pattern))
    
    if not files:
        return None
        
    all_active_lines = []
    
    for f in files:
        features = np.load(f) # [T, N, F]
        # Degree is at index 10. Sum of degrees / 2 = Number of lines
        degrees = features[:, :, FeatureIndices.DEGREE]
        line_counts = np.sum(degrees, axis=1) / 2
        all_active_lines.append(line_counts)
    
    if not all_active_lines:
        return None
        
    line_series = np.concatenate(all_active_lines)
    t_total = len(line_series)
    max_lines = np.max(line_series)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Plot active lines
    ax.step(range(t_total), line_series, where='post', color='#c0392b', linewidth=2, label='Active Lines')
    
    # Highlight trips
    trips = np.where(line_series < max_lines)[0]
    if len(trips) > 0:
        ax.fill_between(range(t_total), line_series, max_lines, 
                        where=(line_series < max_lines), color='#e74c3c', alpha=0.3, label='Line Outage')
    
    ax.set_ylim(max_lines - 2.5, max_lines + 0.5)
    ax.set_xlabel('Timestep (Chronological)', fontweight='bold')
    ax.set_ylabel('Active Transmission Lines', fontweight='bold')
    ax.set_title(f'Topology Integrity & Contingency Events — {case_name.upper()}', fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='lower left', frameon=True, shadow=True)
    
    # Text annotation for trips
    num_outages = int(np.sum(line_series < max_lines))
    if num_outages > 0:
        ax.text(0.02, 0.95, f"WARNING: {num_outages} timesteps with outages detected", 
                transform=ax.transAxes, color='darkred', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
    else:
        ax.text(0.02, 0.95, "Grid Topology: FULLY CONNECTED (No Outages)", 
                transform=ax.transAxes, color='green', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path
