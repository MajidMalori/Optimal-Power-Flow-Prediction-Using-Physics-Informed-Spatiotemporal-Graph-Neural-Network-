"""
Professional Topology & Reliability Report
Consolidates topology events (switching/trips) and data generation convergence quality.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List

from src.processing.topology import load_network

# Professional style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def load_convergence_data(data_dir: str, case_name: str):
    pattern = os.path.join(data_dir, f'{case_name}_data_quality_audit_frac*.json')
    files = glob.glob(pattern)
    audits = {}
    for f in files:
        try:
            filename = os.path.basename(f)
            parts = filename.replace('.json', '').split('_')
            ren_frac = 0.0
            for part in parts:
                if part.startswith('frac'):
                    ren_frac = float(part.replace('frac', ''))
                    break
            with open(f, 'r') as file:
                audits[ren_frac] = json.load(file)
        except: continue
    return audits

def plot_topology_events(data_dir: str, case_name: str, output_path: str, config: dict = None) -> str:
    """
    Consolidated Report: Topology (Switches) + Convergence (Success/Methods)
    """
    from src.constants import FeatureIndices
    
    # 1. Load Base Network for Initial Line Status
    try:
        net = load_network(case_name)
        n_lines = len(net.line)
        base_status = net.line.in_service.values.astype(int)
    except Exception as e:
        print(f"Failed to load network for {case_name}: {e}")
        return None

    # 2. Load Convergence and Switching Data
    audits = load_convergence_data(data_dir, case_name)
    if not audits: return None
    
    sorted_fracs = sorted(audits.keys())
    conv_rows = []
    total_switches = 0
    state_matrices = []
    
    for frac in sorted_fracs:
        audit = audits[frac]
        total = audit.get('total_timesteps', 96)
        succ = audit.get('successful', 0)
        res = audit.get('resolution_methods', {})
        val = audit.get('validation_stats', {})
        events = audit.get('switching_events', [])
        
        # Build state matrix for this fraction
        frac_matrix = np.zeros((n_lines, total), dtype=int)
        for t in range(total):
            frac_matrix[:, t] = base_status
            
        for event in events:
            t = event['t']
            if t < total:
                closed_idx = event['closed']
                opened_idx = event['opened']
                if opened_idx < n_lines:
                    frac_matrix[opened_idx, t] = 0
                if closed_idx < n_lines:
                    frac_matrix[closed_idx, t] = 2
                    
        state_matrices.append(frac_matrix)

        # Count switches from resolution methods (or events)
        switches = res.get('strict_contingency', 0) + res.get('relaxed_contingency', 0)
        total_switches += switches
        
        # Calculate trip count
        known = sum(res.values())
        trip = val.get('generator_trips', max(0, succ - known))
        
        conv_rows.append({
            'frac': frac * 100,
            'rate': (succ / total * 100),
            'strict_normal': res.get('strict_normal', 0),
            'strict_contingency': res.get('strict_contingency', 0),
            'relaxed': res.get('relaxed_contingency', 0),
            'restored': res.get('restored_line', 0),
            'trip': trip,
            'fail': audit.get('failed', 0)
        })
    
    df_conv = pd.DataFrame(conv_rows).sort_values('frac')
    state_matrix = np.hstack(state_matrices)
    t_total = state_matrix.shape[1]

    # Create figure with 1x2 layout and larger size to prevent overlap and squishing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), gridspec_kw={'width_ratios': [2, 1]})
    
    # ========== PLOT 1: Spatiotemporal Grid Configuration Matrix (Heatmap) ==========
    # Clearer Color Palette:
    # 0 = Open (Dark Slate - looks like a cut/empty)
    # 1 = Base Closed (Light Gray - unobtrusive background)
    # 2 = Strategic Closed (Bright Red - stands out clearly)
    cmap = mcolors.ListedColormap(['#34495e', '#ecf0f1', '#e74c3c'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    cax = ax1.imshow(state_matrix, aspect='auto', cmap=cmap, norm=norm, origin='lower', interpolation='none')
    
    ax1.set_xlabel('Simulation Timestep', fontweight='bold')
    ax1.set_ylabel('Transmission Line ID', fontweight='bold')
    ax1.set_title(f'Spatiotemporal Grid Configuration Matrix\n({total_switches} Total Rerouting Events Observed)', fontweight='bold', pad=40)
    
    # Custom Legend - move it BELOW the plot to completely avoid top labels
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#34495e', edgecolor='none', label='Line Open (Disconnected)'),
        Patch(facecolor='#ecf0f1', edgecolor='#bdc3c7', label='Line Closed (Base Config)'),
        Patch(facecolor='#e74c3c', edgecolor='none', label='Strategic Tie-Line Closed')
    ]
    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=12)
    
    # Y-axis Grid Formatting
    if n_lines < 60:
        ax1.set_yticks(np.arange(-0.5, n_lines, 1), minor=True)
        ax1.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
        ax1.tick_params(which='minor', bottom=False, left=False)
        ax1.set_yticks(np.arange(0, n_lines, 5))
        ax1.set_yticklabels(np.arange(0, n_lines, 5))
    else:
        ax1.set_yticks(np.arange(0, n_lines, 10))

    # X-axis Fraction Boundaries
    timesteps_per_frac = state_matrices[0].shape[1]
    ax1.set_xticks(np.arange(0, t_total + 1, timesteps_per_frac))
    ax1.set_xticklabels([f"{i*timesteps_per_frac}" for i in range(len(sorted_fracs)+1)])
    
    for i in range(1, len(sorted_fracs)):
        ax1.axvline(i * timesteps_per_frac - 0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        # Move the percentage labels slightly higher to avoid legend overlap
        ax1.text(i * timesteps_per_frac - (timesteps_per_frac/2), n_lines + 1.0, f'{int(sorted_fracs[i-1]*100)}%', ha='center', va='bottom', fontweight='bold', color='darkgreen', fontsize=12)
    
    ax1.text(len(sorted_fracs) * timesteps_per_frac - (timesteps_per_frac/2), n_lines + 1.0, f'{int(sorted_fracs[-1]*100)}%', ha='center', va='bottom', fontweight='bold', color='darkgreen', fontsize=12)
    
    # ========== PLOT 2: Stability Resolution Strategies (Side-by-Side) ==========
    methods = ['strict_normal', 'strict_contingency', 'relaxed', 'restored', 'trip', 'fail']
    labels = ['Strict (Normal)', 'Strict (Switch)', 'Relaxed Opt', 'Line Restored', 'Gen Trip', 'Solver Fail']
    colors = ['#2ecc71', '#16a085', '#f1c40f', '#3498db', '#9b59b6', '#c0392b']
    
    x = np.arange(len(df_conv))
    bottom = np.zeros(len(df_conv))
    for m, l, c in zip(methods, labels, colors):
        vals = df_conv[m].values
        ax2.bar(x, vals, bottom=bottom, label=l, color=c, alpha=0.85, width=0.6)
        bottom += vals
        
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(f)}%" for f in df_conv['frac']])
    ax2.set_xlabel('Renewable Penetration (%)', fontweight='bold')
    ax2.set_ylabel('Timestep Allocation', fontweight='bold')
    ax2.set_title('Stability Resolution Strategies', fontweight='bold', pad=40)
    # Move legend outside to the right to prevent covering data
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, shadow=False, fontsize=11)
    ax2.grid(True, alpha=0.2, axis='y')

    # Final layout
    fig.suptitle(f'Comprehensive Reliability & Topology Report — {case_name.upper()}', fontweight='bold', y=1.05, fontsize=20)
    # Adjust layout to make room for bottom legend
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path
