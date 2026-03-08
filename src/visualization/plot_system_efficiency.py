"""
Professional System Efficiency Visualization
Shows the thermodynamic impact of renewable penetration (Line Losses).
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from src.constants import FeatureIndices, SYSTEM_PHYSICS

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
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

def plot_system_efficiency(config, case_name: str, output_path: str) -> str:
    """
    Create professional system efficiency visualization.
    """
    data_dir = os.path.join(config.get('output_dir', 'data/01_raw'), case_name)
    feature_pattern = os.path.join(data_dir, f"{case_name}_features_frac*.npy")
    feature_files = sorted(glob.glob(feature_pattern))
    
    if not feature_files:
        print(f"No data found for {case_name} in {data_dir}")
        return None
        
    physics = SYSTEM_PHYSICS.get(case_name, SYSTEM_PHYSICS['default'])
    s_base = physics['base_mva']
    
    fracs = []
    mean_losses = []
    std_losses = []
    
    for f in feature_files:
        filename = os.path.basename(f)
        try:
            parts = filename.replace('.npy', '').split('_')
            frac = 0.0
            for p in parts:
                if p.startswith('frac'):
                    frac = float(p.replace('frac', ''))
                    break
        except:
            continue
            
        features = np.load(f)
        
        p_load = features[:, :, FeatureIndices.P_LOAD]
        p_ext = features[:, :, FeatureIndices.P_EXT_GRID]
        p_conv = features[:, :, FeatureIndices.P_CONV]
        p_ren = features[:, :, FeatureIndices.P_REN]
        
        total_load = np.sum(p_load, axis=1)
        total_gen = np.sum(p_ext + p_conv + p_ren, axis=1)
        
        # Power Balance: Generation = Load + Losses
        losses = total_gen - total_load
        
        fracs.append(frac * 100) # Convert to percentage
        mean_losses.append(np.mean(losses))
        std_losses.append(np.std(losses))
        
    if not fracs:
        return None
        
    # Sort by fraction
    sort_idx = np.argsort(fracs)
    fracs = np.array(fracs)[sort_idx]
    mean_losses = np.array(mean_losses)[sort_idx]
    std_losses = np.array(std_losses)[sort_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate smooth curve
    ax.plot(fracs, mean_losses, marker='o', markersize=8, linewidth=3, color='#e74c3c', label='Mean Active Power Loss (MW)')
    ax.fill_between(fracs, mean_losses - std_losses, mean_losses + std_losses, color='#e74c3c', alpha=0.15, label='±1 Standard Deviation')
    
    ax.set_xlabel('Renewable Penetration Level (%)', fontweight='bold')
    ax.set_ylabel('Total System Line Losses (MW)', fontweight='bold')
    ax.set_title(f'Thermodynamic System Efficiency - {case_name.upper()}\n(Illustrating the Impact of Local Generation vs Reverse Power Flow)', pad=15, fontweight='bold')
    
    ax.set_xticks(fracs)
    ax.set_xticklabels([f'{int(f)}%' for f in fracs])
    
    # Annotate the minimum loss point if there's variation
    if len(mean_losses) > 1 and np.max(mean_losses) > np.min(mean_losses):
        min_idx = np.argmin(mean_losses)
        min_frac = fracs[min_idx]
        min_loss = mean_losses[min_idx]
        
        ax.annotate(f'Optimal Efficiency\n({min_loss:.2f} MW)',
                    xy=(min_frac, min_loss),
                    xytext=(min_frac, min_loss + (np.max(mean_losses) - np.min(mean_losses)) * 0.2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    ha='center', va='bottom', fontweight='bold')
    
    ax.legend(loc='lower right', frameon=True, shadow=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path
