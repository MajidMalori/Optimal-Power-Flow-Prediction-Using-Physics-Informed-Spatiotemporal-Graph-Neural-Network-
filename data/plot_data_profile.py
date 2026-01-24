"""
Professional Data Profile Visualization
Creates publication-quality plots showing load/generation patterns and data quality
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from config import FeatureIndices

# Set professional style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})


def plot_data_profile(config, case_name: str, output_path: str) -> str:
    """
    Create professional data profile visualization.
    
    Shows:
    1. Daily load pattern across renewable levels
    2. Daily generation pattern (conv + renewable)
    3. Renewable generation by penetration level
    4. Data quality metrics (voltage health + diversity)
    """
    
    # Load RAW data directly
    feature_pattern = os.path.join(config.DATA_DIR, f"{case_name}_features_frac*.npy")
    feature_files = sorted(glob.glob(feature_pattern))
    
    if not feature_files:
        print(f"No data found for {case_name}")
        return None
    
    # Determine system base power for proper unit conversion
    case_num = int(case_name.replace('case', ''))
    s_base = 10.0 if case_num == 33 else 100.0
    
    # Load all data
    all_features = []
    all_ren_fracs = []
    
    for f in feature_files:
        filename = os.path.basename(f)
        frac = float(filename.split('frac')[-1].replace('.npy', ''))
        features = np.array(np.load(f, mmap_mode='r'))  # [samples, buses, 10]
        
        all_features.append(features)
        all_ren_fracs.extend([frac] * features.shape[0])
    
    features = np.vstack(all_features)  # [total_samples, buses, 10]
    ren_fracs = np.array(all_ren_fracs)
    n_samples, n_buses, _ = features.shape
    
    # Extract features (already in physical units: MW, MVar, p.u.)
    p_load = features[:, :, FeatureIndices.P_LOAD]  # MW
    p_conv = features[:, :, FeatureIndices.P_CONV]  # MW
    p_ren = features[:, :, FeatureIndices.P_REN]    # MW
    vm = features[:, :, 8]  # Voltage magnitude in p.u.
    
    # Calculate hourly indices
    hours = np.arange(n_samples) % config.HOURS_PER_DAY
    unique_fracs = np.sort(np.unique(np.round(ren_fracs, 1)))
    
    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Color palette - professional gradients
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(unique_fracs)))
    
    # ========== PLOT 1: Daily Load Pattern ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    for i, frac in enumerate(unique_fracs):
        mask = np.abs(ren_fracs - frac) < 0.05
        total_load = np.sum(p_load[mask], axis=1)  # Total system load in MW
        
        hourly_mean = []
        hourly_std = []
        for h in range(config.HOURS_PER_DAY):
            hour_data = total_load[hours[mask] == h]
            if len(hour_data) > 0:
                hourly_mean.append(np.mean(hour_data))
                hourly_std.append(np.std(hour_data))
            else:
                hourly_mean.append(np.nan)
                hourly_std.append(np.nan)
        
        hourly_mean = np.array(hourly_mean)
        hourly_std = np.array(hourly_std)
        
        time_axis = np.arange(config.HOURS_PER_DAY)
        ax1.plot(time_axis, hourly_mean, color=colors[i], 
                label=f'{int(frac*100)}% Renewable', linewidth=2.5, alpha=0.9)
        ax1.fill_between(time_axis, hourly_mean - hourly_std, hourly_mean + hourly_std,
                        color=colors[i], alpha=0.15)
    
    ax1.set_xlabel('Hour of Day', fontweight='bold')
    ax1.set_ylabel('Total Active Load (MW)', fontweight='bold')
    ax1.set_title('Daily Load Profile', fontweight='bold', pad=10)
    ax1.set_xticks(range(0, config.HOURS_PER_DAY, 3))
    ax1.set_xticklabels([f'{h:02d}:00' for h in range(0, config.HOURS_PER_DAY, 3)])
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # ========== PLOT 2: Daily Generation Pattern ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    for i, frac in enumerate(unique_fracs):
        mask = np.abs(ren_fracs - frac) < 0.05
        total_gen = np.sum(p_conv[mask] + p_ren[mask], axis=1)  # Total generation in MW
        
        hourly_mean = []
        hourly_std = []
        for h in range(config.HOURS_PER_DAY):
            hour_data = total_gen[hours[mask] == h]
            if len(hour_data) > 0:
                hourly_mean.append(np.mean(hour_data))
                hourly_std.append(np.std(hour_data))
            else:
                hourly_mean.append(np.nan)
                hourly_std.append(np.nan)
        
        hourly_mean = np.array(hourly_mean)
        hourly_std = np.array(hourly_std)
        
        time_axis = np.arange(config.HOURS_PER_DAY)
        ax2.plot(time_axis, hourly_mean, color=colors[i], 
                label=f'{int(frac*100)}% Renewable', linewidth=2.5, alpha=0.9)
        ax2.fill_between(time_axis, hourly_mean - hourly_std, hourly_mean + hourly_std,
                        color=colors[i], alpha=0.15)
    
    ax2.set_xlabel('Hour of Day', fontweight='bold')
    ax2.set_ylabel('Total Generation (MW)', fontweight='bold')
    ax2.set_title('Daily Generation Profile', fontweight='bold', pad=10)
    ax2.set_xticks(range(0, config.HOURS_PER_DAY, 3))
    ax2.set_xticklabels([f'{h:02d}:00' for h in range(0, config.HOURS_PER_DAY, 3)])
    ax2.legend(loc='best', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    # ========== PLOT 3: Renewable Generation by Level ==========
    ax3 = fig.add_subplot(gs[1, 0])
    
    ren_by_level = []
    labels = []
    for i, frac in enumerate(unique_fracs):
        mask = np.abs(ren_fracs - frac) < 0.05
        total_ren = np.sum(p_ren[mask], axis=1)
        ren_by_level.append(total_ren)
        labels.append(f'{int(frac*100)}%')
    
    bp = ax3.boxplot(ren_by_level, labels=labels, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='darkred', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    # Color boxes with gradient
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_xlabel('Renewable Penetration Level', fontweight='bold')
    ax3.set_ylabel('Total Renewable Generation (MW)', fontweight='bold')
    ax3.set_title('Renewable Generation Distribution', fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ========== PLOT 4: Voltage Health & Data Diversity ==========
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate voltage statistics
    vm_mean = np.mean(vm, axis=0)  # Mean voltage per bus
    vm_std = np.std(vm, axis=0)    # Std voltage per bus
    
    # Create scatter plot: x=bus index, y=mean voltage, size=std
    bus_indices = np.arange(n_buses)
    scatter = ax4.scatter(bus_indices, vm_mean, s=vm_std*500, 
                         c=vm_std, cmap='RdYlGn_r', alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Add voltage limits
    v_min = 0.95 if case_num == 33 else 0.90
    v_max = 1.05 if case_num == 33 else 1.10
    ax4.axhline(v_min, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Min Limit ({v_min} p.u.)')
    ax4.axhline(v_max, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Max Limit ({v_max} p.u.)')
    ax4.axhline(1.0, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label='Nominal (1.0 p.u.)')
    
    ax4.set_xlabel('Bus Index', fontweight='bold')
    ax4.set_ylabel('Mean Voltage (p.u.)', fontweight='bold')
    ax4.set_title('Voltage Health (bubble size = variability)', fontweight='bold', pad=10)
    ax4.set_ylim(max(0.85, vm_mean.min() - 0.02), min(1.15, vm_mean.max() + 0.02))
    ax4.legend(loc='best', frameon=True, shadow=True, fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4, label='Voltage Std Dev (p.u.)')
    cbar.ax.tick_params(labelsize=9)
    
    # Main title
    fig.suptitle(f'Data Profile Analysis - {case_name.upper()} System ({n_buses} buses, {s_base:.0f} MVA base)',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path
