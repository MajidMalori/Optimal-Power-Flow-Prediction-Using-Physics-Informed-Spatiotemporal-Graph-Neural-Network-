"""
Professional Physics Health Visualization
Shows voltage distribution and system health metrics using REAL data
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
})


def plot_physics_health(config, case_name: str, output_path: str) -> str:
    """
    Create professional physics health visualization.
    """
    data_dir = os.path.join(config.get('output_dir', 'data/01_raw'), case_name)
    feature_pattern = os.path.join(data_dir, f"{case_name}_features_frac*.npy")
    feature_files = sorted(glob.glob(feature_pattern))
    
    if not feature_files:
        print(f"No data found for {case_name} in {data_dir}")
        return None
    
    # Determine system-specific limits
    physics = SYSTEM_PHYSICS.get(case_name, SYSTEM_PHYSICS['default'])
    v_min = physics['v_min']
    v_max = physics['v_max']
    
    # Load voltage data from all files
    all_voltages = []
    for f in feature_files:
        features = np.array(np.load(f, mmap_mode='r'))  # [samples, buses, 10]
        vm = features[:, :, FeatureIndices.VM]  # Voltage magnitude in p.u.
        all_voltages.append(vm)
    
    voltages = np.vstack(all_voltages)  # [total_samples, buses]
    n_samples, n_buses = voltages.shape
    
    # Create figure with 1x2 layout
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ========== LEFT: Overall Voltage Distribution (FULL VIEW) ==========
    ax1 = axes[0]
    
    # Flatten all voltages
    v_flat = voltages.flatten()
    
    # Create histogram with KDE - Full Range (0.0 to 1.2)
    sns.histplot(v_flat, bins=100, kde=False, color='#3498db', stat='density',
                alpha=0.6, edgecolor='black', linewidth=0.5, ax=ax1)
    
    # Add limit lines
    ax1.axvline(v_min, color='orange', linestyle='--', linewidth=2.5, 
               label=f'Operating Min ({v_min} p.u.)', alpha=0.8)
    ax1.axvline(v_max, color='orange', linestyle='--', linewidth=2.5, 
               label=f'Operating Max ({v_max} p.u.)', alpha=0.8)
    ax1.axvline(1.0, color='green', linestyle='-', linewidth=2, 
               label='Nominal (1.0 p.u.)', alpha=0.7)
    
    # Annotate Missing Data (0.0 values)
    zero_count = np.sum(v_flat < 0.1)
    zero_pct = zero_count / len(v_flat) * 100
    
    # Add arrow and text for missing data
    ax1.annotate(f'Missing Data / Unobserved\n({zero_pct:.1f}% of measurements)',
                xy=(0.0, 0), xytext=(0.2, ax1.get_ylim()[1]*0.5),
                arrowprops=dict(facecolor='red', shrink=0.05),
                fontsize=11, fontweight='bold', color='red')
    
    # Calculate violation statistics (excluding missing data)
    valid_mask = v_flat > 0.1
    v_valid = v_flat[valid_mask]
    
    if len(v_valid) > 0:
        v_low = np.sum(v_valid < v_min) / len(v_valid) * 100
        v_high = np.sum(v_valid > v_max) / len(v_valid) * 100
        v_good = 100 - v_low - v_high
        mean_valid = np.mean(v_valid)
    else:
        v_low, v_high, v_good, mean_valid = 0, 0, 0, 0
    
    # Add statistics text box
    stats_text = f'Global Statistics:\n'
    stats_text += f'Total Points: {len(v_flat):,}\n'
    stats_text += f'Missing (0.0): {zero_pct:.1f}%\n'
    stats_text += f'Observed: {100-zero_pct:.1f}%\n'
    stats_text += f'-- Of Observed --\n'
    stats_text += f'Mean: {mean_valid:.4f} p.u.\n'
    stats_text += f'Within Limits: {v_good:.2f}%'
    
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5))
    
    ax1.set_xlabel('Voltage Magnitude (p.u.)', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Density', fontweight='bold', fontsize=13)
    ax1.set_title(f'Full Voltage Distribution (Including Missing Data)',
                 fontweight='bold', fontsize=14, pad=15)
    ax1.legend(loc='upper center', frameon=True, shadow=True, fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(-0.05, 1.25)  # Show full range from 0 to 1.2
    
    # ========== RIGHT: Per-Bus Voltage Statistics ==========
    ax2 = axes[1]
    
    # Calculate coverage per bus (percentage of non-zero samples)
    coverage_per_bus = np.sum(voltages > 0.1, axis=0) / n_samples * 100
    
    # Calculate statistics on VALID data only
    v_mean_per_bus = []
    v_std_per_bus = []
    v_min_per_bus = []
    v_max_per_bus = []
    
    for i in range(n_buses):
        bus_v = voltages[:, i]
        valid_v = bus_v[bus_v > 0.1]
        if len(valid_v) > 0:
            v_mean_per_bus.append(np.mean(valid_v))
            v_std_per_bus.append(np.std(valid_v))
            v_min_per_bus.append(np.min(valid_v))
            v_max_per_bus.append(np.max(valid_v))
        else:
            v_mean_per_bus.append(0.0)
            v_std_per_bus.append(0.0)
            v_min_per_bus.append(0.0)
            v_max_per_bus.append(0.0)
            
    v_mean_per_bus = np.array(v_mean_per_bus)
    v_std_per_bus = np.array(v_std_per_bus)
    v_min_per_bus = np.array(v_min_per_bus)
    v_max_per_bus = np.array(v_max_per_bus)
    
    bus_indices = np.arange(n_buses)
    
    # Categorize buses by coverage
    observed_mask = coverage_per_bus > 5  # At least 5% coverage
    unobserved_mask = ~observed_mask
    
    # 1. Observed Buses (Blue Circles)
    if np.any(observed_mask):
        ax2.errorbar(bus_indices[observed_mask], v_mean_per_bus[observed_mask], yerr=v_std_per_bus[observed_mask],
                    fmt='o', markersize=6, color='#3498db', ecolor='lightblue',
                    elinewidth=2, capsize=3, alpha=0.8, label='Observed Mean ± Std')
        
        # Add range for observed
        ax2.fill_between(bus_indices[observed_mask], v_min_per_bus[observed_mask], v_max_per_bus[observed_mask],
                        color='lightblue', alpha=0.3, label='Observed Range')

    # 2. Unobserved Buses (Gray Xs)
    if np.any(unobserved_mask):
        ax2.scatter(bus_indices[unobserved_mask], np.zeros(np.sum(unobserved_mask)), 
                   color='gray', marker='x', s=60, linewidths=2, alpha=0.8, label='Unobserved / Missing')
    else:
        # If NO unobserved buses, add a note explaining why
        ax2.text(0.95, 0.05, "Note: All buses are partially observed (Dynamic PMUs)", 
                transform=ax2.transAxes, ha='right', va='bottom', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#3498db'))
    
    # Add limit lines
    ax2.axhline(v_min, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axhline(v_max, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axhline(1.0, color='green', linestyle='-', linewidth=1.5, alpha=0.5)
    
    # Add Missing Data Line at 0.0
    ax2.axhline(0.0, color='red', linestyle='-', linewidth=1, alpha=0.3)
    ax2.text(n_buses, 0.0, " Missing Data Level (0.0)", color='red', va='center', fontsize=9, alpha=0.7)
    
    ax2.set_xlabel('Bus Index', fontweight='bold', fontsize=13)
    ax2.set_ylabel('Voltage Magnitude (p.u.)', fontweight='bold', fontsize=13)
    ax2.set_title('Per-Bus Statistics (Observed vs Missing)', fontweight='bold', fontsize=14, pad=15)
    ax2.set_ylim(-0.05, 1.2)
    ax2.legend(loc='lower right', frameon=True, shadow=True, fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Main title
    fig.suptitle(f'Physics Health Report - {case_name.upper()} System\n(Showing Impact of {100-np.mean(coverage_per_bus):.1f}% Missing PMU Data)',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path
