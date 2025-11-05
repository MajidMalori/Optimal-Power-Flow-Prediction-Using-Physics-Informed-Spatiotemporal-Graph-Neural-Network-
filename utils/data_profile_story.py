"""
Data Profile Story: Comprehensive Data Authenticity and Quality Analysis

This script tells the complete story of your power system data:
- Daily load and generation profiles across renewable penetration levels
- Data variability and consistency patterns
- Visual data integrity checks (flatlining buses, anomalies)

Purpose: When someone asks "Which data did you use and how authentic is it?", 
         you can show them these graphs.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Tuple

# Add project root to path (parent directory of utils)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from config import Config, Args
from utils.data_loader import load_power_system_data

def analyze_data_profiles(config: Config, case_name: str, features=None, normalizer=None, renewable_fractions=None):
    """
    Analyze load, wind, and solar profiles from generated data.
    
    Args:
        config: Configuration object
        case_name: Case name (e.g., "case33")
        features: Optional pre-loaded features array [n_samples, n_buses, 10] (if None, loads from disk)
        normalizer: Optional pre-loaded normalizer (if None, loads from disk)
        renewable_fractions: Optional pre-loaded renewable fractions [n_samples] (if None, loads from disk)
    
    Returns:
        List of issues found (empty if no issues)
    """
    print(f"Generating Data Profile Story for {case_name}")
    
    try:
        # Use provided data or load from disk
        if features is None or normalizer is None or renewable_fractions is None:
            # Load data
            data_tuple = load_power_system_data(config, case_name)
            features, adjacency, ybus_matrices, targets, bus_types, energy_coeffs, carbon_coeffs, renewable_fractions, normalizer = data_tuple
        
        # Features shape: [n_samples, n_buses, 10]
        # Features: [vm_pu, va_rad, p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren]
        n_samples, n_buses, n_features = features.shape
        
        # CRITICAL: Denormalize features to check actual values (not normalized)
        # Normalized features can be negative, which is why we see negative loads
        # Convert numpy to torch tensor for denormalize, then back to numpy
        import torch
        features_tensor = torch.from_numpy(features).float()
        features_denorm = normalizer.denormalize(features_tensor).numpy()  # [n_samples, n_buses, 10]
        
        # Extract denormalized features
        p_load = features_denorm[:, :, 2]   # Active load (MW, should be >= 0)
        q_load = features_denorm[:, :, 3]     # Reactive load
        p_conv = features_denorm[:, :, 6]    # Conventional generation
        p_ren = features_denorm[:, :, 8]     # Renewable generation (MW, should be >= 0)
        q_ren = features_denorm[:, :, 9]     # Reactive renewable
        
        # Calculate hourly patterns
        hours_per_day = config.HOURS_PER_DAY
        hours = np.arange(n_samples) % hours_per_day
        
        # Group by renewable fraction
        unique_ren_fractions = np.sort(np.unique(np.round(renewable_fractions, decimals=1)))
        
        # Create single figure with 2x2 grid for efficiency and impact
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Data Profile Story - {case_name.upper()}', fontsize=18, fontweight='bold')
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_ren_fractions)))
        
        # ========================================================================
        # TOP ROW: Daily Load and Generation Profiles
        # ========================================================================
        
        # 1. Total Active Load (sum across all buses)
        ax = axes[0, 0]
        ax.set_title('Total Active Load (Sum across all buses)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Load (p.u.)', fontsize=11)
        
        for i, ren_frac in enumerate(unique_ren_fractions):
            mask = np.abs(renewable_fractions - ren_frac) < 0.05
            if not np.any(mask):
                continue
            
            load_data = np.sum(p_load[mask], axis=1)  # Sum across buses
            hourly_mean = []
            hourly_std = []
            
            for h in range(hours_per_day):
                hour_mask = (hours[mask] == h)
                if np.any(hour_mask):
                    hourly_mean.append(np.mean(load_data[hour_mask]))
                    hourly_std.append(np.std(load_data[hour_mask]))
                else:
                    hourly_mean.append(np.nan)
                    hourly_std.append(np.nan)
            
            hourly_mean = np.array(hourly_mean)
            hourly_std = np.array(hourly_std)
            
            ax.plot(range(hours_per_day), hourly_mean, color=colors[i], 
                   label=f'{int(ren_frac*100)}% Renewables', linewidth=2)
            ax.fill_between(range(hours_per_day), hourly_mean - hourly_std, 
                           hourly_mean + hourly_std, color=colors[i], alpha=0.2)
        
        ax.set_xticks(range(0, hours_per_day, 3))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, hours_per_day, 3)])
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Set xlabel for top row
        axes[0, 0].set_xlabel('Hour of Day', fontsize=11)
        
        # Expected pattern check
        max_hour = np.argmax([np.nanmean([np.mean(np.sum(p_load[hours == h], axis=1)) for _ in range(1)]) for h in range(hours_per_day)])
        if max_hour not in [17, 18, 19]:
            print(f"  ⚠️  WARNING: Load peak is at hour {max_hour}, not at expected evening peak!")
        
        # 2. Total Renewable Generation
        ax = axes[0, 1]
        ax.set_title('Total Renewable Generation (Sum across all buses)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Generation (p.u.)', fontsize=11)
        
        for i, ren_frac in enumerate(unique_ren_fractions):
            mask = np.abs(renewable_fractions - ren_frac) < 0.05
            if not np.any(mask):
                continue
            
            ren_data = np.sum(p_ren[mask], axis=1)
            hourly_mean = []
            hourly_std = []
            
            for h in range(hours_per_day):
                hour_mask = (hours[mask] == h)
                if np.any(hour_mask):
                    hourly_mean.append(np.mean(ren_data[hour_mask]))
                    hourly_std.append(np.std(ren_data[hour_mask]))
                else:
                    hourly_mean.append(np.nan)
                    hourly_std.append(np.nan)
            
            hourly_mean = np.array(hourly_mean)
            hourly_std = np.array(hourly_std)
            
            ax.plot(range(hours_per_day), hourly_mean, color=colors[i], 
                   label=f'{int(ren_frac*100)}% Renewables', linewidth=2)
            ax.fill_between(range(hours_per_day), hourly_mean - hourly_std, 
                           hourly_mean + hourly_std, color=colors[i], alpha=0.2)
        
        ax.set_xticks(range(0, hours_per_day, 3))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, hours_per_day, 3)])
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Check renewable pattern
        # NOTE: p_ren includes both wind AND solar, so patterns are mixed
        # Solar peaks at noon, but wind can be high at night
        ren_max_hour = np.argmax([np.nanmean([np.mean(np.sum(p_ren[hours == h], axis=1)) for _ in range(1)]) for h in range(hours_per_day)])
        
        # Check if renewables are zero at night
        # NOTE: Wind can generate at night, so this is expected
        night_hours = [0, 1, 2, 3, 4, 20, 21, 22, 23]
        night_ren_mean = np.mean([np.mean(np.sum(p_ren[hours == h], axis=1)) for h in night_hours if np.any(hours == h)])
        print(f"  Night renewable generation (hours 0-4, 20-23): {night_ren_mean:.4f} MW")
        print(f"    (Note: Wind generation can occur at night, so this is expected if wind is present)")
        
        # Set xlabel for top row
        axes[0, 1].set_xlabel('Hour of Day', fontsize=11)
        
        # ========================================================================
        # BOTTOM ROW: Data Variability and Integrity
        # ========================================================================
        
        # 3. Variability (Coefficient of Variation) across time
        ax = axes[1, 0]
        ax.set_title('Coefficient of Variation (Std/Mean) across Time for Each Hour', fontweight='bold', fontsize=12)
        ax.set_ylabel('CV = Std/Mean', fontsize=11)
        
        for i, ren_frac in enumerate(unique_ren_fractions):
            mask = np.abs(renewable_fractions - ren_frac) < 0.05
            if not np.any(mask):
                continue
            
            ren_data = np.sum(p_ren[mask], axis=1)
            hourly_cv = []
            
            for h in range(hours_per_day):
                hour_mask = (hours[mask] == h)
                if np.any(hour_mask):
                    hour_data = ren_data[hour_mask]
                    mean_val = np.mean(hour_data)
                    if mean_val > 1e-6:  # Avoid division by zero
                        cv = np.std(hour_data) / mean_val
                    else:
                        cv = 0.0
                    hourly_cv.append(cv)
                else:
                    hourly_cv.append(np.nan)
            
            ax.plot(range(hours_per_day), hourly_cv, color=colors[i], 
                   label=f'{int(ren_frac*100)}% Renewables', linewidth=2, marker='o')
        
        ax.set_xticks(range(0, hours_per_day, 3))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, hours_per_day, 3)])
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('CV (Std/Mean)', fontsize=11)
        ax.set_xlabel('Hour of Day', fontsize=11)
        
        # 4. Data Integrity Visualization (Visual check for flatlining buses)
        ax = axes[1, 1]
        ax.set_title('Data Integrity: Unique Load Values per Bus', fontweight='bold', fontsize=12)
        
        issues = []
        
        # Check for flatlining buses (visual bar chart)
        unique_values_per_bus = []
        problematic_buses = []
        
        for bus in range(n_buses):
            bus_load = p_load[:, bus]
            n_unique = len(np.unique(bus_load))
            unique_values_per_bus.append(n_unique)
            if n_unique < 10:  # Less than 10 unique values = possible flatlining
                problematic_buses.append(bus)
        
        # Create bar chart
        bus_indices = np.arange(n_buses)
        colors_bar = ['red' if bus in problematic_buses else 'steelblue' for bus in bus_indices]
        bars = ax.bar(bus_indices, unique_values_per_bus, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Highlight problematic buses
        if problematic_buses:
            ax.axhline(y=10, color='red', linestyle='--', linewidth=2, label='Threshold (10 unique values)')
            for bus in problematic_buses:
                ax.text(bus, unique_values_per_bus[bus] + 5, f'Bus {bus}', 
                       ha='center', fontsize=8, fontweight='bold', color='red')
        
        ax.set_xlabel('Bus Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Unique Load Values', fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(unique_values_per_bus) * 1.2 if unique_values_per_bus else 100)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)
        
        # Check for other issues (for console output)
        load_diffs = np.diff(np.sum(p_load, axis=1))
        large_jumps = np.abs(load_diffs) > 3 * np.std(load_diffs)
        if np.any(large_jumps):
            issues.append(f"Found {np.sum(large_jumps)} large jumps in load (>{3*np.std(load_diffs):.2f} p.u.)")
        
        # Check for flat lines (already visualized above)
        for bus in problematic_buses:
            bus_load = p_load[:, bus]
            issues.append(f"Bus {bus} load has only {len(np.unique(bus_load))} unique values (possible flatlining)")
        
        # Check negative values
        negative_loads = np.sum(p_load < -0.1)
        if negative_loads > 0:
            issues.append(f"Found {negative_loads} negative load values < -0.1 MW (should be >= 0)")
        
        # Check renewable generation bounds
        base_power = getattr(config, 'S_BASE', 100.0)
        if hasattr(config, 'CASE_NAME'):
            if 'case33' in config.CASE_NAME.lower():
                base_power = 10.0
            elif 'case57' in config.CASE_NAME.lower() or 'case118' in config.CASE_NAME.lower():
                base_power = 100.0
        
        p_ren_pu = p_ren / base_power
        if np.any(p_ren_pu > 1.5):
            issues.append(f"Found {np.sum(p_ren_pu > 1.5)} renewable values > 1.5 p.u. (unrealistic, max should be ~1.0)")
        
        # Layout for single figure
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save to experimental results directory (respects current run directory)
        if hasattr(config, 'CURRENT_RUN_DIR') and config.CURRENT_RUN_DIR:
            output_dir = os.path.join(config.CURRENT_RUN_DIR, f"{case_name.replace('case', '')}bus")
        else:
            output_dir = os.path.join(config.EXPERIMENTAL_RESULTS_DIR, f"{case_name.replace('case', '')}bus")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save single combined figure
        save_path = os.path.join(output_dir, 'data_profile_story.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"  Total load range: {np.sum(p_load, axis=1).min():.4f} - {np.sum(p_load, axis=1).max():.4f} MW")
        print(f"  Total renewable range: {np.sum(p_ren, axis=1).min():.4f} - {np.sum(p_ren, axis=1).max():.4f} MW")
        print(f"  Renewable std/mean (overall CV): {np.std(p_ren) / (np.mean(p_ren) + 1e-6):.4f}")
        
        return issues
        
    except Exception as e:
        print(f"✗ Error analyzing {case_name}: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """Run data profile analysis for all bus systems."""
    args = Args()
    config = Config(
        data_mode=args.data_mode,
        save_results=True,
        test_timesteps=args.test_timesteps,
        clear_results=False,
        hours_per_day=args.hours_per_day,
        sequence_length=args.sequence_length
    )
    
    all_issues = []
    for num_buses in config.NUM_BUSES:
        case_name = f"case{num_buses}"
        config.CASE_NAME = case_name
        issues = analyze_data_profiles(config, case_name)
        all_issues.extend(issues)
    
    print(f"\n{'='*80}")
    print("DATA PROFILE ANALYSIS COMPLETE")
    print(f"{'='*80}")
    if all_issues:
        print(f"\n⚠️  Total issues found: {len(all_issues)}")
        for issue in all_issues:
            print(f"  • {issue}")
    else:
        print("\n✓ No major issues detected in data profiles!")


if __name__ == "__main__":
    main()

