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

from config import Config
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
    # Generating Data Profile Story (silent)
    
    try:
        # Use provided data or load from disk
        if features is None or normalizer is None or renewable_fractions is None:
            # Load data using lazy loading system
            from utils.data_loader import load_power_system_data, PowerSystemLazyDataset
            data_tuple = load_power_system_data(config, case_name)
            # Handle both new (6 values) and old (4 values) return formats
            if len(data_tuple) == 6:
                file_metadata, adjacency, ybus_metadata, normalizer, topology_cache, topology_ids = data_tuple
            else:
                file_metadata, adjacency, ybus_metadata, normalizer = data_tuple[:4]
                topology_cache, topology_ids = None, None
            
            # Create lazy dataset and load all data for analysis (one-time full load)
            dataset = PowerSystemLazyDataset(
                file_metadata=file_metadata,
                adjacency_matrix=adjacency,
                normalizer=normalizer,
                ybus_metadata=ybus_metadata,
                is_static=True,
                sequence_length=1,
                hours_per_day=config.HOURS_PER_DAY,
                topology_cache=topology_cache,
                topology_ids=topology_ids
            )
            
            # Load all data from lazy dataset (for analysis purposes)
            # Loading all data for analysis (silent)
            all_features = []
            all_renewable_fractions = []
            
            for idx in range(len(dataset)):
                sample = dataset[idx]
                # Denormalize features for analysis
                features_denorm = normalizer.denormalize(sample['features'].unsqueeze(0)).squeeze(0).numpy()
                all_features.append(features_denorm)
                all_renewable_fractions.append(sample['renewable_fraction'].item())
            
            # Stack into arrays (features are already denormalized from dataset)
            features = np.stack(all_features, axis=0)  # [n_samples, n_buses, 10] - already denormalized
            renewable_fractions = np.array(all_renewable_fractions)  # [n_samples]
        
        # Import feature indices constants (single source of truth)
        from config import FeatureIndices
        
        # Features shape: [n_samples, n_buses, 10]
        n_samples, n_buses, n_features = features.shape
        
        # Features are already denormalized (done in the loop above)
        features_denorm = features  # [n_samples, n_buses, 10]
        
        # Extract denormalized features using constants (single source of truth)
        p_load = features_denorm[:, :, FeatureIndices.P_LOAD]      # Active load (MW, should be >= 0)
        q_load = features_denorm[:, :, FeatureIndices.Q_LOAD]      # Reactive load
        p_conv = features_denorm[:, :, FeatureIndices.P_CONV]      # Conventional generation
        p_ren = features_denorm[:, :, FeatureIndices.P_REN]        # Renewable generation (MW, should be >= 0)
        q_ren = features_denorm[:, :, FeatureIndices.Q_REN]        # Reactive renewable
        
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
        
        # Store data for consistent y-axis scaling
        load_data_all = []
        gen_data_all = []
        
        # 1. Total Active Load (sum across all buses)
        ax = axes[0, 0]
        ax.set_title('Total Active Load (Sum across all buses)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Load (p.u.)', fontsize=11)
        
        for i, ren_frac in enumerate(unique_ren_fractions):
            mask = np.abs(renewable_fractions - ren_frac) < 0.05
            if not np.any(mask):
                continue
            
            load_data = np.sum(p_load[mask], axis=1)  # Sum across buses
            load_data_all.append(load_data)
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
        axes[0, 0].set_xlabel('Hour of Day', fontsize=11)
        
        # 2. Total Generation (Conventional + Renewable)
        ax = axes[0, 1]
        ax.set_title('Total Generation (Conventional + Renewable)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Generation (p.u.)', fontsize=11)
        
        for i, ren_frac in enumerate(unique_ren_fractions):
            mask = np.abs(renewable_fractions - ren_frac) < 0.05
            if not np.any(mask):
                continue
            
            # Total generation = conventional + renewable
            total_gen = np.sum(p_conv[mask] + p_ren[mask], axis=1)  # Sum across buses
            gen_data_all.append(total_gen)
            hourly_mean = []
            hourly_std = []
            
            for h in range(hours_per_day):
                hour_mask = (hours[mask] == h)
                if np.any(hour_mask):
                    hourly_mean.append(np.mean(total_gen[hour_mask]))
                    hourly_std.append(np.std(total_gen[hour_mask]))
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
        axes[0, 1].set_xlabel('Hour of Day', fontsize=11)
        
        # Let each plot have its own y-axis scale for better visibility
        
        # ========================================================================
        # BOTTOM ROW: Data Variability and Integrity
        # ========================================================================
        
        # 3. Variability (Coefficient of Variation) across time
        # CV = std/mean (standardized measure of dispersion)
        ax = axes[1, 0]
        ax.set_title('Coefficient of Variation (Std/Mean) across Time for Each Hour', fontweight='bold', fontsize=12)
        ax.set_ylabel('CV = Std/Mean', fontsize=11)
        
        for i, ren_frac in enumerate(unique_ren_fractions):
            mask = np.abs(renewable_fractions - ren_frac) < 0.05
            if not np.any(mask):
                continue
            
            # Use total generation (conventional + renewable) for CV calculation
            total_gen = np.sum(p_conv[mask] + p_ren[mask], axis=1)
            hourly_cv = []
            
            for h in range(hours_per_day):
                hour_mask = (hours[mask] == h)
                if np.any(hour_mask):
                    hour_data = total_gen[hour_mask]
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
        
        # Save to current run directory (in run_XXXXXX/XXbus folder, not experimental_results/XXbus)
        # CURRENT_RUN_DIR is a property, so we need to check if timestamp is set
        try:
            # Check if timestamp is initialized (CURRENT_RUN_DIR depends on it)
            if hasattr(config, '_CURRENT_RUN_TIMESTAMP') and config._CURRENT_RUN_TIMESTAMP:
                current_run_dir = config.CURRENT_RUN_DIR
                if current_run_dir and os.path.exists(os.path.dirname(current_run_dir)):
                    output_dir = os.path.join(current_run_dir, f"{case_name.replace('case', '')}bus")
                else:
                    # Fallback if run directory doesn't exist yet
                    output_dir = os.path.join(config.EXPERIMENTAL_RESULTS_DIR, f"{case_name.replace('case', '')}bus")
            else:
                # Timestamp not set yet, use experimental_results (shouldn't happen, but safe fallback)
                output_dir = os.path.join(config.EXPERIMENTAL_RESULTS_DIR, f"{case_name.replace('case', '')}bus")
        except (AttributeError, TypeError):
            # Fallback if CURRENT_RUN_DIR property fails
            output_dir = os.path.join(config.EXPERIMENTAL_RESULTS_DIR, f"{case_name.replace('case', '')}bus")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save single combined figure
        save_path = os.path.join(output_dir, 'data_profile_story.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Summary statistics computed but not printed (plots show the data)
        
        return issues
        
    except Exception as e:
        print(f"Error analyzing {case_name}: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """Run data profile analysis for all bus systems."""
    # Load config from YAML (required - no Args class)
    # Config.__init__ will load all values from config.yaml
    config = Config(
        data_mode='test',  # Parameter default - actual value comes from YAML
        save_results=True,
        clear_results=False
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
        print(f"\nTotal issues found: {len(all_issues)}")
        for issue in all_issues:
            print(f"  • {issue}")
    else:
        print("\nNo major issues detected in data profiles!")


if __name__ == "__main__":
    main()

