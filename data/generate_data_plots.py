"""
Standalone script to generate ALL data visualization plots in one go.

Generates: convergence stories, data profiles, audit plots, contingency heatmaps, etc.

Usage:
    # Run with defaults (test mode, all buses)
    python data/generate_data_plots.py
    
    # Specify mode and buses
    python data/generate_data_plots.py --mode test --buses 33,57,118
    python data/generate_data_plots.py --mode train --buses all
"""

import os
import sys
import argparse
import shutil

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import Config
from data.plot_consolidator import generate_all_data_plots


def parse_bus_systems(bus_systems_arg, available_buses):
    """Parse bus systems argument and return list of bus numbers."""
    if bus_systems_arg.lower() == 'all':
        return available_buses
    else:
        # Parse comma-separated values
        bus_list = []
        for bus_str in bus_systems_arg.split(','):
            bus_str = bus_str.strip()
            try:
                bus_num = int(bus_str)
                if bus_num in available_buses:
                    bus_list.append(bus_num)
                else:
                    print(f"Warning: {bus_num}-bus system not available. Available: {available_buses}")
            except ValueError:
                print(f"Warning: Invalid bus system '{bus_str}'. Skipping.")
        return bus_list if bus_list else available_buses


def main():
    parser = argparse.ArgumentParser(
        description='Generate data visualization plots for saved power system data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (test mode, all buses) - SIMPLEST
  python data/generate_data_plots.py
  
  # Generate plots for train data
  python data/generate_data_plots.py --mode train
  
  # Specific bus systems only
  python data/generate_data_plots.py --buses 33,57
  
  # Custom output directory
  python data/generate_data_plots.py --output ./my_plots
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='test',
        choices=['train', 'test'],
        help='Data mode: train or test (default: test)'
    )
    
    parser.add_argument(
        '--buses',
        type=str,
        default='all',
        help='Bus systems to plot (e.g., "33,57,118" or "all")'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for plots (default: data/plots_{mode})'
    )
    
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Keep old plots instead of cleaning up'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config.yaml file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = Config(
        yaml_config_path=args.config,
        load_yaml=True,
        data_mode=args.mode
    )
    
    # Parse bus systems
    available_buses = config.NUM_BUSES
    bus_systems = parse_bus_systems(args.buses, available_buses)
    
    if not bus_systems:
        print("Error: No valid bus systems specified")
        sys.exit(1)
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        # Default: data/plots_{mode}
        # DATA_DIR is typically data/processed/{mode} or data/{mode}
        # We want data/plots_{mode}
        
        # Robust way: go to 'data' folder and add 'plots_{mode}'
        # Assuming config.DATA_DIR is absolute path to data folder
        # Let's use the project root relative path for safety
        data_root = os.path.join(project_root, 'data')
        output_dir = os.path.join(data_root, f'plots_{args.mode}')
    
    # Clean up old plots unless --no-cleanup is specified
    if not args.no_cleanup and os.path.exists(output_dir):
        print(f"Cleaning up old plots in {output_dir}...")
        try:
            shutil.rmtree(output_dir)
        except Exception as e:
            print(f"Warning: Could not clean up old plots: {e}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Print summary
    print("\n" + "="*80)
    print(f"DATA VISUALIZATION - {args.mode.upper()} MODE")
    print("="*80)
    print(f"Bus systems: {bus_systems}")
    print("="*80 + "\n")
    
    # Generate plots
    try:
        plot_paths = generate_all_data_plots(
            config=config,
            bus_systems=bus_systems,
            data_plots_dir=output_dir
        )
        
        # Print summary of generated plots
        print("\n" + "="*80)
        print("PLOT GENERATION COMPLETE")
        print("="*80)
        total_plots = sum(len([p for p in plots.values() if p]) for plots in plot_paths.values())
        print(f"Total plots generated: {total_plots}")
        
        for bus_num, plots in plot_paths.items():
            successful = len([p for p in plots.values() if p])
            print(f"  {bus_num}-bus: {successful} plots")
        
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nError generating plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
