"""
Standalone script to generate ALL data visualization plots (YAML-Compatible).
"""

import os
import sys
import argparse
import shutil
import yaml

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from visualization.plot_consolidator import generate_all_data_plots

def load_data_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Generate data visualization plots')
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--buses', '--case', type=str, default='all', help='Bus systems (e.g., "33,57")')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--config', type=str, default='data/data_generation.yaml', help='Path to config YAML')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_data_config(args.config)
    
    # Parse bus systems
    available_buses = [33, 57, 118] # Standard cases
    if args.buses.lower() == 'all':
        bus_systems = available_buses
    else:
        bus_systems = [int(b.strip()) for b in args.buses.split(',') if b.strip().isdigit()]
    
    # Determine output directory
    output_dir = args.output or config.get('reports_dir', 'reports/figures/01_raw_data')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"DATA VISUALIZATION - {args.mode.upper()} MODE")
    print("="*80)
    print(f"Bus systems: {bus_systems}")
    print(f"Output: {output_dir}")
    print("="*80 + "\n")
    
    # Generate plots
    try:
        plot_paths = generate_all_data_plots(
            config=config,
            bus_systems=bus_systems,
            data_plots_dir=output_dir
        )
        print("\nPlot generation complete.")
    except Exception as e:
        print(f"\nError generating plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
