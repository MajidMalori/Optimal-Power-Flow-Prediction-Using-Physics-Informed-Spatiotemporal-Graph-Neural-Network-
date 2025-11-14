"""
Convergence Story: Comprehensive Convergence Analysis Visualization

This script tells the complete story of load flow convergence:
- Success rates across renewable penetration levels
- Convergence statistics and failure counts
- Resolution methods distribution
- Summary statistics

Purpose: When someone asks "How reliable is your data generation?", 
         you can show them these graphs.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

def load_convergence_reports(data_dir: str, mode: str = 'train'):
    """Load all convergence report JSON files."""
    # Convergence reports are saved directly in data_dir (not in time_series subfolder)
    # Pattern: {case_name}_convergence_report_frac{renewable_fraction:.1f}_{timestamp}.json
    search_pattern = os.path.join(data_dir, '*_convergence_report_*.json')
    
    report_files = glob.glob(search_pattern)
    reports = {}
    
    for report_file in report_files:
        filename = os.path.basename(report_file)
        parts = filename.replace('_convergence_report_', '_').replace('.json', '').split('_')
        
        case_name = None
        renewable_fraction = None
        
        for i, part in enumerate(parts):
            if part.startswith('case') and part[4:].isdigit():
                case_name = part
            elif part.startswith('frac'):
                try:
                    renewable_fraction = float(part[4:])
                except:
                    pass
        
        if case_name and renewable_fraction is not None:
            key = (case_name, renewable_fraction)
            with open(report_file, 'r') as f:
                reports[key] = json.load(f)
    
    return reports

def analyze_convergence_story(config, case_name: str, data_dir: str = None, mode: str = None):
    """
    Analyze and visualize convergence failures for a specific case.
    Creates a single high-quality 2x2 grid plot (like data profile story).
    
    Args:
        config: Configuration object
        case_name: Case name (e.g., 'case33')
        data_dir: Optional data directory (uses config if not provided)
        mode: Optional data mode (uses config if not provided)
    
    Returns:
        List of issues found (empty if no issues)
    """
    print(f"Generating Convergence Story for {case_name}")
    
    if data_dir is None:
        data_dir = config.DATA_DIR
    if mode is None:
        mode = config.DATA_MODE
    
    try:
        reports = load_convergence_reports(data_dir, mode)
        
        if not reports:
            return []
        
        # Filter by case_name (CRITICAL: only show data for this case)
        reports = {k: v for k, v in reports.items() if k[0] == case_name}
        
        if not reports:
            # No data available - create placeholder
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Convergence Story - {case_name.upper()}', fontsize=18, fontweight='bold')
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No convergence data available', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save to current run directory
            case_num = case_name.replace('case', '')
            try:
                if hasattr(config, '_CURRENT_RUN_TIMESTAMP') and config._CURRENT_RUN_TIMESTAMP:
                    current_run_dir = config.CURRENT_RUN_DIR
                    if current_run_dir:
                        output_dir = os.path.join(current_run_dir, f"{case_num}bus")
                    else:
                        output_dir = os.path.join(config.EXPERIMENTAL_RESULTS_DIR, f"{case_num}_bus")
                else:
                    output_dir = os.path.join(config.EXPERIMENTAL_RESULTS_DIR, f"{case_num}_bus")
            except (AttributeError, TypeError):
                output_dir = os.path.join(config.EXPERIMENTAL_RESULTS_DIR, f"{case_num}_bus")
            
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'convergence_story.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return []
        
        # Prepare data (only for the filtered case)
        data_rows = []
        for (case_key, renewable_fraction), stats in reports.items():
            total = stats.get('total_timesteps', 0)
            successful = stats.get('successful', 0)
            failed = stats.get('failed', 0)
            success_rate = (successful / total * 100) if total > 0 else 0.0
            
            resolution_methods = stats.get('resolution_methods', {})
            
            data_rows.append({
                'case': case_key,
                'renewable_fraction': renewable_fraction,
                'renewable_percent': renewable_fraction * 100,
                'total': total,
                'successful': successful,
                'failed': failed,
                'success_rate': success_rate,
                'strict_normal': resolution_methods.get('strict_normal', 0),
                'strict_contingency': resolution_methods.get('strict_contingency', 0),
                'relaxed_contingency': resolution_methods.get('relaxed_contingency', 0),
                'restored_line': resolution_methods.get('restored_line', 0),
            })
        
        df = pd.DataFrame(data_rows)
        case_display = case_name.upper().replace('CASE', '')
        
        # Create single figure with 2x2 grid (match data profile story style)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Convergence Story - {case_display}', fontsize=18, fontweight='bold')
        
        # ========================================================================
        # TOP-LEFT: Success Rate by Renewable Penetration (Line plot)
        # ========================================================================
        ax = axes[0, 0]
        df_sorted = df.sort_values('renewable_percent')
        ax.plot(df_sorted['renewable_percent'], df_sorted['success_rate'], 
               'o-', linewidth=2.5, markersize=10, color='#2ecc71', 
               markerfacecolor='white', markeredgewidth=2, markeredgecolor='#2ecc71')
        ax.axhline(y=100, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xlabel('Renewable Penetration (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
        ax.set_title('Convergence Success Rate', fontweight='bold', fontsize=12)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_ylim(max(95, df_sorted['success_rate'].min() - 1), 100.5)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for _, row in df_sorted.iterrows():
            ax.text(row['renewable_percent'], row['success_rate'] + 0.3, 
                   f"{row['success_rate']:.2f}%", ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        # ========================================================================
        # TOP-RIGHT: Convergence by Renewable Penetration (Stacked bar)
        # ========================================================================
        ax = axes[0, 1]
        df_sorted = df.sort_values('renewable_percent')
        renewable_percents = df_sorted['renewable_percent'].values
        successful = df_sorted['successful'].values
        failed = df_sorted['failed'].values
        
        x = np.arange(len(renewable_percents))
        width = 0.6
        
        ax.bar(x, successful, width, label='Converged', color='#2ecc71', edgecolor='black', linewidth=1.2)
        ax.bar(x, failed, width, bottom=successful, label='Failed', color='#e74c3c', edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Renewable Penetration (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Timesteps', fontsize=11, fontweight='bold')
        ax.set_title('Convergence by Renewable Penetration', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{f:.0f}%' for f in renewable_percents])
        ax.legend(loc='best', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (tot, fail) in enumerate(zip(successful + failed, failed)):
            if tot > 0:
                ax.text(i, tot + max(tot * 0.02, 10), f'{int(tot):,}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=9)
            if fail > 0:
                ax.text(i, successful[i] + fail/2, f'{int(fail):,}', ha='center', va='center',
                       fontweight='bold', color='white', fontsize=9)
        
        # ========================================================================
        # BOTTOM-LEFT: Resolution Methods Distribution (Pie chart)
        # ========================================================================
        ax = axes[1, 0]
        
        # Aggregate resolution methods
        total_methods = {
            'Strict (Normal)': df['strict_normal'].sum(),
            'Strict (Contingency)': df['strict_contingency'].sum(),
            'Relaxed (Contingency)': df['relaxed_contingency'].sum(),
            'Restored Line': df['restored_line'].sum(),
            'Failed': df['failed'].sum(),
        }
        
        # Remove zero values for cleaner pie chart
        total_methods = {k: v for k, v in total_methods.items() if v > 0}
        
        if total_methods:
            colors = ['#27ae60', '#3498db', '#f39c12', '#9b59b6', '#e74c3c']
            color_map = dict(zip(['Strict (Normal)', 'Strict (Contingency)', 'Relaxed (Contingency)', 'Restored Line', 'Failed'], colors))
            pie_colors = [color_map.get(k, '#95a5a6') for k in total_methods.keys()]
            
            explode = tuple(0.1 if 'Failed' in k else 0 for k in total_methods.keys())
            
            wedges, texts, autotexts = ax.pie(total_methods.values(), labels=total_methods.keys(), 
                                              autopct='%1.1f%%', startangle=90, colors=pie_colors,
                                              explode=explode, textprops={'fontsize': 10, 'fontweight': 'bold'})
            
            # Make percentage text white for better visibility
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(9)
            
            # Add total count in center
            total = sum(total_methods.values())
            ax.text(0, 0, f'Total:\n{total:,}\nTimesteps', ha='center', va='center',
                    fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'No resolution data available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, fontweight='bold')
        
        ax.set_title('Resolution Methods Distribution', fontweight='bold', fontsize=12)
        
        # ========================================================================
        # BOTTOM-RIGHT: Summary Statistics (Text/Table)
        # ========================================================================
        ax = axes[1, 1]
        ax.axis('off')
        
        # Calculate summary statistics
        total_timesteps = df['total'].sum()
        total_successful = df['successful'].sum()
        total_failed = df['failed'].sum()
        overall_success_rate = (total_successful / total_timesteps * 100) if total_timesteps > 0 else 0.0
        
        # Create summary text
        summary_text = f"Overall Statistics\n{'='*30}\n\n"
        summary_text += f"Total Timesteps: {total_timesteps:,}\n"
        summary_text += f"Successful: {total_successful:,} ({overall_success_rate:.2f}%)\n"
        summary_text += f"Failed: {total_failed:,} ({100-overall_success_rate:.2f}%)\n\n"
        
        summary_text += f"By Renewable Penetration:\n{'-'*30}\n"
        for _, row in df.sort_values('renewable_percent').iterrows():
            summary_text += f"{row['renewable_percent']:.0f}%: {row['success_rate']:.2f}% "
            summary_text += f"({row['successful']:,}/{row['total']:,})\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        ax.set_title('Summary Statistics', fontweight='bold', fontsize=12)
        
        # Layout for single figure (match data profile story style)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save to current run directory (in run_XXXXXX/XXbus folder, not experimental_results/XXbus)
        case_num = case_name.replace('case', '')
        try:
            if hasattr(config, '_CURRENT_RUN_TIMESTAMP') and config._CURRENT_RUN_TIMESTAMP:
                current_run_dir = config.CURRENT_RUN_DIR
                if current_run_dir and os.path.exists(os.path.dirname(current_run_dir)):
                    output_dir = os.path.join(current_run_dir, f"{case_num}bus")
                else:
                    output_dir = os.path.join(config.EXPERIMENTAL_RESULTS_DIR, f"{case_num}_bus")
            else:
                output_dir = os.path.join(config.EXPERIMENTAL_RESULTS_DIR, f"{case_num}_bus")
        except (AttributeError, TypeError):
            output_dir = os.path.join(config.EXPERIMENTAL_RESULTS_DIR, f"{case_num}_bus")
        
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'convergence_story.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return []
        
    except Exception as e:
        print(f"  Warning: Could not generate convergence story: {e}")
        import traceback
        traceback.print_exc()
        return [f"Convergence story generation failed: {e}"]

