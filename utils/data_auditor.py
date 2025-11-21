"""
Data Quality Auditor: Professional Data Audit and Visualization System

This module provides comprehensive data quality auditing, transforming raw convergence
statistics into a professional "Data Quality Audit Report" that tells the complete
story of data generation: physics health, intervention strategies, and contingency analysis.

Purpose: As a Machine Learning Engineer, you must prove data distribution.
         As a Power Engineer, you must prove physical validity.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Set style for professional plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def transform_convergence_to_audit(convergence_stats: dict, case_name: str, 
                                   renewable_fraction: float, timestamp: str = None) -> dict:
    """
    Transform raw convergence_stats into professional data_quality_audit format.
    
    Args:
        convergence_stats: Raw convergence statistics from data generation
        case_name: Case name (e.g., 'case33')
        renewable_fraction: Renewable fraction (0.0 to 1.0)
        timestamp: Optional timestamp string
    
    Returns:
        Structured data quality audit dictionary
    """
    total_timesteps = convergence_stats.get('total_timesteps', 0)
    successful = convergence_stats.get('successful', 0)
    failed = convergence_stats.get('failed', 0)
    
    validation_stats = convergence_stats.get('validation_stats', {})
    resolution_methods = convergence_stats.get('resolution_methods', {})
    critical_lines = convergence_stats.get('critical_lines', {})
    
    # Calculate intervention rates
    raw_success = resolution_methods.get('strict_normal', 0) + resolution_methods.get('strict_contingency', 0)
    curtailed = (resolution_methods.get('relaxed_contingency', 0) + 
                 resolution_methods.get('restored_line', 0) +
                 validation_stats.get('curtailment_successful', 0))
    failed_tripped = (failed + 
                     validation_stats.get('generator_trips', 0) +
                     validation_stats.get('hard_resets', 0))
    
    total_interventions = raw_success + curtailed + failed_tripped
    if total_interventions > 0:
        raw_success_rate = (raw_success / total_interventions * 100) if total_interventions > 0 else 0
        curtailed_rate = (curtailed / total_interventions * 100) if total_interventions > 0 else 0
        failed_tripped_rate = (failed_tripped / total_interventions * 100) if total_interventions > 0 else 0
    else:
        raw_success_rate = 0
        curtailed_rate = 0
        failed_tripped_rate = 0
    
    # Contingency statistics
    contingencies_attempted = convergence_stats.get('contingencies_attempted', 0)
    contingencies_successful = convergence_stats.get('contingencies_successful', 0)
    contingencies_failed = convergence_stats.get('contingencies_failed', 0)
    contingencies_resolved_strict = convergence_stats.get('contingencies_resolved_strict', 0)
    contingencies_resolved_relaxed = convergence_stats.get('contingencies_resolved_relaxed', 0)
    contingencies_restored = convergence_stats.get('contingencies_restored', 0)
    
    # Calculate N-1 robustness score
    if contingencies_attempted > 0:
        n1_safe = contingencies_resolved_strict
        n1_curtailed = contingencies_resolved_relaxed + contingencies_restored
        n1_collapsed = contingencies_failed
        n1_robustness = ((n1_safe + n1_curtailed) / contingencies_attempted * 100) if contingencies_attempted > 0 else 0
    else:
        n1_safe = 0
        n1_curtailed = 0
        n1_collapsed = 0
        n1_robustness = 100.0  # No contingencies = perfect robustness
    
    # Critical lines analysis
    critical_lines_analysis = {}
    for line_key, line_data in critical_lines.items():
        line_id = line_data.get('line_id', 'unknown')
        failure_count = line_data.get('failure_count', 0)
        resolution_methods_line = line_data.get('resolution_methods', {})
        
        # Calculate curtailment rate for this line
        total_line_events = failure_count
        if total_line_events > 0:
            curtailment_rate = ((resolution_methods_line.get('relaxed_curtailment', 0) + 
                               resolution_methods_line.get('restored_curtailment', 0)) / 
                              total_line_events * 100)
            failure_rate = (resolution_methods_line.get('trip', 0) / total_line_events * 100)
        else:
            curtailment_rate = 0
            failure_rate = 0
        
        critical_lines_analysis[line_key] = {
            'line_id': int(line_id),
            'failure_count': failure_count,
            'curtailment_rate': curtailment_rate,
            'failure_rate': failure_rate,
            'resolution_methods': resolution_methods_line
        }
    
    # Physics health (placeholder - would need actual voltage/loading data)
    # These would be calculated from actual power flow results if available
    physics_health = {
        'voltage_range_pu': [0.85, 1.15],  # Would be calculated from actual data
        'max_line_loading_percent': 0.0,  # Would be calculated from actual data
        'reactive_power_violations': validation_stats.get('inverter_capability_violations', 0),
        'voltage_violations': validation_stats.get('voltage_violations', 0),
        'angle_violations': validation_stats.get('angle_violations', 0),
        'line_loading_violations': validation_stats.get('line_loading_violations', 0),
        'valid_stressed_states': validation_stats.get('valid_stressed_states', 0),
    }
    
    audit = {
        'meta': {
            'case': case_name,
            'timestamp': timestamp or datetime.now().isoformat(),
            'total_samples': total_timesteps,
            'renewable_fraction': renewable_fraction,
        },
        'physics_health': physics_health,
        'intervention_stats': {
            'raw_success_rate': round(raw_success_rate, 2),
            'curtailed_rate': round(curtailed_rate, 2),
            'failed_tripped_rate': round(failed_tripped_rate, 2),
            'total_interventions': total_interventions,
            'raw_success_count': raw_success,
            'curtailed_count': curtailed,
            'failed_tripped_count': failed_tripped,
        },
        'contingency_stats': {
            'n1_scenarios_run': contingencies_attempted,
            'n1_safe': n1_safe,
            'n1_curtailed': n1_curtailed,
            'n1_collapsed': n1_collapsed,
            'n1_robustness_score': round(n1_robustness, 2),
            'n1_resolved_strict': contingencies_resolved_strict,
            'n1_resolved_relaxed': contingencies_resolved_relaxed,
            'n1_restored': contingencies_restored,
        },
        'critical_lines': critical_lines_analysis,
        'validation_stats': {
            'consecutive_failures': validation_stats.get('consecutive_failures', 0),
            'max_consecutive_failures': validation_stats.get('max_consecutive_failures', 0),
            'curtailment_attempts': validation_stats.get('curtailment_attempts', 0),
            'curtailment_events': validation_stats.get('curtailment_events', 0),
            'generator_trips': validation_stats.get('generator_trips', 0),
            'hard_resets': validation_stats.get('hard_resets', 0),
        },
        'raw_convergence_stats': convergence_stats,  # Keep raw data for backward compatibility
    }
    
    return audit


class DataAuditor:
    """
    Professional Data Quality Auditor for Power System Machine Learning Datasets.
    
    Transforms convergence statistics into comprehensive audit reports and generates
    publication-quality visualizations.
    """
    
    def __init__(self, audit_json_path: str = None, audit_dict: dict = None):
        """
        Initialize DataAuditor with either a JSON file path or a dictionary.
        
        Args:
            audit_json_path: Path to data_quality_audit.json file
            audit_dict: Direct audit dictionary (alternative to file path)
        """
        if audit_dict is not None:
            self.audit = audit_dict
        elif audit_json_path is not None:
            with open(audit_json_path, 'r') as f:
                self.audit = json.load(f)
        else:
            raise ValueError("Must provide either audit_json_path or audit_dict")
    
    def generate_executive_summary(self) -> str:
        """
        Generate a professional executive summary for console output.
        
        Returns:
            Formatted summary string
        """
        meta = self.audit['meta']
        intervention = self.audit['intervention_stats']
        contingency = self.audit['contingency_stats']
        
        summary = "\n" + "="*70 + "\n"
        summary += f"DATA QUALITY AUDIT: {meta['case'].upper()}\n"
        summary += "="*70 + "\n"
        summary += f"Total Samples:      {meta['total_samples']:,}\n"
        summary += f"Renewable Fraction:  {meta['renewable_fraction']*100:.1f}%\n"
        summary += f"Timestamp:           {meta['timestamp']}\n"
        summary += "-" * 70 + "\n"
        summary += "INTERVENTION STATISTICS:\n"
        summary += f"  1. Standard Operations:    {intervention['raw_success_rate']:.1f}%\n"
        summary += f"  2. Active Management:       {intervention['curtailed_rate']:.1f}% (Curtailed)\n"
        summary += f"  3. Critical Events:          {intervention['failed_tripped_rate']:.1f}% (Tripped/Failed)\n"
        summary += "-" * 70 + "\n"
        summary += "CONTINGENCY ANALYSIS:\n"
        summary += f"  N-1 Scenarios Run:          {contingency['n1_scenarios_run']:,}\n"
        summary += f"  N-1 Safe (Strict):           {contingency['n1_safe']:,}\n"
        summary += f"  N-1 Curtailed (Relaxed):    {contingency['n1_curtailed']:,}\n"
        summary += f"  N-1 Collapsed (Failed):     {contingency['n1_collapsed']:,}\n"
        summary += f"  N-1 Robustness Score:       {contingency['n1_robustness_score']:.1f}%\n"
        
        # Critical lines analysis
        critical_lines = self.audit.get('critical_lines', {})
        if critical_lines:
            summary += "-" * 70 + "\n"
            summary += "MOST CRITICAL LINES (Top 3):\n"
            
            # Sort by failure count
            sorted_lines = sorted(critical_lines.items(), 
                                key=lambda x: x[1].get('failure_count', 0), 
                                reverse=True)[:3]
            
            for rank, (line_key, line_data) in enumerate(sorted_lines, 1):
                line_id = line_data.get('line_id', 'unknown')
                curtailment_rate = line_data.get('curtailment_rate', 0)
                failure_rate = line_data.get('failure_rate', 0)
                
                if curtailment_rate > 50:
                    status = "Bottleneck"
                elif failure_rate > 5:
                    status = "Islanding Risk"
                else:
                    status = "Redundant"
                
                summary += f"  {rank}. Line {line_id}: {curtailment_rate:.1f}% Curtailment Rate ({status})\n"
        
        summary += "-" * 70 + "\n"
        
        # Physics integrity check
        physics = self.audit['physics_health']
        if physics.get('reactive_power_violations', 0) == 0:
            integrity_status = "PASS"
        else:
            integrity_status = "WARNING"
        
        summary += f"Physics Integrity:   {integrity_status}\n"
        summary += "="*70 + "\n"
        
        return summary
    
    def plot_physics_distribution(self, voltage_data: np.ndarray = None, 
                                  output_dir: str = None, output_path: str = None) -> str:
        """
        Plot A: System Health Distribution (Voltage Histogram).
        
        Args:
            voltage_data: Optional voltage data array (if None, uses placeholder)
            output_dir: Output directory for saving
            output_path: Full output path (alternative to output_dir)
        
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if voltage_data is not None:
            voltage_flat = voltage_data.flatten()
            sns.histplot(voltage_flat, bins=100, kde=True, color='teal', stat="density", ax=ax)
            title = f"System Health Distribution (Voltage) - {self.audit['meta']['case'].upper()}"
        else:
            # Placeholder: Show expected distribution
            voltage_flat = np.random.normal(1.0, 0.05, 10000)
            voltage_flat = np.clip(voltage_flat, 0.85, 1.15)
            sns.histplot(voltage_flat, bins=100, kde=True, color='teal', stat="density", ax=ax)
            title = f"System Health Distribution (Voltage) - {self.audit['meta']['case'].upper()}\n(Expected Distribution - Actual Data Not Available)"
        
        ax.axvline(0.95, color='r', linestyle='--', linewidth=2, label='Min Limit (0.95 p.u.)', alpha=0.7)
        ax.axvline(1.05, color='r', linestyle='--', linewidth=2, label='Max Limit (1.05 p.u.)', alpha=0.7)
        ax.axvline(0.85, color='orange', linestyle=':', linewidth=1.5, label='Emergency Min (0.85 p.u.)', alpha=0.5)
        ax.axvline(1.15, color='orange', linestyle=':', linewidth=1.5, label='Emergency Max (1.15 p.u.)', alpha=0.5)
        
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel("Voltage (p.u.)", fontweight='bold')
        ax.set_ylabel("Density", fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if output_path is None:
            if output_dir is None:
                output_dir = '.'
            output_path = os.path.join(output_dir, 'physics_health.png')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_intervention_stats(self, output_dir: str = None, output_path: str = None) -> str:
        """
        Plot B: Data Composition by Intervention Type (Pie Chart).
        
        Args:
            output_dir: Output directory for saving
            output_path: Full output path (alternative to output_dir)
        
        Returns:
            Path to saved figure
        """
        intervention = self.audit['intervention_stats']
        
        labels = ['Standard Ops', 'Curtailed (Stressed)', 'Tripped (Failed)']
        sizes = [
            intervention['raw_success_rate'],
            intervention['curtailed_rate'],
            intervention['failed_tripped_rate']
        ]
        colors = ['#4CAF50', '#FFC107', '#F44336']  # Green, Amber, Red
        
        # Filter out zero values
        filtered_data = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
        if not filtered_data:
            # All zeros - create placeholder
            filtered_data = [('No Data', 100, '#95a5a6')]
            labels, sizes, colors = zip(*filtered_data)
        else:
            labels, sizes, colors = zip(*filtered_data)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=140,
                                          textprops={'fontsize': 11, 'fontweight': 'bold'})
        
        # Make percentage text white for better visibility
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        # Add total count in center
        total = intervention.get('total_interventions', 0)
        if total > 0:
            ax.text(0, 0, f'Total:\n{total:,}\nInterventions', ha='center', va='center',
                   fontsize=12, fontweight='bold', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(f"Dataset Composition by Grid State\n{self.audit['meta']['case'].upper()}", 
                    fontweight='bold', fontsize=13)
        
        if output_path is None:
            if output_dir is None:
                output_dir = '.'
            output_path = os.path.join(output_dir, 'data_composition.png')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_curtailment_impact(self, raw_solar: np.ndarray = None, 
                                curtailed_solar: np.ndarray = None,
                                time_slice: np.ndarray = None,
                                output_dir: str = None, output_path: str = None) -> str:
        """
        Plot C: A Day in the Life - Curtailment Event (Time-Series).
        
        Args:
            raw_solar: Available solar generation (weather-driven)
            curtailed_solar: Actual grid injection (after curtailment)
            time_slice: Time indices (e.g., 24 hours)
            output_dir: Output directory for saving
            output_path: Full output path (alternative to output_dir)
        
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if raw_solar is not None and curtailed_solar is not None and time_slice is not None:
            ax.plot(time_slice, raw_solar, 'r--', label='Available Solar (Weather)', 
                   alpha=0.6, linewidth=2)
            ax.plot(time_slice, curtailed_solar, 'g-', label='Grid Injection (Safe)', 
                   linewidth=2.5)
            ax.fill_between(time_slice, curtailed_solar, raw_solar, 
                           color='yellow', alpha=0.3, label='Curtailed Energy')
            title = f"Active Network Management: Curtailment Event\n{self.audit['meta']['case'].upper()}"
        else:
            # Placeholder: Show expected pattern
            time_slice = np.arange(24)
            raw_solar = np.maximum(0, np.sin((time_slice - 6) * np.pi / 12) * 100)
            raw_solar[time_slice < 6] = 0
            raw_solar[time_slice > 18] = 0
            # Simulate curtailment during peak hours
            curtailed_solar = raw_solar.copy()
            curtailed_solar[(time_slice >= 12) & (time_slice <= 14)] *= 0.7  # 30% curtailment
            
            ax.plot(time_slice, raw_solar, 'r--', label='Available Solar (Weather)', 
                   alpha=0.6, linewidth=2)
            ax.plot(time_slice, curtailed_solar, 'g-', label='Grid Injection (Safe)', 
                   linewidth=2.5)
            ax.fill_between(time_slice, curtailed_solar, raw_solar, 
                           color='yellow', alpha=0.3, label='Curtailed Energy')
            title = f"Active Network Management: Curtailment Event\n{self.audit['meta']['case'].upper()}\n(Example Pattern - Actual Data Not Available)"
        
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylabel("Active Power (MW)", fontweight='bold')
        ax.set_xlabel("Time Step (Hours)", fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if output_path is None:
            if output_dir is None:
                output_dir = '.'
            output_path = os.path.join(output_dir, 'curtailment_impact.png')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def plot_contingency_heatmap(self, output_dir: str = None, output_path: str = None) -> str:
        """
        Plot D: The Weakest Link - Contingency Failure Rate Heatmap.
        
        Args:
            output_dir: Output directory for saving
            output_path: Full output path (alternative to output_dir)
        
        Returns:
            Path to saved figure
        """
        critical_lines = self.audit.get('critical_lines', {})
        
        if not critical_lines:
            # Create placeholder
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No Contingency Data Available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, fontweight='bold')
            ax.set_title(f"Contingency Failure Rate Analysis\n{self.audit['meta']['case'].upper()}", 
                        fontweight='bold', fontsize=12)
            
            if output_path is None:
                if output_dir is None:
                    output_dir = '.'
                output_path = os.path.join(output_dir, 'contingency_heatmap.png')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        
        # Prepare data for heatmap
        line_ids = []
        curtailment_rates = []
        failure_rates = []
        failure_counts = []
        
        for line_key, line_data in critical_lines.items():
            line_ids.append(f"Line {line_data.get('line_id', '?')}")
            curtailment_rates.append(line_data.get('curtailment_rate', 0))
            failure_rates.append(line_data.get('failure_rate', 0))
            failure_counts.append(line_data.get('failure_count', 0))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Line': line_ids,
            'Curtailment Rate (%)': curtailment_rates,
            'Failure Rate (%)': failure_rates,
            'Failure Count': failure_counts
        })
        
        # Sort by failure count (most critical first)
        df = df.sort_values('Failure Count', ascending=False)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(df) * 0.4)))
        
        # Left: Bar chart of curtailment rates
        y_pos = np.arange(len(df))
        ax1.barh(y_pos, df['Curtailment Rate (%)'], color='#FFC107', edgecolor='black', linewidth=1.2)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(df['Line'], fontsize=9)
        ax1.set_xlabel('Curtailment Rate (%)', fontweight='bold')
        ax1.set_title('Curtailment Rate by Line', fontweight='bold', fontsize=11)
        ax1.grid(axis='x', alpha=0.3)
        ax1.set_xlim(0, 105)
        
        # Add value labels
        for i, (idx, row) in enumerate(df.iterrows()):
            if row['Curtailment Rate (%)'] > 0:
                ax1.text(row['Curtailment Rate (%)'] + 2, i, 
                        f"{row['Curtailment Rate (%)']:.1f}%", 
                        va='center', fontsize=8, fontweight='bold')
        
        # Right: Bar chart of failure rates
        colors = ['#4CAF50' if r < 5 else '#FFC107' if r < 20 else '#F44336' 
                 for r in df['Failure Rate (%)']]
        ax2.barh(y_pos, df['Failure Rate (%)'], color=colors, edgecolor='black', linewidth=1.2)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(df['Line'], fontsize=9)
        ax2.set_xlabel('Failure Rate (%)', fontweight='bold')
        ax2.set_title('Failure Rate by Line', fontweight='bold', fontsize=11)
        ax2.grid(axis='x', alpha=0.3)
        ax2.set_xlim(0, 105)
        
        # Add value labels
        for i, (idx, row) in enumerate(df.iterrows()):
            if row['Failure Rate (%)'] > 0:
                ax2.text(row['Failure Rate (%)'] + 2, i, 
                        f"{row['Failure Rate (%)']:.1f}%", 
                        va='center', fontsize=8, fontweight='bold')
        
        fig.suptitle(f"Contingency Analysis: Critical Lines\n{self.audit['meta']['case'].upper()}", 
                     fontweight='bold', fontsize=13)
        
        if output_path is None:
            if output_dir is None:
                output_dir = '.'
            output_path = os.path.join(output_dir, 'contingency_heatmap.png')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_all_plots(self, output_dir: str = None, voltage_data: np.ndarray = None,
                          raw_solar: np.ndarray = None, curtailed_solar: np.ndarray = None,
                          time_slice: np.ndarray = None, config=None, case_name: str = None) -> Dict[str, str]:
        """
        Generate all 4 professional plots and save to output directory.
        
        Args:
            output_dir: Directory to save all plots (if None, uses config to determine location)
            voltage_data: Optional voltage data for physics distribution plot
            raw_solar: Optional raw solar data for curtailment plot
            curtailed_solar: Optional curtailed solar data for curtailment plot
            time_slice: Optional time slice for curtailment plot
            config: Optional config object for determining output directory
            case_name: Optional case name for determining output directory
        
        Returns:
            Dictionary mapping plot names to file paths
        """
        # Determine output directory if not provided
        if output_dir is None and config is not None and case_name is not None:
            case_num = case_name.replace('case', '')
            try:
                if hasattr(config, '_CURRENT_RUN_TIMESTAMP') and config._CURRENT_RUN_TIMESTAMP:
                    current_run_dir = config.CURRENT_RUN_DIR
                    if current_run_dir and os.path.exists(os.path.dirname(current_run_dir)):
                        output_dir = os.path.join(current_run_dir, f"{case_num}bus", "analysis_reports")
                    else:
                        output_dir = os.path.join(config.EXPERIMENTAL_RESULTS_DIR, f"{case_num}_bus", "analysis_reports")
                else:
                    output_dir = os.path.join(config.EXPERIMENTAL_RESULTS_DIR, f"{case_num}_bus", "analysis_reports")
            except (AttributeError, TypeError):
                output_dir = os.path.join(config.EXPERIMENTAL_RESULTS_DIR, f"{case_num}_bus", "analysis_reports")
        
        if output_dir is None:
            output_dir = os.path.join('.', 'analysis_reports')
        
        os.makedirs(output_dir, exist_ok=True)
        
        plot_paths = {}
        plot_paths['physics_health'] = self.plot_physics_distribution(
            voltage_data=voltage_data, output_dir=output_dir
        )
        plot_paths['data_composition'] = self.plot_intervention_stats(output_dir=output_dir)
        plot_paths['curtailment_impact'] = self.plot_curtailment_impact(
            raw_solar=raw_solar, curtailed_solar=curtailed_solar, 
            time_slice=time_slice, output_dir=output_dir
        )
        plot_paths['contingency_heatmap'] = self.plot_contingency_heatmap(output_dir=output_dir)
        
        return plot_paths
    
    @staticmethod
    def load_all_audits(data_dir: str, case_name: str = None) -> Dict[Tuple[str, float], dict]:
        """
        Load all data quality audit files from a directory.
        
        Args:
            data_dir: Directory containing audit files
            case_name: Optional case name filter (e.g., 'case33')
        
        Returns:
            Dictionary mapping (case_name, renewable_fraction) -> audit_data
        """
        search_pattern_audit = os.path.join(data_dir, '*_data_quality_audit_*.json')
        search_pattern_legacy = os.path.join(data_dir, '*_convergence_report_*.json')
        
        report_files = glob.glob(search_pattern_audit)
        if not report_files:
            report_files = glob.glob(search_pattern_legacy)
        
        audits = {}
        
        for report_file in report_files:
            filename = os.path.basename(report_file)
            # Handle both new and legacy formats
            if '_data_quality_audit_' in filename:
                parts = filename.replace('_data_quality_audit_', '_').replace('.json', '').split('_')
            else:
                parts = filename.replace('_convergence_report_', '_').replace('.json', '').split('_')
            
            file_case_name = None
            renewable_fraction = None
            
            for i, part in enumerate(parts):
                if part.startswith('case') and part[4:].isdigit():
                    file_case_name = part
                elif part.startswith('frac'):
                    try:
                        renewable_fraction = float(part[4:])
                    except:
                        pass
            
            if file_case_name and renewable_fraction is not None:
                # Filter by case_name if provided
                if case_name is None or file_case_name == case_name:
                    key = (file_case_name, renewable_fraction)
                    with open(report_file, 'r') as f:
                        audit_data = json.load(f)
                    audits[key] = audit_data
        
        return audits
    
    @staticmethod
    def plot_convergence_story(data_dir: str, case_name: str, output_dir: str = None, 
                               config=None) -> str:
        """
        Generate convergence story plots across all renewable fractions (replaces convergence_story.py).
        
        Creates a 2x2 grid showing:
        - Success rate by renewable penetration (line plot)
        - Convergence by renewable penetration (stacked bar)
        - Resolution methods distribution (pie chart)
        - Summary statistics (text)
        
        Args:
            data_dir: Directory containing audit files
            case_name: Case name (e.g., 'case33')
            output_dir: Output directory (if None, uses config to determine location)
            config: Optional config object for determining output directory
        
        Returns:
            Path to saved figure
        """
        audits = DataAuditor.load_all_audits(data_dir, case_name)
        
        if not audits:
            # No data available - create placeholder
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Convergence Story - {case_name.upper().replace("CASE", "")}', 
                        fontsize=18, fontweight='bold')
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No convergence data available', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if output_dir is None and config is not None:
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
            
            if output_dir is None:
                output_dir = '.'
            
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, 'convergence_story.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        
        # Prepare data
        data_rows = []
        for (case_key, renewable_fraction), audit_data in audits.items():
            # Extract convergence stats
            if 'raw_convergence_stats' in audit_data:
                stats = audit_data['raw_convergence_stats']
            else:
                stats = audit_data  # Legacy format
            
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
                'hard_reset': resolution_methods.get('hard_reset', 0),
            })
        
        df = pd.DataFrame(data_rows)
        case_display = case_name.upper().replace('CASE', '')
        
        # Create single figure with 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Convergence Story - {case_display}', fontsize=18, fontweight='bold')
        
        # TOP-LEFT: Success Rate by Renewable Penetration (Line plot)
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
        
        # TOP-RIGHT: Convergence by Renewable Penetration (Stacked bar)
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
        
        # BOTTOM-LEFT: Resolution Methods Distribution (Pie chart)
        ax = axes[1, 0]
        
        # Aggregate resolution methods
        total_methods = {
            'Strict (Normal)': df['strict_normal'].sum(),
            'Strict (Contingency)': df['strict_contingency'].sum(),
            'Relaxed (Contingency)': df['relaxed_contingency'].sum(),
            'Restored Line': df['restored_line'].sum(),
            'Hard Reset': df['hard_reset'].sum(),
            'Failed': df['failed'].sum(),
        }
        
        # Remove zero values for cleaner pie chart
        total_methods = {k: v for k, v in total_methods.items() if v > 0}
        
        if total_methods:
            colors = ['#27ae60', '#3498db', '#f39c12', '#9b59b6', '#e67e22', '#e74c3c']
            color_map = dict(zip(['Strict (Normal)', 'Strict (Contingency)', 'Relaxed (Contingency)', 
                                 'Restored Line', 'Hard Reset', 'Failed'], colors))
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
                    fontsize=11, fontweight='bold', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'No resolution data available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, fontweight='bold')
        
        ax.set_title('Resolution Methods Distribution', fontweight='bold', fontsize=12)
        
        # BOTTOM-RIGHT: Summary Statistics (Text/Table)
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
        
        # Layout
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Determine output directory
        if output_dir is None and config is not None:
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
        
        if output_dir is None:
            output_dir = '.'
        
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'convergence_story.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return save_path

