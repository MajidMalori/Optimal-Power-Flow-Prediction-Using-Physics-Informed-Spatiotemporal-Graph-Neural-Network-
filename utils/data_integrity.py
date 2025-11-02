# File: utils/data_integrity.py
"""
Comprehensive data integrity and quality analysis module.

This module performs exhaustive analysis of generated power system data including:
- Convergence analysis and resolution methods
- Feature/target statistics (mean, median, std over time)
- Power balance validation (generation = load + losses)
- Precision and numerical quality checks
- Outlier detection and data quality metrics

All results are saved to experimental_results/data_integrity/ with plots and reports.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class DataIntegrityAnalyzer:
    """Comprehensive data integrity and quality analysis."""
    
    def __init__(self, data_dir: str, output_dir: str = None):
        """
        Initialize analyzer.
        
        Args:
            data_dir: Path to data directory (e.g., data/time_series/test)
            output_dir: Path to output directory (defaults to experimental_results/data_integrity)
        """
        self.data_dir = Path(data_dir)
        
        if output_dir is None:
            # Default: experimental_results/data_integrity/
            self.output_dir = Path("experimental_results/data_integrity")
        else:
            self.output_dir = Path(output_dir)
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "convergence").mkdir(exist_ok=True)
        (self.output_dir / "statistics").mkdir(exist_ok=True)
        (self.output_dir / "power_balance").mkdir(exist_ok=True)
        (self.output_dir / "quality").mkdir(exist_ok=True)
        
        # Feature names (10 features per bus)
        self.feature_names = [
            'Voltage Magnitude (pu)',
            'Voltage Angle (rad)',
            'Load P (MW)',
            'Load Q (Mvar)',
            'External Grid P (MW)',
            'External Grid Q (Mvar)',
            'Conventional Gen P (MW)',
            'Conventional Gen Q (Mvar)',
            'Renewable Gen P (MW)',
            'Renewable Gen Q (Mvar)'
        ]
        
        print(f"\n{'='*80}")
        print(f"DATA INTEGRITY ANALYZER")
        print(f"{'='*80}")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*80}\n")
    
    def analyze_all(self, cases: List[str] = None):
        """
        Run complete data integrity analysis for all cases and renewable fractions.
        
        Args:
            cases: List of case names (e.g., ['case57', 'case118']). If None, auto-detect.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Auto-detect cases if not provided
        if cases is None:
            cases = self._detect_cases()
        
        print(f"Analyzing cases: {cases}")
        
        # Display convergence analysis table first
        self._display_convergence_table(cases)
        
        # Initialize summary report
        summary_report = {
            'timestamp': timestamp,
            'cases_analyzed': {},
            'overall_statistics': {}
        }
        
        for case in cases:
            case_report = self._analyze_case(case)
            summary_report['cases_analyzed'][case] = case_report
        
        # Save summary report
        self._save_summary_report(summary_report, timestamp)
        
        print(f"\n{'='*80}")
        print(f"DATA INTEGRITY ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Reports saved to: {self.output_dir}")
        print(f"{'='*80}\n")
    
    def _detect_cases(self) -> List[str]:
        """Auto-detect available case names from data directory."""
        cases = set()
        for file in self.data_dir.glob("*.npy"):
            # Extract case name from filename (e.g., case57_features_frac0.0_*.npy)
            parts = file.stem.split('_')
            if parts[0].startswith('case'):
                cases.add(parts[0])
        return sorted(list(cases))
    
    def _display_convergence_table(self, cases: List[str]):
        """Display convergence summary table for all cases and renewable fractions."""
        import json
        
        renewable_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        print("\n" + "="*80)
        print("CONVERGENCE ANALYSIS")
        print("="*80)
        
        # Collect all data first
        all_data = {}  # {case: {frac: stats}}
        total_successful = 0
        total_failed = 0
        
        for case in cases:
            all_data[case] = {}
            
            for frac in renewable_fractions:
                # Find convergence report
                pattern = f"{case}_convergence_report_frac{frac:.1f}_*.json"
                report_files = list(self.data_dir.glob(pattern))
                
                if report_files:
                    # Use most recent report
                    report_file = sorted(report_files)[-1]
                    try:
                        with open(report_file, 'r') as f:
                            stats = json.load(f)
                        all_data[case][frac] = stats
                        total_successful += stats.get('successful', 0)
                        total_failed += stats.get('failed', 0)
                    except:
                        pass
        
        # Print horizontal table header with fixed column width
        col_width = 17  # Fixed width for each case column
        header = "Renewable%  "
        for case in cases:
            case_num = case.replace('case', '')
            header += f"| CASE{case_num:<{col_width-6}} "
        print(f"\n{header}")
        print("-" * len(header))
        
        # Print each renewable fraction row
        for frac in renewable_fractions:
            row = f"  {frac*100:>5.1f}%    "
            for case in cases:
                if frac in all_data[case]:
                    stats = all_data[case][frac]
                    rate = stats.get('success_rate', 0)
                    
                    # Show ✓ for 100%, ⚠ for <100%
                    icon = "✓" if rate >= 99.9 else "⚠"
                    cell = f"{icon} {rate:5.1f}%"
                    
                    # Pad to fixed width
                    row += f"| {cell:<{col_width-3}} "
                else:
                    row += f"| {'---':<{col_width-3}} "
            print(row)
        
        # Calculate summary statistics
        total_timesteps = total_successful + total_failed
        overall_rate = (total_successful / total_timesteps * 100) if total_timesteps > 0 else 0
        
        print(f"\nOverall convergence: {overall_rate:.1f}% ({total_successful}/{total_timesteps} timesteps)")
        print("="*80)
    
    def _analyze_case(self, case: str) -> Dict:
        """Analyze all renewable fractions for a single case."""
        # Detect available renewable fractions
        fractions = self._detect_fractions(case)
        
        print(f"\nAnalyzing {case.upper()}: {len(fractions)} renewable fractions")
        
        case_report = {
            'renewable_fractions': {},
            'convergence_summary': {},
            'data_quality_summary': {}
        }
        
        for frac in fractions:
            frac_report = self._analyze_fraction(case, frac)
            case_report['renewable_fractions'][f"{frac:.1f}"] = frac_report
        
        # Generate cross-fraction comparison plots
        self._plot_cross_fraction_analysis(case, case_report)
        
        return case_report
    
    def _detect_fractions(self, case: str) -> List[float]:
        """Detect available renewable fractions for a case."""
        fractions = set()
        for file in self.data_dir.glob(f"{case}_features_frac*.npy"):
            # Extract fraction from filename
            parts = file.stem.split('_frac')
            if len(parts) == 2:
                frac_str = parts[1].split('_')[0]  # Get fraction part before timestamp
                try:
                    fractions.add(float(frac_str))
                except ValueError:
                    continue
        return sorted(list(fractions))
    
    def _analyze_fraction(self, case: str, frac: float) -> Dict:
        """Analyze data for a single case and renewable fraction."""
        # Load data
        data = self._load_data(case, frac)
        
        if data is None:
            print(f"  ⚠ Could not load data for {case} @ {frac*100:.0f}%")
            return {'error': 'Data not found'}
        
        # Perform all analyses and generate plots silently
        convergence_report = self._analyze_convergence(case, frac, data)
        stats_report = self._analyze_statistics(case, frac, data)
        power_balance_report = self._analyze_power_balance(case, frac, data)
        quality_report = self._analyze_quality(case, frac, data)
        self._generate_plots(case, frac, data, stats_report)
        
        # Single summary line
        print(f"  ✓ {case} @ {frac*100:.0f}% renewable - Analysis complete")
        
        return {
            'convergence': convergence_report,
            'statistics': stats_report,
            'power_balance': power_balance_report,
            'quality': quality_report
        }
    
    def _load_data(self, case: str, frac: float) -> Optional[Dict]:
        """Load all data files for a case and renewable fraction."""
        # Find all feature files for this case and fraction
        feature_files = list(self.data_dir.glob(f"{case}_features_frac{frac:.1f}_*.npy"))
        
        if not feature_files:
            return None
        
        # Extract all timestamps and use the most recent one
        # Filename format: case57_features_frac0.0_20251101_190857.npy
        # Timestamp is the last TWO parts: YYYYMMDD_HHMMSS
        timestamps = []
        for f in feature_files:
            parts = f.stem.split('_')
            # Last two parts form the timestamp
            if len(parts) >= 2:
                ts = f"{parts[-2]}_{parts[-1]}"
                timestamps.append(ts)
        
        # Use most recent timestamp (lexicographically sorted)
        timestamp = sorted(timestamps)[-1] if timestamps else None
        
        if timestamp is None:
            return None
        
        data = {}
        
        # Load features and targets
        try:
            data['features'] = np.load(self.data_dir / f"{case}_features_frac{frac:.1f}_{timestamp}.npy", allow_pickle=True)
            data['targets'] = np.load(self.data_dir / f"{case}_targets_frac{frac:.1f}_{timestamp}.npy", allow_pickle=True)
            data['adjacency'] = np.load(self.data_dir / f"{case}_adjacency_frac{frac:.1f}_{timestamp}.npy", allow_pickle=True)
            
            # Load convergence report
            conv_file = self.data_dir / f"{case}_convergence_report_frac{frac:.1f}_{timestamp}.json"
            if conv_file.exists():
                with open(conv_file, 'r') as f:
                    data['convergence'] = json.load(f)
            
            # Load Ybus data
            data['ybus_base'] = np.load(self.data_dir / f"{case}_ybus_base_frac{frac:.1f}_{timestamp}.npy", allow_pickle=False)
            data['ybus_contingency_timesteps'] = np.load(self.data_dir / f"{case}_ybus_contingency_timesteps_frac{frac:.1f}_{timestamp}.npy", allow_pickle=False)
            
            # Load coefficients
            data['energy_coeffs'] = np.loadtxt(self.data_dir / f"{case}_time_energy_coeffs_frac{frac:.1f}_{timestamp}.txt")
            data['carbon_coeffs'] = np.loadtxt(self.data_dir / f"{case}_time_carbon_coeffs_frac{frac:.1f}_{timestamp}.txt")
            
            return data
            
        except Exception as e:
            print(f"  Error loading data: {e}")
            return None
    
    def _analyze_convergence(self, case: str, frac: float, data: Dict) -> Dict:
        """Analyze convergence statistics and resolution methods."""
        if 'convergence' not in data:
            return {
                'error': 'No convergence data available',
                'total_timesteps': 0,
                'successful': 0,
                'failed': 0,
                'success_rate': 0,
                'resolution_methods': {},
                'contingency_stats': {
                    'attempted': 0,
                    'successful': 0,
                    'failed': 0,
                    'resolved_strict': 0,
                    'resolved_relaxed': 0,
                    'restored_line': 0,
                },
                'critical_lines': {}
            }
        
        conv = data['convergence']
        
        report = {
            'total_timesteps': conv.get('total_timesteps', 0),
            'successful': conv.get('successful', 0),
            'failed': conv.get('failed', 0),
            'success_rate': conv.get('success_rate', 0),
            'resolution_methods': conv.get('resolution_methods', {}),
            'contingency_stats': {
                'attempted': conv.get('contingencies_attempted', 0),
                'successful': conv.get('contingencies_successful', 0),
                'failed': conv.get('contingencies_failed', 0),
                'resolved_strict': conv.get('contingencies_resolved_strict', 0),
                'resolved_relaxed': conv.get('contingencies_resolved_relaxed', 0),
                'restored_line': conv.get('contingencies_restored', 0),
            },
            'critical_lines': conv.get('critical_lines', {})
        }
        
        # Generate convergence plots (only if data is valid)
        try:
            self._plot_convergence_analysis(case, frac, conv, report)
        except Exception as e:
            print(f"    Warning: Could not generate convergence plot: {e}")
        
        return report
    
    def _analyze_statistics(self, case: str, frac: float, data: Dict) -> Dict:
        """Compute comprehensive statistics for all features and targets."""
        features = data['features']
        targets = data['targets']
        
        # Shape: (timesteps, num_buses, 10)
        num_timesteps, num_buses, num_features = features.shape
        
        report = {
            'shape': {
                'timesteps': num_timesteps,
                'buses': num_buses,
                'features': num_features
            },
            'features': {},
            'targets': {}
        }
        
        # Compute statistics for each feature
        for i, name in enumerate(self.feature_names):
            # Features across all timesteps and buses
            feat_data = features[:, :, i].flatten()
            targ_data = targets[:, :, i].flatten()
            
            # Remove zeros (might be from buses without that feature)
            feat_nonzero = feat_data[feat_data != 0]
            targ_nonzero = targ_data[targ_data != 0]
            
            report['features'][name] = {
                'mean': float(np.mean(feat_nonzero)) if len(feat_nonzero) > 0 else 0.0,
                'median': float(np.median(feat_nonzero)) if len(feat_nonzero) > 0 else 0.0,
                'std': float(np.std(feat_nonzero)) if len(feat_nonzero) > 0 else 0.0,
                'min': float(np.min(feat_nonzero)) if len(feat_nonzero) > 0 else 0.0,
                'max': float(np.max(feat_nonzero)) if len(feat_nonzero) > 0 else 0.0,
                'nonzero_count': int(len(feat_nonzero)),
                'zero_count': int(len(feat_data) - len(feat_nonzero))
            }
            
            report['targets'][name] = {
                'mean': float(np.mean(targ_nonzero)) if len(targ_nonzero) > 0 else 0.0,
                'median': float(np.median(targ_nonzero)) if len(targ_nonzero) > 0 else 0.0,
                'std': float(np.std(targ_nonzero)) if len(targ_nonzero) > 0 else 0.0,
                'min': float(np.min(targ_nonzero)) if len(targ_nonzero) > 0 else 0.0,
                'max': float(np.max(targ_nonzero)) if len(targ_nonzero) > 0 else 0.0,
                'nonzero_count': int(len(targ_nonzero)),
                'zero_count': int(len(targ_data) - len(targ_nonzero))
            }
        
        return report
    
    def _analyze_power_balance(self, case: str, frac: float, data: Dict) -> Dict:
        """Validate power balance: generation = load + losses."""
        targets = data['targets']
        
        # Extract power components (targets are ground truth)
        p_load = targets[:, :, 2]  # Load P
        p_ext_grid = targets[:, :, 4]  # External grid P
        p_conv_gen = targets[:, :, 6]  # Conventional gen P
        p_ren_gen = targets[:, :, 8]  # Renewable gen P
        
        # Total generation and load per timestep
        total_gen = np.sum(p_ext_grid + p_conv_gen + p_ren_gen, axis=1)
        total_load = np.sum(p_load, axis=1)
        
        # Power imbalance (should be small, representing losses)
        imbalance = total_gen - total_load
        imbalance_percent = (imbalance / total_load) * 100
        
        report = {
            'mean_generation_mw': float(np.mean(total_gen)),
            'mean_load_mw': float(np.mean(total_load)),
            'mean_imbalance_mw': float(np.mean(imbalance)),
            'mean_imbalance_percent': float(np.mean(imbalance_percent)),
            'max_imbalance_mw': float(np.max(np.abs(imbalance))),
            'max_imbalance_percent': float(np.max(np.abs(imbalance_percent))),
            'power_balance_ok': bool(np.max(np.abs(imbalance_percent)) < 10),  # Less than 10% imbalance
        }
        
        # Plot power balance
        self._plot_power_balance(case, frac, total_gen, total_load, imbalance)
        
        return report
    
    def _analyze_quality(self, case: str, frac: float, data: Dict) -> Dict:
        """Check data quality: precision, outliers, NaN/Inf, etc."""
        features = data['features']
        targets = data['targets']
        
        report = {
            'features': self._check_array_quality(features, 'Features'),
            'targets': self._check_array_quality(targets, 'Targets'),
            'measurement_noise': self._analyze_measurement_noise(features, targets)
        }
        
        return report
    
    def _check_array_quality(self, arr: np.ndarray, name: str) -> Dict:
        """Check array for NaN, Inf, outliers, precision issues."""
        report = {
            'has_nan': bool(np.isnan(arr).any()),
            'has_inf': bool(np.isinf(arr).any()),
            'nan_count': int(np.isnan(arr).sum()),
            'inf_count': int(np.isinf(arr).sum()),
            'dtype': str(arr.dtype),
            'memory_mb': float(arr.nbytes / 1024 / 1024),
        }
        
        # Check for precision issues (values too small/large)
        finite_vals = arr[np.isfinite(arr)]
        if len(finite_vals) > 0:
            report['min_value'] = float(np.min(finite_vals))
            report['max_value'] = float(np.max(finite_vals))
            report['has_underflow'] = bool(np.any((finite_vals != 0) & (np.abs(finite_vals) < 1e-10)))
            report['has_overflow'] = bool(np.any(np.abs(finite_vals) > 1e10))
        
        return report
    
    def _analyze_measurement_noise(self, features: np.ndarray, targets: np.ndarray) -> Dict:
        """Analyze measurement noise (difference between features and targets)."""
        # Measurement error = |features - targets| / targets
        nonzero_targets = targets != 0
        relative_error = np.abs(features - targets) / (np.abs(targets) + 1e-10)
        relative_error = relative_error[nonzero_targets]
        
        report = {
            'mean_relative_error': float(np.mean(relative_error)),
            'median_relative_error': float(np.median(relative_error)),
            'max_relative_error': float(np.max(relative_error)),
            'std_relative_error': float(np.std(relative_error)),
        }
        
        return report
    
    def _plot_convergence_analysis(self, case: str, frac: float, conv: Dict, report: Dict):
        """Generate comprehensive convergence analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Convergence Analysis: {case.upper()} @ {frac*100:.0f}% Renewables', 
                     fontsize=16, fontweight='bold')
        
        # 1. Resolution methods pie chart
        ax = axes[0, 0]
        methods = report['resolution_methods']
        if methods:
            labels = []
            sizes = []
            colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
            
            if methods.get('strict_normal', 0) > 0:
                labels.append(f"Strict (Normal)\n{methods['strict_normal']}")
                sizes.append(methods['strict_normal'])
            if methods.get('strict_contingency', 0) > 0:
                labels.append(f"Strict (Contingency)\n{methods['strict_contingency']}")
                sizes.append(methods['strict_contingency'])
            if methods.get('relaxed_contingency', 0) > 0:
                labels.append(f"Relaxed (Contingency)\n{methods['relaxed_contingency']}")
                sizes.append(methods['relaxed_contingency'])
            if methods.get('restored_line', 0) > 0:
                labels.append(f"Restored Line\n{methods['restored_line']}")
                sizes.append(methods['restored_line'])
            
            ax.pie(sizes, labels=labels, colors=colors[:len(sizes)], autopct='%1.1f%%', startangle=90)
            ax.set_title('Resolution Methods Distribution')
        else:
            ax.text(0.5, 0.5, 'No resolution data', ha='center', va='center')
            ax.set_title('Resolution Methods Distribution')
        
        # 2. Contingency handling success
        ax = axes[0, 1]
        cont_stats = report['contingency_stats']
        
        # Calculate successful (should match strict + relaxed + restored)
        calculated_successful = cont_stats['resolved_strict'] + cont_stats['resolved_relaxed'] + cont_stats['restored_line']
        
        categories = ['Attempted', 'Successful', 'Failed', 'Strict', 'Relaxed', 'Restored']
        values = [
            cont_stats['attempted'],
            calculated_successful,  # Use calculated value
            cont_stats['failed'],
            cont_stats['resolved_strict'],
            cont_stats['resolved_relaxed'],
            cont_stats['restored_line']
        ]
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#27ae60', '#f39c12', '#e67e22']
        bars = ax.bar(categories, values, color=colors)
        ax.set_ylabel('Count')
        ax.set_title('Contingency Handling Statistics')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only show label if non-zero
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
        
        # Add accounting check
        total_check = calculated_successful + cont_stats['failed']
        if cont_stats['attempted'] > 0 and total_check != cont_stats['attempted']:
            ax.text(0.5, 0.95, f"⚠ Accounting error: {calculated_successful}+{cont_stats['failed']}≠{cont_stats['attempted']}", 
                   transform=ax.transAxes, ha='center', va='top', 
                   fontsize=8, color='red', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # 3. Critical lines (if any)
        ax = axes[1, 0]
        critical_lines = report['critical_lines']
        if critical_lines:
            # Sort by failure count and get top 10
            sorted_lines = sorted(critical_lines.items(), 
                                 key=lambda x: x[1]['failure_count'], 
                                 reverse=True)
            
            top_n = min(10, len(sorted_lines))  # Show max 10 lines
            display_lines = sorted_lines[:top_n]
            remaining = len(sorted_lines) - top_n
            
            line_ids = []
            failure_counts = []
            for line_key, line_data in display_lines:
                line_id = line_data['line_id']
                relaxed = line_data['resolution_methods'].get('relaxed', 0)
                restored = line_data['resolution_methods'].get('restored', 0)
                
                # Show how it was resolved
                if restored > relaxed:
                    label = f"Line {line_id} (restored {restored}×)"
                elif relaxed > 0:
                    label = f"Line {line_id} (relaxed {relaxed}×)"
                else:
                    label = f"Line {line_id}"
                
                line_ids.append(label)
                failure_counts.append(line_data['failure_count'])
            
            bars = ax.barh(line_ids, failure_counts, color='#e74c3c')
            ax.set_xlabel('Times This Line Failed Contingency')
            
            title = f'Critical Lines (Top {top_n}'
            if remaining > 0:
                title += f' of {len(sorted_lines)}'
            title += ')'
            ax.set_title(title)
            ax.grid(axis='x', alpha=0.3)
            
            # Add explanation text
            if top_n > 0:
                note = "\nThese lines cause convergence\nfailure when removed (N-1)."
                ax.text(0.98, 0.02, note, transform=ax.transAxes, 
                       fontsize=9, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        else:
            ax.text(0.5, 0.5, 'No critical lines identified\n(all contingencies handled easily)', 
                   ha='center', va='center', fontsize=11)
            ax.set_title('Critical Lines')
        
        # 4. Success rate summary
        ax = axes[1, 1]
        ax.axis('off')
        
        # Calculate actual success rate (successful / total)
        actual_success_rate = (report['successful'] / report['total_timesteps'] * 100) if report['total_timesteps'] > 0 else 0
        
        summary_text = f"""
        CONVERGENCE SUMMARY
        {'='*40}
        
        Total Timesteps: {report['total_timesteps']:,}
        Successful: {report['successful']:,}
        Actually Failed: {report['failed']:,}
        Success Rate: {actual_success_rate:.2f}%
        
        RESOLUTION BREAKDOWN
        {'='*40}
        
        Normal Operation: {methods.get('strict_normal', 0):,}
        Contingency (Strict): {methods.get('strict_contingency', 0):,}
        Contingency (Relaxed): {methods.get('relaxed_contingency', 0):,}
        Line Restored: {methods.get('restored_line', 0):,}
        
        CONTINGENCY HANDLING
        {'='*40}
        
        Contingencies Attempted: {cont_stats['attempted']}
        
        Successfully Handled: {calculated_successful}
        - Handled with Strict: {cont_stats['resolved_strict']}
        - Needed Relaxed: {cont_stats['resolved_relaxed']}
        - Line Restored: {cont_stats['restored_line']}
        
        Failed Completely: {cont_stats['failed']}
        (Failed even after restoring line)
        
        Success Rate: {(calculated_successful / cont_stats['attempted'] * 100) if cont_stats['attempted'] > 0 else 100:.1f}%
        
        Note: Successful + Failed should equal
        Attempted. If not, there's a bug!
        """
        
        ax.text(0.1, 0.9, summary_text, fontsize=11, family='monospace', 
               verticalalignment='top', transform=ax.transAxes)
        
        plt.tight_layout()
        output_file = self.output_dir / "convergence" / f"{case}_frac{frac:.1f}_convergence.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_plots(self, case: str, frac: float, data: Dict, stats_report: Dict):
        """Generate statistical distribution and time-series plots."""
        features = data['features']
        targets = data['targets']
        num_timesteps = features.shape[0]
        
        # Plot 1: Feature statistics over time (6 key features)
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Feature Statistics Over Time: {case.upper()} @ {frac*100:.0f}% Renewables', 
                     fontsize=16, fontweight='bold')
        
        key_features = [0, 2, 4, 6, 8, 3]  # Voltage, Load P, Ext Grid P, Conv Gen P, Ren Gen P, Load Q
        
        for idx, feat_idx in enumerate(key_features):
            ax = axes[idx // 2, idx % 2]
            
            # Compute mean across buses for each timestep
            feat_mean = np.mean(features[:, :, feat_idx], axis=1)
            targ_mean = np.mean(targets[:, :, feat_idx], axis=1)
            
            timesteps = np.arange(num_timesteps)
            
            ax.plot(timesteps, targ_mean, label='Ground Truth', color='#2ecc71', alpha=0.8, linewidth=1.5)
            ax.plot(timesteps, feat_mean, label='Measurements', color='#3498db', alpha=0.6, linewidth=1.0)
            ax.fill_between(timesteps, targ_mean, feat_mean, alpha=0.2)
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel(self.feature_names[feat_idx])
            ax.set_title(self.feature_names[feat_idx])
            ax.legend(loc='best')
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "statistics" / f"{case}_frac{frac:.1f}_features_timeseries.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Distribution histograms
        self._plot_feature_distributions(case, frac, features, targets)
    
    def _plot_feature_distributions(self, case: str, frac: float, features: np.ndarray, targets: np.ndarray):
        """Plot distribution histograms for all features."""
        fig, axes = plt.subplots(5, 2, figsize=(15, 20))
        fig.suptitle(f'Feature Distributions: {case.upper()} @ {frac*100:.0f}% Renewables', 
                     fontsize=16, fontweight='bold')
        
        for i in range(10):
            ax = axes[i // 2, i % 2]
            
            # Flatten and remove zeros
            feat_data = features[:, :, i].flatten()
            targ_data = targets[:, :, i].flatten()
            
            feat_nonzero = feat_data[feat_data != 0]
            targ_nonzero = targ_data[targ_data != 0]
            
            if len(targ_nonzero) > 0:
                ax.hist(targ_nonzero, bins=50, alpha=0.6, label='Ground Truth', color='#2ecc71', density=True)
            if len(feat_nonzero) > 0:
                ax.hist(feat_nonzero, bins=50, alpha=0.4, label='Measurements', color='#3498db', density=True)
            
            ax.set_xlabel(self.feature_names[i])
            ax.set_ylabel('Density')
            ax.set_title(self.feature_names[i])
            ax.legend(loc='best')
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "statistics" / f"{case}_frac{frac:.1f}_distributions.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_power_balance(self, case: str, frac: float, total_gen: np.ndarray, 
                           total_load: np.ndarray, imbalance: np.ndarray):
        """Plot power balance validation."""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'Power Balance: {case.upper()} @ {frac*100:.0f}% Renewables', 
                     fontsize=16, fontweight='bold')
        
        timesteps = np.arange(len(total_gen))
        
        # Plot 1: Generation vs Load
        ax = axes[0]
        ax.plot(timesteps, total_gen, label='Total Generation', color='#e74c3c', linewidth=1.5)
        ax.plot(timesteps, total_load, label='Total Load', color='#3498db', linewidth=1.5)
        ax.fill_between(timesteps, total_gen, total_load, alpha=0.2, color='#95a5a6')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Power (MW)')
        ax.set_title('Generation vs. Load')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        # Plot 2: Power Imbalance
        ax = axes[1]
        imbalance_percent = (imbalance / total_load) * 100
        ax.plot(timesteps, imbalance_percent, color='#f39c12', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axhline(y=5, color='r', linestyle='--', alpha=0.3, label='±5% threshold')
        ax.axhline(y=-5, color='r', linestyle='--', alpha=0.3)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Imbalance (%)')
        ax.set_title('Power Imbalance (Generation - Load) %')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "power_balance" / f"{case}_frac{frac:.1f}_power_balance.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_cross_fraction_analysis(self, case: str, case_report: Dict):
        """Generate cross-fraction comparison plots."""
        all_fractions = sorted([float(f) for f in case_report['renewable_fractions'].keys()])
        
        # Extract metrics across fractions (only for successfully loaded data)
        fractions = []
        success_rates = []
        mean_generation = []
        mean_load = []
        mean_renewable_gen = []
        
        for frac in all_fractions:
            frac_str = f"{frac:.1f}"
            frac_data = case_report['renewable_fractions'][frac_str]
            
            if 'error' not in frac_data:
                fractions.append(frac)
                success_rates.append(frac_data['convergence']['success_rate'])
                mean_generation.append(frac_data['power_balance']['mean_generation_mw'])
                mean_load.append(frac_data['power_balance']['mean_load_mw'])
                mean_renewable_gen.append(frac_data['statistics']['targets']['Renewable Gen P (MW)']['mean'])
        
        # If no data available, skip plotting
        if len(fractions) == 0:
            print(f"  Warning: No valid data for cross-fraction analysis of {case}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Cross-Fraction Analysis: {case.upper()}', fontsize=16, fontweight='bold')
        
        # Plot 1: Success rate vs renewable fraction
        ax = axes[0, 0]
        ax.plot(fractions, success_rates, marker='o', linewidth=2, markersize=8, color='#2ecc71')
        ax.set_xlabel('Renewable Fraction')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Convergence Success Rate')
        ax.grid(alpha=0.3)
        ax.set_ylim([90, 105])
        
        # Plot 2: Generation vs Load
        ax = axes[0, 1]
        ax.plot(fractions, mean_generation, marker='o', label='Generation', linewidth=2, markersize=8, color='#e74c3c')
        ax.plot(fractions, mean_load, marker='s', label='Load', linewidth=2, markersize=8, color='#3498db')
        ax.set_xlabel('Renewable Fraction')
        ax.set_ylabel('Power (MW)')
        ax.set_title('Mean Generation vs. Load')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        # Plot 3: Renewable generation scaling
        ax = axes[1, 0]
        ax.plot(fractions, mean_renewable_gen, marker='o', linewidth=2, markersize=8, color='#27ae60')
        ax.set_xlabel('Renewable Fraction')
        ax.set_ylabel('Mean Renewable Generation (MW)')
        ax.set_title('Renewable Generation Scaling')
        ax.grid(alpha=0.3)
        
        # Plot 4: Summary table
        ax = axes[1, 1]
        ax.axis('off')
        
        table_data = []
        for i, frac in enumerate(fractions):
            table_data.append([
                f"{frac*100:.0f}%",
                f"{success_rates[i]:.2f}%",
                f"{mean_renewable_gen[i]:.1f}",
                f"{mean_generation[i]:.1f}"
            ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Renewable\nFraction', 'Success\nRate', 'Renewable\nGen (MW)', 'Total\nGen (MW)'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        output_file = self.output_dir / f"{case}_cross_fraction_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_summary_report(self, summary_report: Dict, timestamp: str):
        """Save comprehensive summary report as JSON."""
        output_file = self.output_dir / f"data_integrity_report_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        # Also save a human-readable text report
        self._save_text_report(summary_report, timestamp)
    
    def _save_text_report(self, summary_report: Dict, timestamp: str):
        """Save human-readable text report."""
        output_file = self.output_dir / f"data_integrity_report_{timestamp}.txt"
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DATA INTEGRITY ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {summary_report['timestamp']}\n")
            f.write("="*80 + "\n\n")
            
            for case, case_data in summary_report['cases_analyzed'].items():
                f.write(f"\n{'='*80}\n")
                f.write(f"CASE: {case.upper()}\n")
                f.write(f"{'='*80}\n\n")
                
                for frac_str, frac_data in case_data['renewable_fractions'].items():
                    f.write(f"\n--- Renewable Fraction: {float(frac_str)*100:.0f}% ---\n\n")
                    
                    if 'error' in frac_data:
                        f.write(f"ERROR: {frac_data['error']}\n")
                        continue
                    
                    # Convergence
                    conv = frac_data['convergence']
                    f.write(f"CONVERGENCE:\n")
                    f.write(f"  Success Rate: {conv['success_rate']:.2f}%\n")
                    f.write(f"  Successful: {conv['successful']} / {conv['total_timesteps']}\n")
                    f.write(f"  Failed: {conv['failed']}\n\n")
                    
                    # Power Balance
                    power = frac_data['power_balance']
                    f.write(f"POWER BALANCE:\n")
                    f.write(f"  Mean Generation: {power['mean_generation_mw']:.2f} MW\n")
                    f.write(f"  Mean Load: {power['mean_load_mw']:.2f} MW\n")
                    f.write(f"  Mean Imbalance: {power['mean_imbalance_percent']:.3f}%\n")
                    f.write(f"  Balance OK: {power['power_balance_ok']}\n\n")
                    
                    # Quality
                    quality = frac_data['quality']
                    f.write(f"DATA QUALITY:\n")
                    f.write(f"  Has NaN: {quality['features']['has_nan'] or quality['targets']['has_nan']}\n")
                    f.write(f"  Has Inf: {quality['features']['has_inf'] or quality['targets']['has_inf']}\n")
                    f.write(f"  Mean Measurement Error: {quality['measurement_noise']['mean_relative_error']:.4f}\n\n")


def analyze_data_integrity(data_dir: str, output_dir: str = None, cases: List[str] = None):
    """
    Convenience function to run data integrity analysis.
    
    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory (defaults to experimental_results/data_integrity)
        cases: List of case names to analyze (auto-detect if None)
    """
    analyzer = DataIntegrityAnalyzer(data_dir, output_dir)
    analyzer.analyze_all(cases)


if __name__ == "__main__":
    import sys
    
    # Usage: python -m utils.data_integrity [data_dir] [output_dir]
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/time_series/test"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyze_data_integrity(data_dir, output_dir)

