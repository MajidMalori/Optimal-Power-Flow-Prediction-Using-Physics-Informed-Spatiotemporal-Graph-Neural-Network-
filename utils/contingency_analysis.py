"""
Targeted Contingency Analysis Module

Provides tools for identifying critical lines and systematically testing
specific N-1 contingencies to evaluate model robustness and system security.

Contingency analysis is crucial for power system security assessment:
- Identifies critical transmission lines
- Tests system response to line outages
- Evaluates model performance under stressed conditions
- Supports N-1 security criterion (system must remain stable after any single line outage)
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
from collections import defaultdict

try:
    import pandapower as pp
    import pandapower.topology as top
    PANDAPOWER_AVAILABLE = True
except ImportError:
    PANDAPOWER_AVAILABLE = False


class ContingencyAnalyzer:
    """
    Analyzes power system contingencies and identifies critical lines.
    
    Provides methods for:
    1. Identifying critical lines (by power flow, centrality, or custom criteria)
    2. Testing specific N-1 contingencies
    3. Evaluating model performance under contingencies
    4. Ranking contingencies by severity
    """
    
    def __init__(self, net: Optional[Any] = None):
        """
        Initialize the contingency analyzer.
        
        Args:
            net: Optional pandapower network (for topology analysis)
        """
        self.net = net
        self.critical_lines = []
        self.contingency_results = {}
    
    def identify_critical_lines_by_power_flow(self, net: Any, 
                                             power_flow_results: Optional[pd.DataFrame] = None,
                                             top_k: int = 10) -> List[int]:
        """
        Identifies critical lines based on power flow magnitude.
        
        Lines with high power flow are more critical because their outage
        causes larger power redistribution and potential violations.
        
        Args:
            net: pandapower network
            power_flow_results: Optional power flow results (if None, runs power flow)
            top_k: Number of top critical lines to return
            
        Returns:
            List of line indices sorted by criticality (most critical first)
        """
        if not PANDAPOWER_AVAILABLE:
            raise ImportError("pandapower is required for contingency analysis")
        
        if power_flow_results is None:
            try:
                pp.runpp(net, numba=False)
            except:
                return []
        
        # Get power flow through each line
        line_flows = []
        for idx, line in net.line.iterrows():
            if line.in_service:
                # Get power flow magnitude (MVA)
                p_from = net.res_line.loc[idx, 'p_from_mw'] if idx in net.res_line.index else 0.0
                q_from = net.res_line.loc[idx, 'q_from_mvar'] if idx in net.res_line.index else 0.0
                s_mva = np.sqrt(p_from**2 + q_from**2)
                
                line_flows.append({
                    'line_idx': idx,
                    's_mva': s_mva,
                    'p_mw': p_from,
                    'q_mvar': q_from
                })
        
        # Sort by power flow magnitude (descending)
        line_flows.sort(key=lambda x: x['s_mva'], reverse=True)
        
        # Return top K line indices
        critical_lines = [item['line_idx'] for item in line_flows[:top_k]]
        self.critical_lines = critical_lines
        
        return critical_lines
    
    def identify_critical_lines_by_centrality(self, net: Any, 
                                             top_k: int = 10) -> List[int]:
        """
        Identifies critical lines based on graph centrality measures.
        
        Uses betweenness centrality to find lines that are on many shortest paths,
        making them critical for power flow routing.
        
        Args:
            net: pandapower network
            top_k: Number of top critical lines to return
            
        Returns:
            List of line indices sorted by centrality (most critical first)
        """
        if not PANDAPOWER_AVAILABLE:
            raise ImportError("pandapower is required for contingency analysis")
        
        # Create networkx graph from pandapower network
        graph = top.create_nxgraph(net, include_trafos=True)
        
        # Calculate edge betweenness centrality
        edge_centrality = nx.edge_betweenness_centrality(graph)
        
        # Map networkx edges to pandapower line indices
        line_centrality = []
        for idx, line in net.line.iterrows():
            if line.in_service:
                from_bus = int(line.from_bus)
                to_bus = int(line.to_bus)
                
                # Check both directions (graph may be undirected)
                edge_key = (from_bus, to_bus)
                if edge_key not in edge_centrality:
                    edge_key = (to_bus, from_bus)
                
                centrality = edge_centrality.get(edge_key, 0.0)
                line_centrality.append({
                    'line_idx': idx,
                    'centrality': centrality
                })
        
        # Sort by centrality (descending)
        line_centrality.sort(key=lambda x: x['centrality'], reverse=True)
        
        # Return top K line indices
        critical_lines = [item['line_idx'] for item in line_centrality[:top_k]]
        self.critical_lines = critical_lines
        
        return critical_lines
    
    def identify_critical_lines_by_historical_failures(self, 
                                                      failure_history: Dict[int, int],
                                                      top_k: int = 10) -> List[int]:
        """
        Identifies critical lines based on historical failure frequency.
        
        Lines that have caused more failures in the past are considered more critical.
        
        Args:
            failure_history: Dictionary mapping line_idx -> failure_count
            top_k: Number of top critical lines to return
            
        Returns:
            List of line indices sorted by failure frequency (most critical first)
        """
        # Sort lines by failure count
        sorted_lines = sorted(failure_history.items(), key=lambda x: x[1], reverse=True)
        
        # Return top K line indices
        critical_lines = [line_idx for line_idx, _ in sorted_lines[:top_k]]
        self.critical_lines = critical_lines
        
        return critical_lines
    
    def test_contingency(self, net: Any, line_idx: int, 
                        run_power_flow: bool = True) -> Dict[str, Any]:
        """
        Tests a specific N-1 contingency by removing a line.
        
        Args:
            net: pandapower network (will be modified)
            line_idx: Index of line to remove
            run_power_flow: Whether to run power flow after contingency
            
        Returns:
            Dictionary with contingency test results:
            - 'success': bool - whether power flow converged
            - 'islanding': bool - whether contingency caused islanding
            - 'voltage_violations': int - number of voltage violations
            - 'line_loading': dict - loading of remaining lines
            - 'power_loss': float - total power loss
        """
        if not PANDAPOWER_AVAILABLE:
            raise ImportError("pandapower is required for contingency analysis")
        
        results = {
            'line_idx': line_idx,
            'success': False,
            'islanding': False,
            'voltage_violations': 0,
            'line_loading': {},
            'power_loss': 0.0,
            'error': None
        }
        
        # Check if line exists and is in service
        if line_idx not in net.line.index:
            results['error'] = f"Line {line_idx} does not exist"
            return results
        
        if not net.line.loc[line_idx, 'in_service']:
            results['error'] = f"Line {line_idx} is already out of service"
            return results
        
        # Check for islanding before removing line
        graph_before = top.create_nxgraph(net, include_trafos=True)
        is_connected_before = nx.is_connected(graph_before)
        
        # Remove line
        net.line.loc[line_idx, 'in_service'] = False
        
        # Check for islanding after removing line
        graph_after = top.create_nxgraph(net, include_trafos=True)
        is_connected_after = nx.is_connected(graph_after)
        
        if not is_connected_after:
            results['islanding'] = True
            # Restore line
            net.line.loc[line_idx, 'in_service'] = True
            results['error'] = "Contingency causes islanding"
            return results
        
        # Run power flow if requested
        if run_power_flow:
            try:
                pp.runpp(net, numba=False)
                results['success'] = True
                
                # Check voltage violations
                v_min = 0.90  # per-unit
                v_max = 1.10  # per-unit
                vm_pu = net.res_bus.vm_pu.values
                violations = np.sum((vm_pu < v_min) | (vm_pu > v_max))
                results['voltage_violations'] = int(violations)
                
                # Get line loading
                for idx, line in net.line.iterrows():
                    if line.in_service and idx in net.res_line.index:
                        p_mw = net.res_line.loc[idx, 'p_from_mw']
                        q_mvar = net.res_line.loc[idx, 'q_from_mvar']
                        s_mva = np.sqrt(p_mw**2 + q_mvar**2)
                        max_s_mva = line.max_i_ka * net.bus.loc[line.from_bus, 'vn_kv'] * np.sqrt(3) / 1000.0
                        loading_pct = (s_mva / max_s_mva * 100.0) if max_s_mva > 0 else 0.0
                        results['line_loading'][idx] = loading_pct
                
                # Get power loss
                results['power_loss'] = float(net.res_line.pl_mw.sum())
                
            except Exception as e:
                results['error'] = str(e)
                results['success'] = False
        
        # Restore line
        net.line.loc[line_idx, 'in_service'] = True
        
        return results
    
    def evaluate_model_under_contingency(self, model: torch.nn.Module,
                                        features: torch.Tensor,
                                        adjacency: torch.Tensor,
                                        ybus_normal: torch.Tensor,
                                        ybus_contingency: torch.Tensor,
                                        bus_types: Optional[torch.Tensor] = None,
                                        device: torch.device = torch.device('cpu')) -> Dict[str, float]:
        """
        Evaluates model performance under a contingency scenario.
        
        Compares model predictions using normal Ybus vs contingency Ybus.
        
        Args:
            model: Trained model
            features: Input features [batch, buses, features] or [batch, seq_len, buses, features]
            adjacency: Adjacency matrix [buses, buses]
            ybus_normal: Normal Ybus matrix [buses, buses] or [batch, buses, buses]
            ybus_contingency: Contingency Ybus matrix [buses, buses] or [batch, buses, buses]
            bus_types: Optional bus type codes [batch, buses]
            device: Device to run evaluation on
            
        Returns:
            Dictionary with performance metrics comparing normal vs contingency
        """
        model.eval()
        
        with torch.no_grad():
            # Move to device
            features = features.to(device)
            adjacency = adjacency.to(device)
            ybus_normal = ybus_normal.to(device)
            ybus_contingency = ybus_contingency.to(device)
            if bus_types is not None:
                bus_types = bus_types.to(device)
            
            # Handle sequential models
            if features.dim() == 4:
                features_input = features[:, -1, :, :]
            else:
                features_input = features
            
            # Get predictions with normal Ybus
            try:
                outputs_normal = model(features_input, adjacency, bus_types=bus_types)
            except TypeError:
                outputs_normal = model(features_input, adjacency)
            
            # Get predictions with contingency Ybus
            # Note: Model doesn't directly use Ybus, but we can evaluate physics loss
            # For a more accurate evaluation, we'd need to modify the model to accept Ybus
            # For now, we assume the model uses the same adjacency and evaluate consistency
            
            # Calculate metrics (simplified - would need full evaluation pipeline)
            results = {
                'normal_prediction_norm': torch.norm(outputs_normal).item(),
                'contingency_available': True,
                'note': 'Full evaluation requires integration with PowerSystemLoss'
            }
        
        return results
    
    def rank_contingencies_by_severity(self, contingency_results: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
        """
        Ranks contingencies by severity score.
        
        Severity is calculated based on:
        - Power flow convergence failure
        - Voltage violations
        - Line loading violations
        - Power loss increase
        
        Args:
            contingency_results: List of contingency test results
            
        Returns:
            List of (line_idx, severity_score) tuples, sorted by severity (highest first)
        """
        severity_scores = []
        
        for result in contingency_results:
            line_idx = result['line_idx']
            score = 0.0
            
            # Convergence failure: high severity
            if not result['success']:
                score += 100.0
            
            # Islanding: very high severity
            if result.get('islanding', False):
                score += 200.0
            
            # Voltage violations: medium severity
            score += result.get('voltage_violations', 0) * 10.0
            
            # High line loading: medium severity
            line_loading = result.get('line_loading', {})
            for loading_pct in line_loading.values():
                if loading_pct > 100.0:  # Overloaded
                    score += (loading_pct - 100.0) * 2.0
            
            # Power loss increase: low severity
            # (Would need baseline for comparison)
            
            severity_scores.append((line_idx, score))
        
        # Sort by severity (descending)
        severity_scores.sort(key=lambda x: x[1], reverse=True)
        
        return severity_scores
    
    def generate_contingency_report(self, contingency_results: List[Dict[str, Any]]) -> str:
        """
        Generates a human-readable report of contingency analysis results.
        
        Args:
            contingency_results: List of contingency test results
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 80,
            "CONTINGENCY ANALYSIS REPORT",
            "=" * 80,
            ""
        ]
        
        # Summary statistics
        total = len(contingency_results)
        successful = sum(1 for r in contingency_results if r.get('success', False))
        failed = total - successful
        islanding = sum(1 for r in contingency_results if r.get('islanding', False))
        
        report_lines.extend([
            f"Total Contingencies Tested: {total}",
            f"Successful Power Flow: {successful} ({successful/total*100:.1f}%)",
            f"Failed Power Flow: {failed} ({failed/total*100:.1f}%)",
            f"Islanding Cases: {islanding} ({islanding/total*100:.1f}%)",
            ""
        ])
        
        # Rank by severity
        severity_ranking = self.rank_contingencies_by_severity(contingency_results)
        
        report_lines.extend([
            "Top 10 Most Critical Contingencies:",
            "-" * 80,
            f"{'Line ID':<10} {'Severity':<15} {'Status':<15} {'Voltage Violations':<20}",
            "-" * 80
        ])
        
        for line_idx, severity in severity_ranking[:10]:
            result = next(r for r in contingency_results if r['line_idx'] == line_idx)
            status = "Islanding" if result.get('islanding', False) else \
                     "Failed" if not result.get('success', False) else \
                     "Success"
            violations = result.get('voltage_violations', 0)
            
            report_lines.append(
                f"{line_idx:<10} {severity:<15.2f} {status:<15} {violations:<20}"
            )
        
        report_lines.extend([
            "",
            "=" * 80
        ])
        
        return "\n".join(report_lines)

