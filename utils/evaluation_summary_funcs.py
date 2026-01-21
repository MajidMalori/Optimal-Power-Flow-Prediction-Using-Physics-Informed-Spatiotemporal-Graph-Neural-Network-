from typing import List, Dict, Any
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

def print_model_summary(best_run, moopf_results, model_name, num_buses, is_physics_informed, final_test_score, final_metric_name):
    """Print summary for a single model."""
    print(f"\n{model_name} ({num_buses}-bus) Summary:")
    print(f"  Test Score ({final_metric_name}): {final_test_score:.4f}")
    # Also show validation MSE for context
    val_mse = best_run.get('training_mse', 'N/A')
    if val_mse != 'N/A':
        print(f"  Validation MSE: {val_mse:.6f}")
    
    # FIX: Always print MOOPF metrics if available (for both Physics and Non-Physics models)
    if moopf_results:
        print(f"  Power Loss: {moopf_results.get('power_loss', 'N/A')}")
        print(f"  Voltage Deviation: {moopf_results.get('voltage_deviation', 'N/A')}")
        carbon_pu = moopf_results.get('carbon_emissions', 'N/A')
        carbon_raw = moopf_results.get('carbon_emissions_raw', 'N/A')
        if carbon_pu != 'N/A':
            print(f"  Carbon Emissions (per-unit): {carbon_pu:.6f}")
        if carbon_raw != 'N/A':
            print(f"  Carbon Emissions (raw): {carbon_raw:.2f}")

def save_best_model_results(best_model, best_run, moopf_results, renewable_impact_data, training_history, config, num_buses, is_physics_informed, iteration_details=None, param_keys=None, model_name="", output_dir=""):
    """Saves detailed results for the best model."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Metrics
    metrics = {
        'test_score': best_run.get('test_score'),
        'val_score': best_run.get('val_score'),
        **moopf_results
    }
    
    with open(os.path.join(output_dir, 'best_model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # Save Renewable Impact Data
    if renewable_impact_data is not None:
        renewable_impact_data.to_csv(os.path.join(output_dir, 'renewable_impact.csv'), index=False)
        
    # Training history is visualized in train_hist.png - no need to save CSV

def save_model_results_csv(best_run, moopf_results, config, num_buses, model_name, output_dir, iteration_details=None):
    """Save comprehensive model results to model_results.csv."""
    config_dict = best_run.get('config_dict', {})
    
    results = {
        'model_name': model_name,
        'bus_system': num_buses,
        'run_timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'hidden_dim': best_run.get('HIDDEN_DIM', config_dict.get('HIDDEN_DIM', '')),
        'gc_layers': best_run.get('NUM_GC_LAYERS', config_dict.get('NUM_GC_LAYERS', '')),
        'sequence_length': config_dict.get('SEQUENCE_LENGTH', ''),
        'train_mse': best_run.get('training_mse', ''),
        'train_rmse': np.sqrt(best_run.get('training_mse', 0)) if best_run.get('training_mse') else '',
        'epochs_trained': config.NUM_EPOCHS,
        'test_mse': best_run.get('test_score', ''),
        'test_rmse': np.sqrt(best_run.get('test_score', 0)) if best_run.get('test_score') else '',
        'avg_power_loss': moopf_results.get('power_loss', ''),
        'avg_voltage_dev': moopf_results.get('voltage_deviation', ''),
        'avg_carbon_emissions': moopf_results.get('carbon_emissions', ''),
        'moopf_score': moopf_results.get('moopf_score', ''),
        'optimization_method': 'MoSOA',
        'best_objective_score': best_run.get('test_score', '')
    }
    
    df = pd.DataFrame([results])
    csv_path = os.path.join(output_dir, 'model_results.csv')
    df.to_csv(csv_path, index=False)

def print_comprehensive_summary(all_results: List[Dict[str, Any]], config: Any = None):
    """Print comprehensive summary of all model performances."""
    if not all_results:
        print("\n No results to summarize.")
        return
    
    print(f"\n{'='*100}")
    print(f" COMPREHENSIVE FINAL SUMMARY - ALL MODELS & BUS SYSTEMS")
    print(f"{'='*100}")
    
    # Print table header
    print(f"{'Model':<15} {'Bus':<8} {'Type':<11} {'H.Dim':<7} {'Layers':<7} {'Train MSE':<12} {'Test MSE':<12} {'MOOPF':<12} {'P.Loss':<10} {'V.Dev':<10} {'Carbon':<10}")
    print("-" * 140)
    
    # Prepare data for CSV
    summary_data = []
    
    # Print each result
    for result in all_results:
        model_type = 'Physics' if result['is_physics_informed'] else 'NonPhys'
        train_mse_str = f"{result['training_mse']:.6f}" if result['training_mse'] != float('inf') else 'Failed'
        
        # Get Test MSE and MOOPF Score explicitly
        test_mse = result.get('test_mse')
        test_mse_str = f"{test_mse:.6f}" if isinstance(test_mse, (int, float)) else 'N/A'
        
        moopf = result.get('moopf_score')
        moopf_str = f"{moopf:.6f}" if isinstance(moopf, (int, float)) else 'N/A'
        
        # Get individual metrics
        p_loss = f"{result.get('power_loss', 'N/A'):.4f}" if isinstance(result.get('power_loss'), (int, float)) else 'N/A'
        v_dev = f"{result.get('voltage_deviation', 'N/A'):.4f}" if isinstance(result.get('voltage_deviation'), (int, float)) else 'N/A'
        carbon = f"{result.get('carbon_emissions', 'N/A'):.4f}" if isinstance(result.get('carbon_emissions'), (int, float)) else 'N/A'
        
        print(f"{result['model_name']:<15} {result['num_buses']:<8} {model_type:<11} {result['best_hidden_dim']:<7} {result['best_gc_layers']:<7} {train_mse_str:<12} {test_mse_str:<12} {moopf_str:<12} {p_loss:<10} {v_dev:<10} {carbon:<10}")
        
        # Append to CSV data
        summary_data.append({
            'Model': result['model_name'],
            'Bus System': result['num_buses'],
            'Type': model_type,
            'Hidden Dim': result['best_hidden_dim'],
            'GC Layers': result['best_gc_layers'],
            'Train MSE': train_mse_str,
            'Test MSE': test_mse_str,
            'MOOPF Score': moopf_str,
            'Power Loss': p_loss,
            'Voltage Deviation': v_dev,
            'Carbon Emissions': carbon,
            'Training Time': result.get('training_time', 'N/A')
        })
    
    print("-" * 140)
    
    # Save to CSV
    if config and hasattr(config, 'CURRENT_RUN_DIR'):
        try:
            csv_path = os.path.join(config.CURRENT_RUN_DIR, "comprehensive_summary.csv")
            pd.DataFrame(summary_data).to_csv(csv_path, index=False)
            print(f"\n [INFO] Comprehensive summary saved to: {csv_path}")
        except Exception as e:
            print(f"\n [WARNING] Failed to save comprehensive summary CSV: {e}")
    
    # Find best performers
    successful_results = [r for r in all_results if r['final_test_score'] != float('inf')]
    
    if successful_results:
        best_overall = min(successful_results, key=lambda x: x['final_test_score'])
        print(f"\n OVERALL BEST: {best_overall['model_name']} on {best_overall['num_buses']}-bus")
        print(f"   {best_overall['final_metric_name']}: {best_overall['final_test_score']:.6f}")
        print(f"   Config: {best_overall['best_hidden_dim']} hidden_dim, {best_overall['best_gc_layers']} layers")
        
        # Best per bus system
        print(f"\n BEST PER BUS SYSTEM:")
        bus_systems = sorted(set(r['num_buses'] for r in successful_results))
        for num_buses in bus_systems:
            bus_results = [r for r in successful_results if r['num_buses'] == num_buses]
            if bus_results:
                best_for_bus = min(bus_results, key=lambda x: x['final_test_score'])
                print(f"   {num_buses}-bus: {best_for_bus['model_name']} ({best_for_bus['final_metric_name']}: {best_for_bus['final_test_score']:.6f})")
        
        total_runs = len(all_results)
        successful_runs = len(successful_results)
        print(f"\n SUCCESS RATE: {successful_runs}/{total_runs} ({100*successful_runs/total_runs:.1f}%)")
    else:
        print("\n No successful model runs to analyze.")
    
    print(f"{'='*100}\n")
