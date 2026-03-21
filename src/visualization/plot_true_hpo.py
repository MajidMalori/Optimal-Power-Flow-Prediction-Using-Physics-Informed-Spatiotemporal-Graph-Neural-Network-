import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def plot_convergence_time(df, out_path):
    plt.figure(figsize=(10, 6))
    
    algorithms = df['Algorithm'].unique()
    
    for alg in algorithms:
        alg_df = df[df['Algorithm'] == alg].sort_values('Time_Elapsed_s')
        alg_df['Cum_Min'] = alg_df['Val_Loss'].cummin()
        plt.plot(alg_df['Time_Elapsed_s'], alg_df['Cum_Min'], label=alg, marker='o', markersize=4, linewidth=2)
        
    plt.xlabel('Cumulative Time Elapsed (s)', fontweight='bold')
    plt.ylabel('Best Validation Loss', fontweight='bold')
    plt.title('Computational Efficiency: Accuracy vs. Computation Time', fontweight='bold', pad=20)
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_convergence_trials(df, out_path):
    plt.figure(figsize=(10, 6))
    
    algorithms = df['Algorithm'].unique()
    
    for alg in algorithms:
        alg_df = df[df['Algorithm'] == alg].sort_values('Trial')
        alg_df['Cum_Min'] = alg_df['Val_Loss'].cummin()
        plt.plot(alg_df['Trial'], alg_df['Cum_Min'], label=alg, marker='o', markersize=4, linewidth=2)
        
    plt.xlabel('Number of Evaluated Trials', fontweight='bold')
    plt.ylabel('Best Validation Loss', fontweight='bold')
    plt.title('Sample Efficiency: Accuracy vs. Number of Trials', fontweight='bold', pad=20)
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_parallel_coordinates(df, out_path):
    from pandas.plotting import parallel_coordinates
    
    plt.figure(figsize=(12, 6))
    
    mosoa_df = df[df['Algorithm'] == 'MoSOA'].copy()
    if len(mosoa_df) == 0: 
        return
    
    hps = [c for c in mosoa_df.columns if c not in ['Algorithm', 'Model', 'Trial', 'Time_Elapsed_s', 'Val_Loss']]
    
    normalized_df = mosoa_df[hps].copy()
    for col in hps:
        min_v = normalized_df[col].min()
        max_v = normalized_df[col].max()
        if max_v > min_v:
            normalized_df[col] = (normalized_df[col] - min_v) / (max_v - min_v)
            
    normalized_df['Val_Loss'] = mosoa_df['Val_Loss']
    
    threshold = normalized_df['Val_Loss'].quantile(0.2)
    normalized_df['Performance'] = ['Top 20%' if v <= threshold else 'Other' for v in normalized_df['Val_Loss']]
    normalized_df = normalized_df.drop('Val_Loss', axis=1)
    
    parallel_coordinates(normalized_df, 'Performance', colormap='coolwarm', alpha=0.6)
    
    plt.title('MoSOA Hyperparameter Exploration Landscape (Normalized)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_tradeoff_scatter(df, out_path):
    plt.figure(figsize=(8, 8))
    
    summary = df.groupby('Algorithm').agg(
        Total_Time=('Time_Elapsed_s', 'max'),
        Best_Val_Loss=('Val_Loss', 'min')
    ).reset_index()
    
    sns.scatterplot(data=summary, x='Total_Time', y='Best_Val_Loss', hue='Algorithm', s=200, style='Algorithm')
    
    for i in range(summary.shape[0]):
        plt.text(summary['Total_Time'][i] * 1.05, summary['Best_Val_Loss'][i], 
                 summary['Algorithm'][i], horizontalalignment='left', size='medium', color='black', weight='semibold')
                 
    plt.xlabel('Total Execution Time (s)')
    plt.ylabel('Minimum Validation Loss')
    plt.title('Executive Summary: Time vs Accuracy Trade-off', fontweight='bold', pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_cross_model_comparison(df, out_path):
    plt.figure(figsize=(14, 7))
    
    summary = df.groupby(['Model', 'Algorithm']).agg(Best_Val_Loss=('Val_Loss', 'min')).reset_index()
    
    sns.barplot(data=summary, x='Model', y='Best_Val_Loss', hue='Algorithm', palette='viridis')
    
    min_val = summary['Best_Val_Loss'].min()
    if min_val > 1e-12:
        plt.yscale('log')
        plt.ylabel('Best Validation Loss (Log Scale)', fontweight='bold')
    else:
        plt.ylabel('Best Validation Loss', fontweight='bold')
        
    plt.xlabel('Model Architecture', fontweight='bold')
    plt.title('Tuner Performance Comparison Across All Models', fontweight='bold', pad=20)
    plt.legend(title='Optimizer', frameon=True, shadow=True, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot True HPO Benchmarking Results')
    parser.add_argument('--input', type=str, default='reports/mosoa/true_hpo/real_hpo_history.csv')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found. Run benchmark_true_hpo.py first.")
        return
        
    df = pd.read_csv(args.input)
    out_dir = os.path.dirname(args.input)
    
    # Generate the requested multi-model grouped bar chart if multiple models exist
    if len(df['Model'].unique()) > 1:
        plot_cross_model_comparison(df, os.path.join(out_dir, 'cross_model_tuning_comparison.png'))
    
    # Automatically generate specific convergence plots for every model independently
    for model_name in df['Model'].unique():
        model_df = df[df['Model'] == model_name].copy()
        
        prefix = f"{model_name}_" if len(df['Model'].unique()) > 1 else ""
        
        plot_convergence_time(model_df, os.path.join(out_dir, f'{prefix}convergence_vs_time.png'))
        plot_convergence_trials(model_df, os.path.join(out_dir, f'{prefix}convergence_vs_trials.png'))
        plot_parallel_coordinates(model_df, os.path.join(out_dir, f'{prefix}mosoa_parallel_coordinates.png'))
        plot_tradeoff_scatter(model_df, os.path.join(out_dir, f'{prefix}tradeoff_scatter.png'))
    
    print(f"Plots saved to {out_dir}")

if __name__ == '__main__':
    main()
