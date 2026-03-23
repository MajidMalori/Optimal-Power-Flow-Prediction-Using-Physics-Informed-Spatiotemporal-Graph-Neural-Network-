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
        
        # If multiple runs exist, group and calculate mean/std for shaded region
        if 'Run' in alg_df.columns and len(alg_df['Run'].unique()) > 1:
            mean_df = alg_df.groupby('Trial').agg(
                Mean_Loss=('Val_Loss', 'mean'),
                Std_Loss=('Val_Loss', 'std'),
                Mean_Time=('Time_Elapsed_s', 'mean')
            ).reset_index()
            
            mean_df['Cum_Min'] = mean_df['Mean_Loss'].cummin()
            std_err = mean_df['Std_Loss'].cummin() # Rough estimate for shading
            
            # Shaded uncertainty region (Standard Deviation)
            plt.fill_between(mean_df['Mean_Time'], 
                            mean_df['Cum_Min'] - std_err, 
                            mean_df['Cum_Min'] + std_err, 
                            alpha=0.15)
            
            # Professional Mean line
            plt.plot(mean_df['Mean_Time'], mean_df['Cum_Min'], label=f"{alg} (Avg)", linewidth=2.5)
        else:
            alg_df['Cum_Min'] = alg_df['Val_Loss'].cummin()
            plt.plot(alg_df['Time_Elapsed_s'], alg_df['Cum_Min'], label=alg, marker='o', markersize=4, linewidth=2)
        
    plt.xlabel('Cumulative Time Elapsed (s)', fontweight='bold')
    plt.ylabel('Best Validation Loss', fontweight='bold')
    case_name = out_path.split(os.sep)[-2] if os.sep in out_path else ""
    title_prefix = f"[{case_name}] " if case_name else ""
    plt.title(f'{title_prefix}Computational Efficiency: Accuracy vs. Computation Time', fontweight='bold', pad=20)
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
    case_name = out_path.split(os.sep)[-2] if os.sep in out_path else ""
    title_prefix = f"[{case_name}] " if case_name else ""
    plt.title(f'{title_prefix}Sample Efficiency: Accuracy vs. Number of Trials', fontweight='bold', pad=20)
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
    
    case_name = out_path.split(os.sep)[-2] if os.sep in out_path else ""
    title_prefix = f"[{case_name}] " if case_name else ""
    plt.title(f'{title_prefix}MoSOA Hyperparameter Exploration Landscape (Normalized)')
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
    
    sns.scatterplot(data=summary, x='Total_Time', y='Best_Val_Loss', hue='Algorithm', s=200, style='Algorithm', palette='magma')
    
    plt.xlabel('Total Execution Time (s)', fontweight='bold')
    plt.ylabel('Minimum Validation Loss', fontweight='bold')
    
    # Improved labeling to avoid overlap with markers (No Arrows as requested)
    for i in range(summary.shape[0]):
        plt.annotate(summary['Algorithm'][i], 
                     (summary['Total_Time'][i], summary['Best_Val_Loss'][i]),
                     textcoords="offset points", 
                     xytext=(10,10), 
                     ha='left', 
                     fontsize=10, 
                     fontweight='bold',
                     color='darkred')
    case_name = out_path.split(os.sep)[-2] if os.sep in out_path else ""
    title_prefix = f"[{case_name}] " if case_name else ""
    plt.title(f'{title_prefix}Executive Summary: Time vs Accuracy Trade-off', fontweight='bold', pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_cross_model_comparison(df, out_path):
    plt.figure(figsize=(14, 7))
    
    summary = df.groupby(['Model', 'Algorithm']).agg(Best_Val_Loss=('Val_Loss', 'min')).reset_index()
    
    # Sort by performance for clarity
    summary = summary.sort_values(['Model', 'Best_Val_Loss'])
    
    ax = sns.barplot(data=summary, x='Model', y='Best_Val_Loss', hue='Algorithm', palette='viridis')
    
    # Annotate with ranks using robust container API
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontweight='bold', fontsize=9)

    min_val = summary['Best_Val_Loss'].min()
    if min_val > 0 and (summary['Best_Val_Loss'].max() / min_val) > 100:
        plt.yscale('log')
        plt.ylabel('Best Validation Loss (Log Scale)', fontweight='bold')
    else:
        plt.ylabel('Best Validation Loss', fontweight='bold')
        
    plt.xlabel('Model Architecture', fontweight='bold')
    case_name = out_path.split(os.sep)[-2] if os.sep in out_path else ""
    title_prefix = f"[{case_name}] " if case_name else ""
    plt.title(f'{title_prefix}Tuner Performance Comparison Across All Models', fontweight='bold', pad=20)
    plt.legend(title='Optimizer', frameon=True, shadow=True, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_hpo_plotting(df, out_dir):
    """
    Core plotting logic that can be called from other scripts.
    Groups results by 'Model' and compares 'Algorithms'.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Executive Summary: Trade-off Scatter (All results in df)
    plot_tradeoff_scatter(df, os.path.join(out_dir, 'algorithm_tradeoff_comparison.png'))
    
    # 2. Performance Comparison Bar Chart (All models in df)
    plot_cross_model_comparison(df, os.path.join(out_dir, 'cross_model_tuning_comparison.png'))
    
    # 3. Model-specific convergence and detailed analysis
    for model_name in df['Model'].unique():
        model_df = df[df['Model'] == model_name].copy()
        
        # If there are multiple models, prefix filenames for clarity
        prefix = f"{model_name}_" if len(df['Model'].unique()) > 1 else ""
        
        plot_convergence_time(model_df, os.path.join(out_dir, f'{prefix}convergence_vs_time.png'))
        plot_convergence_trials(model_df, os.path.join(out_dir, f'{prefix}convergence_vs_trials.png'))
        
        # Parallel coordinates (MoSOA high-dimensional analysis)
        plot_parallel_coordinates(model_df, os.path.join(out_dir, f'{prefix}mosoa_parallel_coordinates.png'))
    
    print(f"Research-grade HPO plots saved to {out_dir}")

def main():
    parser = argparse.ArgumentParser(description='Plot HPO Tuning Results')
    parser.add_argument('--input', type=str, default=None, help='Direct path to CSV file')
    parser.add_argument('--case', type=str, default='case33', help='Power system case to plot')
    args = parser.parse_args()
    
    csv_path = args.input
    if csv_path is None:
        csv_path = os.path.join('reports', 'mosoa', 'hpo_tuning', args.case, 'real_hpo_history.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run benchmark_hpo_tuning.py for {args.case} first.")
        return
        
    df = pd.read_csv(csv_path)
    out_dir = os.path.dirname(csv_path)
    
    run_hpo_plotting(df, out_dir)

if __name__ == '__main__':
    main()
