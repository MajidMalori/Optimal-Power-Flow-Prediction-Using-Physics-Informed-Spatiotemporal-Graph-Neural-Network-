#!/usr/bin/env python3
import os
import sys
import shutil
import warnings

# Absolute silence for environment warnings
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_CONSOLE"] = "off"
os.environ["LIGHTNING_PYTORCH_DISABLE_TIP"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

warnings.filterwarnings("ignore") # Global ignore for all warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import time
import glob
import torch
import yaml
import argparse
import numpy as np
import logging
import warnings
from tqdm import tqdm

# Silence font-related warnings before they trigger
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*findfont: Generic family.*")
warnings.filterwarnings("ignore", message=".*findfont: Font family.*")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models import MODEL_REGISTRY, PowerFlowDataModule, SPATIAL_MODELS, RECURRENT_MODELS
from src.constants import TargetIndices
from src.visualization.plot_uncertainty import plot_spatial_comparison_grid, plot_temporal_comparison_curves

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

def get_latest_checkpoint(model_name, case_name):
    pattern = os.path.join(PROJECT_ROOT, "checkpoints", "**", f"{model_name}_{case_name}", "*.ckpt")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def run_uncertainty_analysis(model_name, case_name, ckpt_path, device, num_tta=10):
    # 1. Load Data
    config_path = os.path.join(PROJECT_ROOT, "configs", "training.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize DataModule correctly
    data_dir = os.path.join(PROJECT_ROOT, "data", "prep")
    dm = PowerFlowDataModule(
        data_dir=data_dir,
        case_name=case_name,
        batch_size=config['data'].get("batch_size", 32),
        seq_len=1 if model_name in SPATIAL_MODELS else config['data'].get("seq_len", 1)
    )
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()

    # 2. Load Model
    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass.load_from_checkpoint(ckpt_path, strict=False)
    model.to(device)
    model.eval()

    # Store predictions, targets, and renewable fractions for uncertainty analysis
    all_preds = []
    all_targets = []

    # 3. Evaluation Loop
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            x = batch["features"]
            targets = batch["targets"]

            # Handle Spatial vs Recurrent batch keys
            is_recurrent = model_name in RECURRENT_MODELS
            edge_index = batch["edge_index_seq"] if is_recurrent else batch["edge_index"]

            # --- TTA Parameters ---
            unc_cfg = config.get("evaluation", {}).get("uncertainty", {})
            noise_scale = unc_cfg.get("tta_noise_scale", 0.05)
            noise_features_names = unc_cfg.get("noise_features", ["P_LOAD", "Q_LOAD", "P_REN", "Q_REN"])
            
            from src.constants import FeatureIndices
            input_indices = [getattr(FeatureIndices, name) for name in noise_features_names]
            
            # --- TTA Loop ---
            tta_preds = []
            
            for _ in range(num_tta):
                # Apply proportional noise to all target features
                noise_std = torch.abs(x) * noise_scale + 0.001
                noise = torch.randn_like(x) * noise_std
                
                # Apply noise to specific inputs
                mask = torch.zeros_like(x)
                mask[..., input_indices] = 1.0
                x_perturbed = x + (noise * mask)
                
                # Forward pass
                p = model(x_perturbed, edge_index)
                tta_preds.append(p)
            
            tta_preds = torch.stack(tta_preds) # [M, B, N, 2]
            
            # Predictive Uncertainty (Model Doubt)
            # We use the standard deviation across TTA samples
            preds_std = tta_preds.std(dim=0) # [B, N, 2]
            # Combined uncertainty: RMS of VM and VA standard deviations
            uncertainty_metric = torch.sqrt(torch.mean(preds_std**2, dim=-1)) # [B, N]

            from src.constants import TargetIndices
            targets_vm_va = targets[..., TargetIndices.VM:TargetIndices.VA+1]

            # Store for uncertainty analysis
            # We store the TTA-based uncertainty metric
            all_preds.append(uncertainty_metric.cpu().numpy())
            all_targets.append(targets_vm_va.cpu().numpy()) # Keep for shape matching if needed

    # Uncertainty Post-processing
    preds_np = np.concatenate(all_preds, axis=0) # (N, nodes)
    targets_np = np.concatenate(all_targets, axis=0) # (N, nodes, 2)
    
    n_samples = preds_np.shape[0]
    num_fractions = 6 # Standard for this project
    n_per_frac = n_samples // num_fractions
    
    unique_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    uncertainty_results = {}
    for i, frac in enumerate(unique_fractions):
        start_idx = i * n_per_frac
        # Ensure we cover the whole set in the last fraction
        end_idx = (i + 1) * n_per_frac if i < num_fractions - 1 else n_samples
        
        if start_idx >= n_samples: break
        
        # TTA-based uncertainty is already in all_preds
        uncertainty = preds_np[start_idx:end_idx] # (M, nodes)
        
        # Spatial: Average uncertainty per node
        spatial_unc = np.mean(uncertainty, axis=0)
        
        # Temporal: Average uncertainty per timestep
        temporal_unc = np.mean(uncertainty, axis=1)
        
        # We need to map temporal to 24 hours if possible
        # For simplicity, if we have 96 steps per day, we'll take mean across days
        steps_per_day = 96
        m_samples = len(temporal_unc)
        
        if m_samples >= steps_per_day:
            n_days = m_samples // steps_per_day
            daily_data = temporal_unc[:n_days*steps_per_day].reshape(n_days, steps_per_day)
            t_mean = np.mean(daily_data, axis=0)
            t_std = np.std(daily_data, axis=0)
        else:
            t_mean = temporal_unc
            t_std = np.zeros_like(t_mean)
        
        uncertainty_results[frac] = {
            'spatial': spatial_unc,
            'mean_spatial': np.mean(spatial_unc),
            'temporal_mean': t_mean,
            'temporal_std': t_std
        }

    return {
        "case": case_name,
        "model": model_name,
        "uncertainty": uncertainty_results
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default="case33")
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tta_samples", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.case.lower() == "all":
        cases = ["case33", "case57", "case118"]
    else:
        cases = [args.case if args.case.startswith("case") else f"case{args.case}"]

    model_list = list(MODEL_REGISTRY.keys()) if args.model.lower() == "all" else [args.model]

    for case in cases:
        print(f"\nUncertainty Analysis: {case}")
        print("-" * 40)
        
        # Reports go to reports/evaluation/uncertainty/[case]
        uncertainty_dir = os.path.join(PROJECT_ROOT, "reports", "evaluation", "uncertainty", case)
        if os.path.exists(uncertainty_dir):
            shutil.rmtree(uncertainty_dir)
        os.makedirs(uncertainty_dir, exist_ok=True)
        
        csv_dir = os.path.join(uncertainty_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        
        all_results = []
        
        pbar = tqdm(model_list, desc=f"Analyzing {case}", leave=True, dynamic_ncols=True)
        
        for model_name in pbar:
            pbar.set_postfix_str(f"Processing {model_name}... ")
            ckpt = args.checkpoint if args.checkpoint else get_latest_checkpoint(model_name, case)
            if not ckpt:
                continue
            try:
                res = run_uncertainty_analysis(model_name, case, ckpt, device, num_tta=args.tta_samples)
                all_results.append(res)
                
                # Immediate Uncertainty Plotting
                if 'uncertainty' in res and res['uncertainty']:
                    spatial_path = os.path.join(uncertainty_dir, f"uncertainty_spatial_{model_name}.png")
                    plot_spatial_comparison_grid(res['uncertainty'], case, spatial_path, model_name)
                    temporal_path = os.path.join(uncertainty_dir, f"uncertainty_temporal_{model_name}.png")
                    plot_temporal_comparison_curves(res['uncertainty'], case, temporal_path, model_name)
                    
            except Exception as e:
                pass # Silently continue on errors to maintain clean output
                
        # Final Tabular Summary report
        if all_results:
            fractions = sorted(all_results[0]['uncertainty'].keys())
            frac_labels = [f"{int(f*100)}%" for f in fractions]
            
            print("\n" + "="*80)
            print("SPATIAL UNCERTAINTY SUMMARY (Mean Node Standard Deviation in p.u.)")
            print("="*80)
            header = f"{'Model':<20} | " + " | ".join([f"{l:>8}" for l in frac_labels])
            for res in all_results:
                row = f"{res['model']:<20} | "
                row += " | ".join([f"{res['uncertainty'][f]['mean_spatial']:8.6f}" for f in fractions])
                print(row)
                
            print("\n" + "="*80)
            print("TEMPORAL UNCERTAINTY SUMMARY (Mean Temporal StdDev in p.u.)")
            print("="*80)
            header = f"{'Model':<20} | " + " | ".join([f"{l:>8}" for l in frac_labels])
            print(header)
            print("-" * len(header))
            
            for res in all_results:
                row = f"{res['model']:<20} | "
                # temporal_mean holds the actual mean TTA uncertainty per hour
                row += " | ".join([f"{np.mean(res['uncertainty'][f]['temporal_mean']):8.6f}" for f in fractions])
                print(row)
            
            # Save to CSV
            import pandas as pd
            
            spatial_data = []
            temporal_data = []
            
            for res in all_results:
                row_sp = {"Model": res['model']}
                row_tmp = {"Model": res['model']}
                
                for f in fractions:
                    label = f"{int(f*100)}%"
                    row_sp[label] = res['uncertainty'][f]['mean_spatial']
                    row_tmp[label] = np.mean(res['uncertainty'][f]['temporal_mean'])
                
                spatial_data.append(row_sp)
                temporal_data.append(row_tmp)
                
            df_sp = pd.DataFrame(spatial_data)
            df_sp.to_csv(os.path.join(csv_dir, f"{case}_spatial_uncertainty.csv"), index=False)
            
            df_tmp = pd.DataFrame(temporal_data)
            df_tmp.to_csv(os.path.join(csv_dir, f"{case}_temporal_uncertainty.csv"), index=False)
            print("")


if __name__ == "__main__":
    main()
