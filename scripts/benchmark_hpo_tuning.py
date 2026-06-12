import os
import sys
import time
import argparse
import yaml
import pandas as pd
import numpy as np
import inspect
import torch
from typing import Dict, Any

import logging
for name in ["lightning.pytorch", "lightning.fabric", "pytorch_lightning", "lightning"]:
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)
    logger.handlers = []
    logger.propagate = False

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", ".*For seamless cloud logging.*")
warnings.filterwarnings("ignore", ".*GPU available.*")
warnings.filterwarnings("ignore", ".*TPU available.*")
import os
os.environ["LIGHTNING_PYTORCH_DISABLE_TIP"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import SPATIAL_MODELS, RECURRENT_MODELS, PowerFlowDataModule, get_model_registry

MODEL_REGISTRY = get_model_registry()
from src.optimizers.mosoa import MoSOA
from src.optimizers.soa import SOA
from src.optimizers.tpe_wrapper import TPEOptimizer
from src.visualization.plot_hpo_tuning import run_hpo_plotting

# Global registry to store history of all HPO trials
TRIAL_RECORDS = []

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_model(model_name: str, params: dict, base_config: dict):
    ModelClass = MODEL_REGISTRY[model_name]
    
    # Merge base config args with the HPO generated params
    kwargs = {
        'in_channels': base_config['in_channels'],
        'out_channels': base_config['out_channels'],
        'learning_rate': params.get('learning_rate', 1e-3),
        'lr_patience': base_config.get('lr_patience', 10),
        'lr_factor': base_config.get('lr_factor', 0.5)
    }

    if 'gcn_hidden' in params:
        kwargs['gcn_hidden'] = params['gcn_hidden']
        if model_name in SPATIAL_MODELS:
            kwargs['hidden_channels'] = params['gcn_hidden']
    if 'lstm_hidden' in params: kwargs['lstm_hidden'] = params['lstm_hidden']
    if 'gru_hidden' in params: kwargs['gru_hidden'] = params['gru_hidden']
    
    if 'num_layers' in params:
        kwargs['num_layers'] = params['num_layers']
        kwargs['num_gcn_layers'] = params['num_layers']
        kwargs['num_res_blocks'] = params['num_layers']

    sig = inspect.signature(ModelClass.__init__)
    valid_params = [p for p in sig.parameters.keys() if p != 'self']
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    if not has_kwargs:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        return ModelClass(**filtered_kwargs)

    return ModelClass(**kwargs)

class SuppressOutput:
    def __enter__(self):
        self._null_fd = os.open(os.devnull, os.O_RDWR)
        self._save_stdout = os.dup(1)
        self._save_stderr = os.dup(2)
        os.dup2(self._null_fd, 1)
        os.dup2(self._null_fd, 2)

    def __exit__(self, *_):
        os.dup2(self._save_stdout, 1)
        os.dup2(self._save_stderr, 2)
        os.close(self._null_fd)
        os.close(self._save_stdout)
        os.close(self._save_stderr)

def get_objective_function(alg_name: str, model_name: str, case_name: str, base_model_cfg: dict, epochs: int, total_trials: int, hpo_config: dict):
    from tqdm import tqdm
    start_time = time.time()
    trial_count = [0]
    best_val_loss = [float('inf')]
    desc = f"{alg_name:<25}"
    pbar = tqdm(total=total_trials, desc=desc, leave=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} trials",
                unit="trial")
    
    actual_seq_len = base_model_cfg.get('seq_len', 4) if model_name in RECURRENT_MODELS else 1
    dm = PowerFlowDataModule(
        data_dir=os.path.join("data", "prep"),
        case_name=case_name,
        batch_size=base_model_cfg.get('batch_size', 32),
        seq_len=actual_seq_len
    )
    dm.setup(stage='fit')
    
    def objective(params: Dict[str, Any]) -> float:
        trial_count[0] += 1
        
        for key in ['gcn_hidden', 'num_layers', 'lstm_hidden', 'gru_hidden', 'batch_size']:
            if key in params:
                params[key] = int(params[key])
                
        model = build_model(model_name, params, base_model_cfg)
        
        try:
            with SuppressOutput():
                trainer = L.Trainer(
                    logger=False,
                    enable_checkpointing=False,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    max_epochs=epochs,
                    limit_train_batches=hpo_config.get('limit_train_batches', 1.0),
                    limit_val_batches=hpo_config.get('limit_val_batches', 1.0),
                    accelerator="auto",
                    callbacks=[EarlyStopping(monitor="val_loss", patience=hpo_config.get('patience', 3), mode="min")]
                )
                trainer.fit(model, datamodule=dm)
            val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(float('inf'))).item()
        except Exception as e:
            val_loss = float('inf')
            
        elapsed = time.time() - start_time
        
        # Record everything
        record = {
            "Algorithm": alg_name,
            "Model": model_name,
            "Trial": trial_count[0],
            "Time_Elapsed_s": elapsed,
            "Val_Loss": val_loss
        }
        record.update(params)
        TRIAL_RECORDS.append(record)
        
        pbar.update(1)
        if val_loss < best_val_loss[0]:
            best_val_loss[0] = val_loss
        pbar.set_postfix(best=f"{best_val_loss[0]:.6g}")
        if trial_count[0] >= total_trials:
            pbar.close()
        
        return val_loss
        
    return objective

def run_random_search(objective_fn, search_space, n_trials, seed=42):
    np.random.seed(seed)
    best_val = float('inf')
    best_params = None
    for _ in range(n_trials):
        params = {}
        for k, limits in search_space.items():
            if isinstance(limits[0], int) and isinstance(limits[1], int):
                params[k] = np.random.randint(limits[0], limits[1] + 1)
            else:
                params[k] = np.random.uniform(limits[0], limits[1])
        val = objective_fn(params)
        if val < best_val:
            best_val = val
            best_params = params
    return best_params

def main():
    parser = argparse.ArgumentParser(description='HPO Tuning Benchmark for PISTGNN')
    parser.add_argument('--case', type=str, default='case33', help='Power system case')
    parser.add_argument('--model', type=str, default='StandardGCN', help='Model to tune')
    parser.add_argument('--all-models', action='store_true', help='Benchmark all available models')
    parser.add_argument('--epochs', type=int, default=-1, help='Override YAML max epochs (optional)')
    parser.add_argument('--trials', type=int, default=-1, help='Override YAML total trials (optional)')
    args = parser.parse_args()

    # Load MoSOA config
    mosoa_config = load_config("configs/mosoa.yaml")
    if 'hpo_tuning' not in mosoa_config:
        raise KeyError("hpo_tuning not found in configs/mosoa.yaml")
    hpo_config = mosoa_config['hpo_tuning']
    search_space = hpo_config['space']
    
    epochs = args.epochs if args.epochs > 0 else hpo_config.get('epochs', 10)
    trials = args.trials if args.trials > 0 else hpo_config.get('trials', 30)
    seed = hpo_config.get('seed', 42)
    base_pop_size = hpo_config.get('pop_size', 10)
    
    # Load Training config for base params
    train_config = load_config("configs/training.yaml")
    base_model_cfg = train_config['model']

    if args.all_models:
        models_to_run = list(MODEL_REGISTRY.keys())
    else:
        models_to_run = [args.model]

    for current_model in models_to_run:
        print(f"\n{'='*85}\n Starting HPO Benchmark | Model: {current_model} | Case: {args.case} | Trials: {trials} \n{'='*85}\n")
        
        # Run Random Search
        print("\n>>> Running Random Search...")
        obj_rs = get_objective_function("Random Search", current_model, args.case, base_model_cfg, epochs, trials, hpo_config)
        run_random_search(obj_rs, search_space, n_trials=trials, seed=seed)

        # Run TPE
        print("\n>>> Running TPE (Optuna)...")
        obj_tpe = get_objective_function("TPE (Optuna)", current_model, args.case, base_model_cfg, epochs, trials, hpo_config)
        tpe_opt = TPEOptimizer(search_space=search_space, seed=seed)
        tpe_opt.optimize(obj_tpe, n_trials=trials, verbose=False)

        # Run SOA
        pop_size = max(1, min(base_pop_size, trials))
        print("\n>>> Running SOA...")
        obj_soa = get_objective_function("SOA", current_model, args.case, base_model_cfg, epochs, trials, hpo_config)
        soa_opt = SOA(search_space=search_space, seed=seed, pop_size=pop_size)
        soa_opt.optimize(obj_soa, n_trials=trials, verbose=False)

        # Run MoSOA
        print("\n>>> Running MoSOA...")
        mosoa_params = mosoa_config.get('mosoa_params', {})
        
        obj_mosoa = get_objective_function("MoSOA", current_model, args.case, base_model_cfg, epochs, trials, hpo_config)
        mosoa_opt = MoSOA(search_space=search_space, seed=seed, pop_size=pop_size, **mosoa_params)
        mosoa_opt.optimize(obj_mosoa, n_trials=trials, verbose=False)

    # Export History
    df = pd.DataFrame(TRIAL_RECORDS)
    out_dir = os.path.join("reports", "mosoa", "hpo_tuning", args.case)
    if os.path.exists(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "real_hpo_history.csv")
    df.to_csv(out_path, index=False)
    
    print(f"\nBenchmark complete\n")
    
    # Print clean results table to terminal
    print("=" * 85)
    print(f" FINAL RESULTS: {args.case} ")
    print("=" * 85)
    
    # Find the row with the best validation loss for each model-algorithm combination
    best_results = df.loc[df.groupby(['Model', 'Algorithm'])['Val_Loss'].idxmin()].copy()
    
    # Map the actual total completion time
    total_times = df.groupby(['Model', 'Algorithm'])['Time_Elapsed_s'].max()
    best_results['Total_Time_s'] = best_results.set_index(['Model', 'Algorithm']).index.map(total_times)
    
    # Organize columns for display
    display_cols = ['Model', 'Algorithm', 'Val_Loss', 'Total_Time_s'] + list(search_space.keys())
    
    print(best_results[display_cols].sort_values(['Model', 'Val_Loss']).to_string(index=False))
    print("=" * 85 + "\n")
    
    # Trigger plotting automatically
    print("Generating HPO performance plots...")
    run_hpo_plotting(df, out_dir)

if __name__ == "__main__":
    main()
