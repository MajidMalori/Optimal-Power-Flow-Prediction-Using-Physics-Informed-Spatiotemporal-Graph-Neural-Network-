import os
import argparse
import sys
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from src.models import (
    StandardGCN, DynamicGCN, PIGCN, 
    PIGCLSTM, PIGCGRU, 
    PIResnetGCLSTM, PIResnetGCGRU, 
    PowerFlowDataModule
)

MODEL_REGISTRY = {
    "StandardGCN": StandardGCN,
    "DynamicGCN": DynamicGCN,
    "PIGCN": PIGCN,
    "PIGCLSTM": PIGCLSTM,
    "PIGCGRU": PIGCGRU,
    "PIResnetGCLSTM": PIResnetGCLSTM,
    "PIResnetGCGRU": PIResnetGCGRU
}

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_model(config):
    model_cfg = config['model']
    model_name = model_cfg['name']
    
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry. Options: {list(MODEL_REGISTRY.keys())}")
        
    ModelClass = MODEL_REGISTRY[model_name]
    
    # Map config keys to model kwargs to handle varying signatures
    kwargs = {
        'in_channels': model_cfg['in_channels'],
        'out_channels': model_cfg['out_channels'],
        'learning_rate': model_cfg.get('learning_rate', 1e-3),
        'lr_patience': model_cfg.get('lr_patience', 10),
        'lr_factor': model_cfg.get('lr_factor', 0.5)
    }
    
    # Fill in specific architecture hidden size parameters
    if 'gcn_hidden' in model_cfg: kwargs['gcn_hidden'] = model_cfg['gcn_hidden']
    if 'lstm_hidden' in model_cfg: kwargs['lstm_hidden'] = model_cfg['lstm_hidden']
    if 'gru_hidden' in model_cfg: kwargs['gru_hidden'] = model_cfg['gru_hidden']
    if 'hidden_channels' in model_cfg: kwargs['hidden_channels'] = model_cfg['hidden_channels']
    
    # Fallbacks: If config says 'gcn_hidden' but model needs 'hidden_channels'
    if model_name in ["StandardGCN", "DynamicGCN", "PIGCN"]:
        if 'hidden_channels' not in kwargs and 'gcn_hidden' in kwargs:
            kwargs['hidden_channels'] = kwargs['gcn_hidden']
            
    if 'num_layers' in model_cfg:
        kwargs['num_layers'] = model_cfg['num_layers']
        kwargs['num_gcn_layers'] = model_cfg['num_layers']
        kwargs['num_res_blocks'] = model_cfg['num_layers']
        
    # Instantiate
    # We pass all kwargs and let **kwargs in __init__ absorb the ones we explicitly grab
    
    # Filter kwargs to exactly what the signatures need (or let models define their own logic, but filtering is safer)
    import inspect
    sig = inspect.signature(ModelClass.__init__)
    valid_params = [p for p in sig.parameters.keys() if p != 'self']
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    
    if not has_kwargs:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        return ModelClass(**filtered_kwargs)
    
    return ModelClass(**kwargs)

import copy

def main():
    parser = argparse.ArgumentParser(description="Train Power Flow Models")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config")
    parser.add_argument("--case", type=str, default=None, help="Comma separated list of cases (e.g., '33,57' or 'all')")
    parser.add_argument("--models", type=str, default=None, help="Comma separated list of models to train or 'all'")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Determine cases to process
    if args.case:
        if args.case.lower() == 'all':
            # Default known datasets
            cases = ['case33', 'case57', 'case118']
        else:
            cases = [c.strip() if c.strip().startswith('case') else f"case{c.strip()}" for c in args.case.split(',')]
    else:
        cases = [config['data']['case_name']]
        
    # Determine models to train
    if args.models:
        if args.models.lower() == 'all':
            models = list(MODEL_REGISTRY.keys())
        else:
            models = [m.strip() for m in args.models.split(',')]
    else:
        models = [config['model']['name']]

    for case_name in cases:
        for model_name in models:
            print(f"\n{'='*50}")
            print(f"Training {model_name} on {case_name}")
            print(f"{'='*50}\n")
            
            run_config = copy.deepcopy(config)
            run_config['data']['case_name'] = case_name
            run_config['model']['name'] = model_name
            
            # Spatial models only support seq_len = 1
            is_recurrent = model_name in ["PIGCLSTM", "PIGCGRU", "PIResnetGCLSTM", "PIResnetGCGRU"]
            actual_seq_len = run_config['data']['seq_len'] if is_recurrent else 1
            
            # 1. Setup Data
            L.seed_everything(42)
            dm = PowerFlowDataModule(
                data_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "data", "03_processed"),
                case_name=case_name,
                batch_size=run_config['data']['batch_size'],
                seq_len=actual_seq_len
            )
            
            # 2. Setup Model
            try:
                model = build_model(run_config)
                print(f"Initialized {model_name}")
            except Exception as e:
                print(f"Failed to initialize {model_name}: {e}")
                continue
            
            # 3. Setup Callbacks & Logger
            tb_logger = TensorBoardLogger(
                save_dir=run_config['logger']['save_dir'],
                name=run_config['logger']['name'],
                version=f"{model_name}_{case_name}"
            )
            
            # Ensure unique checkpoint directory to avoid overwriting between runs
            ckpt_dir = os.path.join(run_config['logger']['save_dir'], run_config['logger']['name'], f"{model_name}_{case_name}", "checkpoints")
            
            checkpoint_callback = ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor=run_config['callbacks']['model_checkpoint']['monitor'],
                filename="{epoch:02d}-{val_loss:.4f}",
                save_top_k=run_config['callbacks']['model_checkpoint']['save_top_k'],
                mode=run_config['callbacks']['model_checkpoint']['mode']
            )
            early_stop_callback = EarlyStopping(**run_config['callbacks']['early_stopping'])
            
            # 4. Setup Trainer
            trainer = L.Trainer(
                logger=tb_logger,
                callbacks=[checkpoint_callback, early_stop_callback],
                max_epochs=run_config['trainer']['max_epochs'],
                accelerator=run_config['trainer']['accelerator'],
                devices=run_config['trainer']['devices'],
                log_every_n_steps=run_config['trainer']['log_every_n_steps']
            )
            
            # 5. Train
            print("Starting training...")
            try:
                trainer.fit(model, datamodule=dm)
            except Exception as e:
                print(f"Training failed for {model_name} on {case_name}: {e}")
                continue
            
            # 6. Test
            print("Starting testing...")
            try:
                trainer.test(model, datamodule=dm, ckpt_path="best")
            except Exception as e:
                print(f"Testing failed for {model_name} on {case_name}: {e}")

if __name__ == "__main__":
    main()
