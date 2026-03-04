import os
import argparse
import yaml
import torch
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

def main():
    parser = argparse.ArgumentParser(description="Train Power Flow Models")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # 1. Setup Data
    L.seed_everything(42)
    dm = PowerFlowDataModule(
        data_dir=config['data']['data_dir'],
        case_name=config['data']['case_name'],
        batch_size=config['data']['batch_size'],
        seq_len=config['data']['seq_len']
    )
    
    # 2. Setup Model
    model = build_model(config)
    print(f"Initialized {config['model']['name']}")
    
    # 3. Setup Callbacks & Logger
    tb_logger = TensorBoardLogger(
        save_dir=config['logger']['save_dir'],
        name=config['logger']['name'],
        version=f"{config['model']['name']}_{config['data']['case_name']}"
    )
    
    checkpoint_callback = ModelCheckpoint(**config['callbacks']['model_checkpoint'])
    early_stop_callback = EarlyStopping(**config['callbacks']['early_stopping'])
    
    # 4. Setup Trainer
    trainer = L.Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=config['trainer']['max_epochs'],
        accelerator=config['trainer']['accelerator'],
        devices=config['trainer']['devices'],
        log_every_n_steps=config['trainer']['log_every_n_steps']
    )
    
    # 5. Train
    print("Starting training...")
    trainer.fit(model, datamodule=dm)
    
    # 6. Test
    print("Starting testing...")
    trainer.test(model, datamodule=dm, ckpt_path="best")

if __name__ == "__main__":
    main()
