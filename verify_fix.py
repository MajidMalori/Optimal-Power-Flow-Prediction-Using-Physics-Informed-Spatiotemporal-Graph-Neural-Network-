import os
import torch
import torch.nn as nn
import numpy as np
from config import Config
from utils.data_loader import load_power_system_data, create_data_loaders
from utils.metrics import PowerSystemLoss
from trainers.model_trainer import PowerSystemTrainer
from models.adaptive_pigcn import AdaptivePIGCN

def verify_fix():
    print("Starting verification of Physics Loss Fix...")
    
    # 1. Setup Config
    # Create a temporary config for verification
    config = Config(
        yaml_config_path='config.yaml',
        load_yaml=True,
        data_mode='test',
        save_results=False,
        train_timesteps=1000, # Shorten dataset for speed
        test_timesteps=100
    )
    
    # Override for verification
    config.NUM_BUSES = 33
    config.CASE_NAME = 'case33'
    config.BATCH_SIZE = 32
    config.NUM_EPOCHS = 5
    config.LEARNING_RATE = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Load Data
    print("Loading data...")
    data_tuple = load_power_system_data(config, 'case33')
    _file_metadata, _adjacency, _ybus_metadata, _normalizer, _topology_cache, _topology_ids = data_tuple
    
    loaders = create_data_loaders(
        _file_metadata, _adjacency, _ybus_metadata, _normalizer, config, 
        is_static=True, topology_cache=_topology_cache, topology_ids=_topology_ids
    )
    train_loader, val_loader, test_loader = loaders
    
    # 3. Create Model
    print("Creating AdaptivePIGCN model...")
    # Minimal kwargs for AdaptivePIGCN
    model_kwargs = {
        'feature_dim': 10,
        'hidden_dim': 32,
        'embedding_dim': 16,
        'num_gc_layers': 2,
        'dropout': 0.0,
        'phi': 0.5,
        'num_buses': 33,
        'config': config,
        'normalizer': _normalizer
    }
    
    model = AdaptivePIGCN(**model_kwargs).to(device)
    
    # 4. Create Loss and Optimizer
    print("Initializing PowerSystemLoss...")
    criterion = PowerSystemLoss(config=config, normalizer=_normalizer, is_gcn=False).to(device)
    
    # Check initialization
    sigma_p = torch.exp(criterion.log_sigma_power).item()
    sigma_v = torch.exp(criterion.log_sigma_voltage).item()
    print(f"Initial Sigmas -> Power: {sigma_p:.4f}, Voltage: {sigma_v:.4f}")
    
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=config.LEARNING_RATE)
    
    # 5. Training Loop
    print("\nStarting Training Loop (5 epochs)...")
    model.train()
    
    for epoch in range(config.NUM_EPOCHS):
        total_loss_accum = 0
        steps = 0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            adjacency = batch['adjacency'].to(device)
            # measurements = batch['features'] # Features ARE the measurements
            bus_types = batch['bus_types'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(features, adjacency)
            
            loss = criterion(
                outputs_norm=outputs,
                targets_norm=targets,
                measurements_norm=features,
                ybus_batch=batch['ybus_matrix'].to(device),
                bus_types=bus_types,
                epoch=epoch
            )
            
            # Extract total_loss from returned dictionary
            total_loss = loss['total_loss']
            total_loss.backward()
            optimizer.step()
            
            total_loss_accum += total_loss.item()
            steps += 1
            
            if steps >= 10: # Just run a few steps per epoch for speed
                break
        
        # Check sigmas after epoch
        sigma_p = torch.exp(criterion.log_sigma_power).item()
        sigma_v = torch.exp(criterion.log_sigma_voltage).item()
        weight_v = 1.0 / (2.0 * sigma_v**2)
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Loss: {total_loss_accum/steps:.4f} | "
              f"Sigma_P: {sigma_p:.4f} | Sigma_V: {sigma_v:.4f} (Weight: {weight_v:.4f})")
        
        if weight_v < 0.01:
            print("WARNING: Voltage weight dropped below 0.01! Cheating detected?")
        
    print("\nVerification Complete.")

if __name__ == "__main__":
    verify_fix()
