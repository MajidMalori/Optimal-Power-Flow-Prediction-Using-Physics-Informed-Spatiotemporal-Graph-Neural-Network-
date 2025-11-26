import torch
import numpy as np
import gc
import warnings
from tqdm import tqdm
from collections import OrderedDict
from .base_trainer import BaseTrainer
from utils.forensic_logger import get_logger
from utils.empirical_bayes import EmpiricalBayesOptimizer

class PowerSystemTrainer(BaseTrainer):
    """
    Specific trainer for power system models. Implements the logic for a single
    training and validation epoch.
    
    FIXED:
    - Removed `torch.cuda.empty_cache()` from the inner training loop. It's a slow,
      synchronizing operation and is only a crutch for bad memory management. The new
      lazy data loading and AMP make it unnecessary and harmful to performance.
    - Kept `del` statements for good memory hygiene.
    - Added Automatic Mixed Precision (AMP) for GPU training.
    - Implemented proper gradient accumulation (not hardcoded to 1).
    """
    def __init__(self, model, criterion, optimizer, config, device, is_physics_informed=True):
        # The __init__ from BaseTrainer is called, which sets up self.current_epoch
        super().__init__(model, criterion, optimizer, config, device)
        self.is_physics_informed = is_physics_informed

        # Forensic logger (will be None if not enabled)
        self.forensic_logger = get_logger()
        
        # PERFORMANCE: Cache GPU availability check (checked once at init, not in every loop)
        self.use_cuda = device.type == 'cuda'
        
        # Gradient accumulation is now handled within the training loop logic
        self.accumulation_steps = getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1)
        
        # Initialize the GradScaler for Automatic Mixed Precision (AMP)
        # PyTorch 2.0+ API: use torch.amp instead of torch.cuda.amp
        if self.use_cuda:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None  # No scaler needed for CPU
        
        # Initialize Empirical Bayes Optimizer (Immer et al., NeurIPS 2023)
        eb_burn_in = getattr(config, 'EB_BURN_IN_EPOCHS', 100)
        eb_update_freq = getattr(config, 'EB_UPDATE_FREQUENCY', 50)
        eb_hyperparam_steps = getattr(config, 'EB_HYPERPARAMETER_STEPS', 50)
        eb_hyperparam_lr = getattr(config, 'EB_HYPERPARAMETER_LR', 0.01)
        
        self.eb_optimizer = EmpiricalBayesOptimizer(
            model=model,
            config=config,
            device=device,
            burn_in_epochs=eb_burn_in,
            update_frequency=eb_update_freq,
            hyperparameter_steps=eb_hyperparam_steps,
            hyperparameter_lr=eb_hyperparam_lr
        )
        
        # Note: log_file is set in BaseTrainer.train() method, accessible via self.log_file

    def _train_epoch(self, train_loader):
        self.model.train()
        # FIX: Initialize with keys that actually match your criterion output
        epoch_losses = {'total_loss': 0, 'mse': 0, 'mse_normalized': 0, 'mse_var1': 0, 'mse_var2': 0}
        if self.is_physics_informed:
            epoch_losses['power_violation'] = 0
            epoch_losses['voltage_violation'] = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}/{self.config.NUM_EPOCHS} [Train]")

        # Zero gradients once at the beginning of the epoch
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            features = batch['features'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)
            ybus = batch['ybus_matrix'].to(self.device, non_blocking=True)
            adjacency_input = batch['adjacency'].to(self.device, non_blocking=True)
            bus_types = batch.get('bus_types', None)
            if bus_types is not None:
                bus_types = bus_types.to(self.device, non_blocking=True)

            # Automatic Mixed Precision (AMP)
            device_type = 'cuda' if self.use_cuda else 'cpu'
            with torch.amp.autocast(device_type=device_type):
                if bus_types is not None:
                    try:
                        outputs = self.model(features, adjacency_input, bus_types=bus_types)
                    except TypeError:
                        outputs = self.model(features, adjacency_input)
                else:
                    outputs = self.model(features, adjacency_input)
                
                loss_dict = self.criterion(outputs, targets, features, ybus, bus_types=bus_types, epoch=self.current_epoch)
                
                total_loss = loss_dict['total_loss']
                
                # Add Empirical Bayes regularization
                if hasattr(self, 'eb_optimizer'):
                    eb_loss = self.eb_optimizer.get_regularization_loss()
                    total_loss = total_loss + eb_loss
                
                # Normalize the loss for accumulation
                loss = total_loss / self.accumulation_steps

            # Scale loss and backpropagate
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
                    torch.nn.utils.clip_grad_norm_(self.criterion.parameters(), self.config.MAX_GRAD_NORM)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
                    torch.nn.utils.clip_grad_norm_(self.criterion.parameters(), self.config.MAX_GRAD_NORM)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # Update losses
            for key in epoch_losses.keys():
                if key in loss_dict:
                    value = loss_dict[key]
                    epoch_losses[key] += (value.item() if isinstance(value, torch.Tensor) else value) / self.accumulation_steps

            # Update progress bar with nMSE (Normalized MSE)
            current_batch_count = batch_idx + 1
            display_mse = epoch_losses['mse_normalized'] # Use normalized MSE for display
            
            postfix_dict = OrderedDict([
                ('nMSE', f"{display_mse / current_batch_count:.4f}")
            ])
            
            if self.is_physics_informed:
                postfix_dict['p_vio'] = f"{epoch_losses['power_violation'] / current_batch_count:.4f}"
                postfix_dict['v_vio'] = f"{epoch_losses['voltage_violation'] / current_batch_count:.4f}"
                
            pbar.set_postfix(postfix_dict)
            
            # Clean up
            del features, targets, ybus, adjacency_input, bus_types, outputs, loss_dict, total_loss, loss
        
        # Update EB hyperparameters
        if hasattr(self, 'eb_optimizer'):
            self.eb_optimizer.update_hyperparameters(train_loader, self.criterion, self.current_epoch)

        num_batches = len(train_loader)
        return {key: val / num_batches * self.accumulation_steps for key, val in epoch_losses.items()}

    def _val_epoch(self, val_loader):
        self.model.eval()
        epoch_losses = {'total_loss': 0, 'mse': 0, 'mse_normalized': 0, 'mse_var1': 0, 'mse_var2': 0}
        if self.is_physics_informed:
            epoch_losses['power_violation'] = 0
            epoch_losses['voltage_violation'] = 0
        
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch}/{self.config.NUM_EPOCHS} [Val]")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                features = batch['features'].to(self.device, non_blocking=True)
                targets = batch['targets'].to(self.device, non_blocking=True)
                ybus = batch['ybus_matrix'].to(self.device, non_blocking=True)
                adjacency_input = batch['adjacency'].to(self.device, non_blocking=True)
                bus_types = batch.get('bus_types', None)
                if bus_types is not None:
                    bus_types = bus_types.to(self.device, non_blocking=True)

                device_type = 'cuda' if self.use_cuda else 'cpu'
                with torch.amp.autocast(device_type=device_type):
                    outputs = self.model(features, adjacency_input)
                    loss_dict = self.criterion(outputs, targets, features, ybus, bus_types=bus_types, epoch=self.current_epoch)
                
                for key in epoch_losses.keys():
                    if key in loss_dict:
                        value = loss_dict[key]
                        epoch_losses[key] += value.item() if isinstance(value, torch.Tensor) else value

                # Display nMSE
                mse_norm = epoch_losses['mse_normalized'] / (batch_idx + 1)
                
                postfix_dict = OrderedDict([
                    ('nMSE', f"{mse_norm:.4f}")
                ])
                
                if self.is_physics_informed:
                    postfix_dict['p_vio'] = f"{epoch_losses['power_violation']/(batch_idx+1):.4f}"
                    postfix_dict['v_vio'] = f"{epoch_losses['voltage_violation']/(batch_idx+1):.4f}"
                    
                pbar.set_postfix(postfix_dict)

        num_batches = len(val_loader)
        return {key: val / num_batches for key, val in epoch_losses.items()}