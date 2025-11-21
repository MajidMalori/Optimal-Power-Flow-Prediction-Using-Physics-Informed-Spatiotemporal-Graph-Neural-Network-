import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from .base_trainer import BaseTrainer
# Removed to_dense_adj import - adjacency is always dense for time-series data
import gc
import warnings

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
        
        # Note: log_file is set in BaseTrainer.train() method, accessible via self.log_file

    def _train_epoch(self, train_loader):
        self.model.train()
        epoch_losses = {'total_loss': 0, 'mse': 0, 'mse_var1': 0, 'mse_var2': 0, 'power_violation': 0, 'voltage_violation': 0}
        
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

            # Automatic Mixed Precision (AMP) context manager
            # PyTorch 2.0+ API: use torch.amp instead of torch.cuda.amp
            device_type = 'cuda' if self.use_cuda else 'cpu'
            with torch.amp.autocast(device_type=device_type):
                # Forward pass - pass bus_types if model supports it
                if bus_types is not None:
                    try:
                        outputs = self.model(features, adjacency_input, bus_types=bus_types)
                    except TypeError:
                        # Model doesn't support bus_types parameter, use default forward
                        outputs = self.model(features, adjacency_input)
                else:
                    outputs = self.model(features, adjacency_input)
                
                loss_dict = self.criterion(outputs, targets, features, ybus, bus_types=bus_types, epoch=self.current_epoch)
                total_loss = loss_dict['total_loss']
                
                if hasattr(self, 'eb_optimizer') and self.eb_optimizer is not None:
                    total_loss += self.eb_optimizer.get_regularization_loss()
                
                # Normalize the loss for accumulation
                loss = total_loss / self.accumulation_steps

            # Scale loss and backpropagate (if using AMP, otherwise direct backward)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # COLLAPSE DIAGNOSTICS: Check for model collapse (predicting constant values)
            if batch_idx == 0 and self.current_epoch % 5 == 0:  # Check every 5 epochs, first batch only
                with torch.no_grad():
                    # Check variance of predictions (η1 values)
                    eta1_var1 = outputs[..., 0]  # [batch, buses]
                    eta1_var2 = outputs[..., 1]  # [batch, buses]
                    
                    var1_variance = eta1_var1.var().item()
                    var2_variance = eta1_var2.var().item()
                    
                    # Warning threshold: variance < 1e-6 suggests collapse
                    if var1_variance < 1e-6 or var2_variance < 1e-6:
                        model_name = self.model.__class__.__name__
                        warning_msg = (
                            f"[COLLAPSE WARNING] {model_name} may be collapsing at epoch {self.current_epoch}. "
                            f"Prediction variance: var1={var1_variance:.2e}, var2={var2_variance:.2e}. "
                            f"Model may be predicting constant values (mean collapse)."
                        )
                        # Log to file instead of terminal
                        if hasattr(self, 'log_file') and self.log_file:
                            try:
                                self.log_file.write(f"WARNING: {warning_msg}\n")
                                self.log_file.flush()
                            except AttributeError:
                                pass
                        
                        # Adaptive learning rate reduction: reduce LR by 50% if collapse detected
                        if not hasattr(self, '_collapse_lr_reduced'):
                            self._collapse_lr_reduced = False
                        
                        if not self._collapse_lr_reduced and self.current_epoch > 5:
                            current_lr = self.optimizer.param_groups[0]['lr']
                            new_lr = current_lr * 0.5
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = new_lr
                            self._collapse_lr_reduced = True
                            recovery_msg = (
                                f"[COLLAPSE RECOVERY] Reduced learning rate from {current_lr:.6f} to {new_lr:.6f} "
                                f"to help model recover from collapse."
                            )
                            if hasattr(self, 'log_file') and self.log_file:
                                try:
                                    self.log_file.write(f"INFO: {recovery_msg}\n")
                                    self.log_file.flush()
                                except AttributeError:
                                    pass

            # Gradient accumulation step
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.scaler is not None:
                    # AMP mode: unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
                    torch.nn.utils.clip_grad_norm_(self.criterion.parameters(), self.config.MAX_GRAD_NORM)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # CPU mode: clip gradients directly
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
                    torch.nn.utils.clip_grad_norm_(self.criterion.parameters(), self.config.MAX_GRAD_NORM)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # --- Good Hygiene: Update losses and delete tensors ---
            # Extract scalar values for logging
            for key in epoch_losses.keys():
                if key in loss_dict:
                    value = loss_dict[key]
                    epoch_losses[key] += (value.item() if isinstance(value, torch.Tensor) else value) / self.accumulation_steps

            # Update progress bar with running average
            # epoch_losses is already divided by accumulation_steps, so just divide by batch count
            current_batch_count = batch_idx + 1
            if self.is_physics_informed:
                pbar.set_postfix(OrderedDict([
                    ('mse_phys', f"{epoch_losses['mse'] / current_batch_count:.6f}"),
                    ('p_vio_rmse', f"{epoch_losses['power_violation'] / current_batch_count:.6f}"),
                    ('v_vio_rmse', f"{epoch_losses['voltage_violation'] / current_batch_count:.6f}")
                ]))
            else:
                pbar.set_postfix(mse_phys=f"{epoch_losses['mse'] / current_batch_count:.6f}")
            
            # Explicitly delete tensors to free memory
            del features, targets, ybus, adjacency_input, bus_types, outputs, loss_dict, total_loss, loss

        num_batches = len(train_loader)
        return {key: val / num_batches * self.accumulation_steps for key, val in epoch_losses.items()}

    def _val_epoch(self, val_loader):
        self.model.eval()
        epoch_losses = {'total_loss': 0, 'mse': 0, 'mse_var1': 0, 'mse_var2': 0, 'power_violation': 0, 'voltage_violation': 0}
        
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

                # PyTorch 2.0+ API: use torch.amp instead of torch.cuda.amp
                device_type = 'cuda' if self.use_cuda else 'cpu'
                with torch.amp.autocast(device_type=device_type):
                    outputs = self.model(features, adjacency_input)
                    loss_dict = self.criterion(outputs, targets, features, ybus, bus_types=bus_types, epoch=self.current_epoch)
                
                for key in epoch_losses.keys():
                    if key in loss_dict:
                        value = loss_dict[key]
                        epoch_losses[key] += value.item() if isinstance(value, torch.Tensor) else value

                if self.is_physics_informed:
                    pbar.set_postfix(OrderedDict([
                        ('mse_phys', f"{epoch_losses['mse']/(batch_idx+1):.6f}"),
                        ('p_vio_rmse', f"{epoch_losses['power_violation']/(batch_idx+1):.6f}"),
                        ('v_vio_rmse', f"{epoch_losses['voltage_violation']/(batch_idx+1):.6f}")
                    ]))
                else:
                    pbar.set_postfix(mse_phys=f"{epoch_losses['mse']/(batch_idx+1):.6f}")

        num_batches = len(val_loader)
        return {key: val / num_batches for key, val in epoch_losses.items()}