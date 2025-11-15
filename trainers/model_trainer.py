import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from .base_trainer import BaseTrainer
# Removed to_dense_adj import - adjacency is always dense for time-series data
import gc
class PowerSystemTrainer(BaseTrainer):
    """
    Specific trainer for power system models. Implements the logic for a single
    training and validation epoch.
    """
    def __init__(self, model, criterion, optimizer, config, device, is_physics_informed=True):
        # The __init__ from BaseTrainer is called, which sets up self.current_epoch
        super().__init__(model, criterion, optimizer, config, device)
        self.is_physics_informed = is_physics_informed
        
        # PERFORMANCE: Cache GPU availability check (checked once at init, not in every loop)
        self.use_cuda = device.type == 'cuda'
    
    def _get_gradient_accumulation_steps(self):
        """Calculate gradient accumulation steps (always returns 1 - accumulation disabled)"""
        return 1  # Gradient accumulation is disabled

    def _train_epoch(self, train_loader):
        self.model.train()
        epoch_losses = {'total_loss': 0, 'mse': 0, 'mse_var1': 0, 'mse_var2': 0, 'power_violation': 0, 'voltage_violation': 0}
        
        accumulation_steps = self._get_gradient_accumulation_steps()
        effective_batch_size = self.config.BATCH_SIZE * accumulation_steps
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}/{self.config.NUM_EPOCHS} [Train]")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device with memory efficiency
            features = batch['features'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)
            ybus = batch['ybus_matrix'].to(self.device, non_blocking=True)
            
            # Adjacency is always dense for time-series data - direct device transfer
            adjacency_input = batch['adjacency'].to(self.device, non_blocking=True)

            bus_types = batch.get('bus_types', None)
            
            # Forward pass - pass bus_types if model supports it (for generator constraints)
            if bus_types is not None:
                bus_types_device = bus_types.to(self.device, non_blocking=True)
                try:
                    outputs = self.model(features, adjacency_input, bus_types=bus_types_device)
                except TypeError:
                    # Model doesn't support bus_types parameter, use default forward
                    outputs = self.model(features, adjacency_input)
            else:
                outputs = self.model(features, adjacency_input)  # [batch, buses, 4]
            
            # Clear batch dict immediately (data already on device)
            del batch
            
            loss_dict = self.criterion(
                outputs,
                targets,
                features,
                ybus,
                bus_types=bus_types,
                return_components=False,
                epoch=self.current_epoch
            )
            
            # Add Empirical Bayes regularization if enabled (from paper Section 4.3)
            total_loss = loss_dict['total_loss']
            if hasattr(self, 'eb_optimizer') and self.eb_optimizer is not None:
                eb_reg = self.eb_optimizer.get_regularization_loss()
                total_loss = total_loss + eb_reg
            
            total_loss = total_loss / accumulation_steps
            total_loss.backward()
            
            # Gradient clipping to prevent explosion (critical for natural parametrization)
            # Root cause: Large f2 → large exp(f2) → large gradients → larger f2 (feedback loop)
            max_grad_norm = getattr(self.config, 'MAX_GRAD_NORM', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.criterion.parameters(), max_grad_norm)
            
            # Extract values BEFORE deleting tensors - optimized .item() calls
            total_loss_val = loss_dict['total_loss'].item()
            total_loss_denorm_val = loss_dict.get('total_loss_denorm', loss_dict['total_loss']).item()
            mse_val = loss_dict['mse'].item()
            power_viol_val = loss_dict['power_violation'].item()
            voltage_viol_val = loss_dict['voltage_violation'].item()
            # Optimize mse_var extraction - single conditional check
            mse_var1_val = loss_dict.get('mse_var1', 0.0)
            mse_var2_val = loss_dict.get('mse_var2', 0.0)
            if isinstance(mse_var1_val, torch.Tensor):
                mse_var1_val = mse_var1_val.item()
            if isinstance(mse_var2_val, torch.Tensor):
                mse_var2_val = mse_var2_val.item()
            
            # MEMORY CLEARING: Delete intermediate tensors after use
            # SAFETY ANALYSIS:
            # - outputs, loss_dict, total_loss: Safe to delete AFTER backward() because:
            #   * PyTorch autograd stores computation graph in parameter.grad attributes
            #   * The backward() call has already computed gradients, inputs no longer needed
            # - features, targets, ybus, adjacency_input: Safe to delete AFTER backward() because:
            #   * Gradients are stored in parameter.grad, not in input tensors
            #   * We delete AFTER optimizer.step() to be extra safe (though not strictly necessary)
            # - For gradient accumulation: We keep inputs until optimizer.step() for safety,
            #   but could delete after backward() if memory is critical (current approach is safer)
            del outputs, loss_dict, total_loss
            
            # Only step optimizer after accumulating gradients
            if (batch_idx + 1) % accumulation_steps == 0:
                self.optimizer.step()
                
                # Step scheduler after each optimizer step (for per-batch schedulers)
                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    # Most schedulers step per-epoch, but some can step per-batch if needed
                    pass  # Scheduler stepping handled per-epoch in base_trainer
                
                self.optimizer.zero_grad()
                # Clear input tensors after optimizer step (safe: gradients already computed and applied)
                del features, targets, ybus, adjacency_input
                if bus_types is not None:
                    del bus_types
            # NOTE: For gradient accumulation, we keep inputs until optimizer.step().
            # This is safe but uses slightly more memory. Could delete after backward() if needed.
            
            # Update running totals (using extracted Python values, not tensors)
            epoch_losses['total_loss'] += total_loss_val  # Normalized (for optimization)
            epoch_losses['total_loss_denorm'] = epoch_losses.get('total_loss_denorm', 0.0) + total_loss_denorm_val  # Denormalized (for display)
            epoch_losses['mse'] += mse_val
            epoch_losses['mse_var1'] += mse_var1_val
            epoch_losses['mse_var2'] += mse_var2_val
            epoch_losses['power_violation'] += power_viol_val
            epoch_losses['voltage_violation'] += voltage_viol_val

            if self.is_physics_informed:
                avg_mse = epoch_losses['mse']/(batch_idx+1)
                pbar.set_postfix(OrderedDict([
                    ('mse_phys', f"{avg_mse:.6f}"),
                    ('p_vio_rmse', f"{epoch_losses['power_violation']/(batch_idx+1):.6f}"),
                    ('v_vio_rmse', f"{epoch_losses['voltage_violation']/(batch_idx+1):.6f}")
                ]))
            else:
                # For non-physics models (GCN), only show MSE (total_loss = MSE for non-physics)
                pbar.set_postfix(mse_phys=f"{epoch_losses['mse']/(batch_idx+1):.6f}")  # Denormalized (physical units)
            
            # AGGRESSIVE MEMORY CLEARING: More frequent for larger systems
            # Clear cache every N batches based on system size to prevent memory accumulation
            # PERFORMANCE: Only clear if using CUDA (checked once at init, not every batch)
            clear_frequency = 50 if self.config.NUM_BUSES >= 118 else (100 if self.config.NUM_BUSES >= 57 else 200)
            if batch_idx % clear_frequency == 0 and batch_idx > 0:  # Skip first batch (idx=0)
                gc.collect()
                if self.use_cuda:  # Use cached value, not repeated torch.cuda.is_available() call
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Ensure all operations complete before clearing

        # Return the average of all loss components
        num_batches = len(train_loader)
        
        return {
            'loss': epoch_losses['total_loss'] / num_batches,
            'mse': epoch_losses['mse'] / num_batches,
            'mse_var1': epoch_losses['mse_var1'] / num_batches,
            'mse_var2': epoch_losses['mse_var2'] / num_batches,
            'power_violation': epoch_losses['power_violation'] / num_batches,
            'voltage_violation': epoch_losses['voltage_violation'] / num_batches
        }

    def _val_epoch(self, val_loader):
        self.model.eval()
        epoch_losses = {'total_loss': 0, 'mse': 0, 'mse_var1': 0, 'mse_var2': 0, 'power_violation': 0, 'voltage_violation': 0}
        
        
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch}/{self.config.NUM_EPOCHS} [Val]")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                features = batch['features'].to(self.device)
                targets = batch['targets'].to(self.device)
                ybus = batch['ybus_matrix'].to(self.device)
                
                # Optimized adjacency matrix handling (validation)
                # Adjacency is always dense for time-series data - direct device transfer
                adjacency_input = batch['adjacency'].to(self.device, non_blocking=True)

                # Forward pass
                outputs = self.model(features, adjacency_input)
                
                
                bus_types = batch.get('bus_types', None)
                
                loss_dict = self.criterion(
                    outputs,
                    targets,
                    features,
                    ybus,
                    bus_types=bus_types,
                    return_components=False,
                    epoch=self.current_epoch
                )
                
                epoch_losses['total_loss'] += loss_dict['total_loss'].item()  # Normalized (for optimization)
                total_loss_denorm_val = loss_dict.get('total_loss_denorm', loss_dict['total_loss']).item()
                epoch_losses['total_loss_denorm'] = epoch_losses.get('total_loss_denorm', 0.0) + total_loss_denorm_val  # Denormalized (for display)
                epoch_losses['mse'] += loss_dict['mse'].item()
                if 'mse_weighted' in loss_dict and loss_dict['mse_weighted'] is not None:
                    epoch_losses['mse_weighted'] = epoch_losses.get('mse_weighted', 0.0) + loss_dict['mse_weighted'].item()
                epoch_losses['mse_var1'] += loss_dict.get('mse_var1', 0.0) if isinstance(loss_dict.get('mse_var1', 0.0), float) else loss_dict.get('mse_var1', torch.tensor(0.0)).item()
                epoch_losses['mse_var2'] += loss_dict.get('mse_var2', 0.0) if isinstance(loss_dict.get('mse_var2', 0.0), float) else loss_dict.get('mse_var2', torch.tensor(0.0)).item()
                epoch_losses['power_violation'] += loss_dict['power_violation'].item()
                epoch_losses['voltage_violation'] += loss_dict['voltage_violation'].item()
                
                if self.is_physics_informed:
                    avg_mse = epoch_losses['mse']/(batch_idx+1)
                    pbar.set_postfix(OrderedDict([
                        ('mse_phys', f"{avg_mse:.6f}"),
                        ('p_vio_rmse', f"{epoch_losses['power_violation']/(batch_idx+1):.6f}"),
                        ('v_vio_rmse', f"{epoch_losses['voltage_violation']/(batch_idx+1):.6f}")
                    ]))
                else:
                    # For non-physics models (GCN), only show MSE (total_loss = MSE for non-physics)
                    pbar.set_postfix(mse_phys=f"{epoch_losses['mse']/(batch_idx+1):.6f}")  # Denormalized (physical units)
        
        # Return the average of all loss components
        num_batches = len(val_loader)
        
        return {
            'loss': epoch_losses['total_loss'] / num_batches,
            'mse': epoch_losses['mse'] / num_batches,
            'mse_var1': epoch_losses['mse_var1'] / num_batches,
            'mse_var2': epoch_losses['mse_var2'] / num_batches,
            'power_violation': epoch_losses['power_violation'] / num_batches,
            'voltage_violation': epoch_losses['voltage_violation'] / num_batches
        }