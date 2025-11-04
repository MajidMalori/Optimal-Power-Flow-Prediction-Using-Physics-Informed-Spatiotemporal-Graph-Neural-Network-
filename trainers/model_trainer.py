# In trainers/model_trainer.py

import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from .base_trainer import BaseTrainer
from torch_geometric.utils import to_dense_adj
import gc
from torch.cuda.amp import autocast, GradScaler

class PowerSystemTrainer(BaseTrainer):
    """
    Specific trainer for power system models. Implements the logic for a single
    training and validation epoch.
    """
    def __init__(self, model, criterion, optimizer, config, device, is_physics_informed=True):
        # The __init__ from BaseTrainer is called, which sets up self.current_epoch
        super().__init__(model, criterion, optimizer, config, device)
        self.is_physics_informed = is_physics_informed
        
        # Initialize mixed precision scaler for GPU training (only if enabled in config)
        use_mixed_precision = getattr(config, 'USE_MIXED_PRECISION', True)
        self.scaler = GradScaler() if (torch.cuda.is_available() and use_mixed_precision) else None
    
    def _get_gradient_accumulation_steps(self):
        """Calculate gradient accumulation steps based on system size to reduce memory usage"""
        num_buses = self.config.NUM_BUSES
        if num_buses >= 118:
            return 4  # Large systems: accumulate 4 batches
        elif num_buses >= 57:
            return 2  # Medium systems: accumulate 2 batches
        else:
            return 1  # Small systems: no accumulation needed

    def _train_epoch(self, train_loader):
        self.model.train()
        # --- START CORRECTION: Initialize trackers for each loss component ---
        epoch_losses = {'total_loss': 0, 'mse': 0, 'mse_vm': 0, 'mse_va': 0, 'power_violation': 0, 'voltage_violation': 0}
        # --- END CORRECTION ---
        
        
        # Calculate gradient accumulation steps based on system size
        accumulation_steps = self._get_gradient_accumulation_steps()
        effective_batch_size = self.config.BATCH_SIZE * accumulation_steps
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}/{self.config.NUM_EPOCHS} [Train]")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device with memory efficiency
            features = batch['features'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)
            ybus = batch['ybus_matrix'].to(self.device, non_blocking=True)
            
            # Optimized adjacency matrix handling
            adjacency_batch = batch['adjacency']
            if isinstance(adjacency_batch, list):
                # Pre-allocate tensor for better memory efficiency
                batch_size = len(adjacency_batch)
                adjacency_input = torch.zeros(batch_size, self.config.NUM_BUSES, self.config.NUM_BUSES, 
                                            device=self.device, dtype=torch.float32)
                for i, adj in enumerate(adjacency_batch):
                    dense_adj = to_dense_adj(adj.to(self.device), max_num_nodes=self.config.NUM_BUSES).squeeze(0)
                    adjacency_input[i] = dense_adj
            else:
                adjacency_input = adjacency_batch.to(self.device)

            # Use mixed precision for forward pass
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(features, adjacency_input)  # [batch, buses, 2]
                    
                    # Get bus types from batch (OPF: bus-type-dependent unknowns)
                    bus_types = batch.get('bus_types', None)  # [batch, buses] or None
                    
                    loss_dict = self.criterion(
                        outputs,      # Predicted unknowns [batch, buses, 2] (OPF: bus-type dependent)
                        targets,      # True unknowns [batch, buses, 2] (OPF: bus-type dependent)
                        features,     # Measured power (use as measurements)
                        ybus,
                        bus_types=bus_types  # OPF: bus type codes [0=PQ, 1=PV, 2=Slack]
                    )
                    total_loss = loss_dict['total_loss'] / accumulation_steps  # Scale loss for accumulation

                # Scale loss and backward pass
                self.scaler.scale(total_loss).backward()
                
                # Only step optimizer after accumulating gradients
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # CPU training without mixed precision
                outputs = self.model(features, adjacency_input)  # [batch, buses, 2]
                
                # DEBUG: Check output shape before passing to loss
                if outputs.shape != targets.shape:
                    print(f"[DEBUG] Shape mismatch: outputs={outputs.shape}, targets={targets.shape}")
                    # Try to fix common issues
                    if outputs.dim() == 3 and outputs.shape[-1] != 2:
                        # If output is [batch, buses, wrong_dim], try to reshape
                        batch_size, num_buses = outputs.shape[0], outputs.shape[1]
                        if outputs.numel() == batch_size * num_buses * 2:
                            # Can be reshaped to [batch, buses, 2]
                            outputs = outputs.view(batch_size, num_buses, 2)
                            print(f"[DEBUG] Reshaped outputs to {outputs.shape}")
                        else:
                            print(f"[DEBUG] Cannot reshape: outputs.numel()={outputs.numel()}, expected={batch_size * num_buses * 2}")
                
                # ETH Zurich Technique 4: Separate VM/VA backward passes (if enabled)
                use_separate_backward = getattr(self.config, 'USE_SEPARATE_VM_VA_BACKWARD', False)
                
                # Get bus types from batch (OPF: bus-type-dependent unknowns)
                bus_types = batch.get('bus_types', None)  # [batch, buses] or None
                
                # OPF: Disable separate backward passes (unknowns vary by bus type)
                # For OPF, bus_types is not None, so we disable separate backward
                use_separate_backward = use_separate_backward and (bus_types is None)
                
                loss_dict = self.criterion(
                    outputs,      # Predicted unknowns [batch, buses, 2] (OPF: bus-type dependent)
                    targets,      # True unknowns [batch, buses, 2] (OPF: bus-type dependent)
                    features,     # Measured power (use as measurements)
                    ybus,
                    bus_types=bus_types,  # OPF: bus type codes [0=PQ, 1=PV, 2=Slack]
                    return_components=use_separate_backward  # Request separate components if enabled (disabled for OPF)
                )
                
                if use_separate_backward and 'mse_vm_loss' in loss_dict:
                    # ETH Zurich approach: Separate backward passes for VM, VA, and physics
                    mse_vm_loss = loss_dict['mse_vm_loss'] / accumulation_steps
                    mse_va_loss = loss_dict['mse_va_loss'] / accumulation_steps
                    physics_loss = loss_dict['physics_loss'] / accumulation_steps
                    
                    # Backward pass for VM (retain graph for subsequent backpasses)
                    mse_vm_loss.backward(retain_graph=True)
                    # Backward pass for VA (retain graph for physics)
                    mse_va_loss.backward(retain_graph=True)
                    # Backward pass for physics (no need to retain graph)
                    physics_loss.backward()
                else:
                    # Standard single backward pass
                    total_loss = loss_dict['total_loss'] / accumulation_steps
                    total_loss.backward()
                
                # Only step optimizer after accumulating gradients
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update running totals for the epoch (use unscaled loss for reporting)
            epoch_losses['total_loss'] += loss_dict['total_loss'].item()
            # Track both raw and weighted MSE for clarity
            epoch_losses['mse'] += loss_dict['mse'].item()  # Raw MSE
            if 'mse_weighted' in loss_dict:
                epoch_losses['mse_weighted'] = epoch_losses.get('mse_weighted', 0.0) + loss_dict['mse_weighted'].item()
            epoch_losses['mse_vm'] += loss_dict.get('mse_vm', 0.0) if isinstance(loss_dict.get('mse_vm', 0.0), float) else loss_dict.get('mse_vm', torch.tensor(0.0)).item()
            epoch_losses['mse_va'] += loss_dict.get('mse_va', 0.0) if isinstance(loss_dict.get('mse_va', 0.0), float) else loss_dict.get('mse_va', torch.tensor(0.0)).item()
            epoch_losses['power_violation'] += loss_dict['power_violation'].item()
            epoch_losses['voltage_violation'] += loss_dict['voltage_violation'].item()

            # Update progress bar with running averages
            if self.is_physics_informed:
                # Display: total, mse, p_vio, v_viol (clean, single line)
                # Note: mse is raw MSE (actual prediction error)
                # total uses weighted MSE component, so total can be < raw MSE initially
                # This is normal with learnable uncertainty weighting (Kendall et al.)
                avg_mse = epoch_losses['mse']/(batch_idx+1)
                # Use weighted MSE if available (for consistency with total), otherwise raw MSE
                if 'mse_weighted' in epoch_losses:
                    avg_mse_weighted = epoch_losses['mse_weighted']/(batch_idx+1)
                    pbar.set_postfix(OrderedDict([
                        ('total', f"{epoch_losses['total_loss']/(batch_idx+1):.7f}"),
                        ('mse', f"{avg_mse:.7f}"),  # Raw MSE (actual error)
                        ('p_vio', f"{epoch_losses['power_violation']/(batch_idx+1):.7f}"),
                        ('v_viol', f"{epoch_losses['voltage_violation']/(batch_idx+1):.7f}")
                    ]))
                else:
                    pbar.set_postfix(OrderedDict([
                        ('total', f"{epoch_losses['total_loss']/(batch_idx+1):.7f}"),
                        ('mse', f"{avg_mse:.7f}"),
                        ('p_vio', f"{epoch_losses['power_violation']/(batch_idx+1):.7f}"),
                        ('v_viol', f"{epoch_losses['voltage_violation']/(batch_idx+1):.7f}")
                    ]))
            else:
                pbar.set_postfix(mse=f"{epoch_losses['mse']/(batch_idx+1):.7f}")
            
            # Periodic memory cleanup for large systems (reduced frequency)
            if batch_idx % 100 == 0 and self.config.NUM_BUSES >= 57:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # --- END CORRECTION ---

        # Return the average of all loss components
        num_batches = len(train_loader)
        
        return {
            'loss': epoch_losses['total_loss'] / num_batches,
            'mse': epoch_losses['mse'] / num_batches,
            'mse_vm': epoch_losses['mse_vm'] / num_batches,  # ETH Zurich: Separate VM loss
            'mse_va': epoch_losses['mse_va'] / num_batches,  # ETH Zurich: Separate VA loss
            'power_violation': epoch_losses['power_violation'] / num_batches,
            'voltage_violation': epoch_losses['voltage_violation'] / num_batches
        }

    def _val_epoch(self, val_loader):
        self.model.eval()
        # --- START CORRECTION: Initialize trackers for each loss component ---
        epoch_losses = {'total_loss': 0, 'mse': 0, 'mse_vm': 0, 'mse_va': 0, 'power_violation': 0, 'voltage_violation': 0}
        # --- END CORRECTION ---
        
        
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch}/{self.config.NUM_EPOCHS} [Val]")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                features = batch['features'].to(self.device)
                targets = batch['targets'].to(self.device)
                ybus = batch['ybus_matrix'].to(self.device)
                
                # Optimized adjacency matrix handling (validation)
                adjacency_batch = batch['adjacency']
                if isinstance(adjacency_batch, list):
                    # Pre-allocate tensor for better memory efficiency
                    batch_size = len(adjacency_batch)
                    adjacency_input = torch.zeros(batch_size, self.config.NUM_BUSES, self.config.NUM_BUSES, 
                                                device=self.device, dtype=torch.float32)
                    for i, adj in enumerate(adjacency_batch):
                        dense_adj = to_dense_adj(adj.to(self.device), max_num_nodes=self.config.NUM_BUSES).squeeze(0)
                        adjacency_input[i] = dense_adj
                else:
                    adjacency_input = adjacency_batch.to(self.device)

                # Use mixed precision for validation forward pass
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(features, adjacency_input)
                else:
                    outputs = self.model(features, adjacency_input)
                
                
                # Get bus types from batch (OPF: bus-type-dependent unknowns)
                bus_types = batch.get('bus_types', None)  # [batch, buses] or None
                
                # --- START CORRECTION: Process the dictionary of losses ---
                loss_dict = self.criterion(
                    outputs,      # Predicted unknowns [batch, buses, 2] (OPF: bus-type dependent)
                    targets,      # True unknowns [batch, buses, 2] (OPF: bus-type dependent)
                    features,     # Measured power (use as measurements)
                    ybus,
                    bus_types=bus_types  # OPF: bus type codes [0=PQ, 1=PV, 2=Slack]
                )
                
                # Update running totals for the epoch
                epoch_losses['total_loss'] += loss_dict['total_loss'].item()
                epoch_losses['mse'] += loss_dict['mse'].item()  # Raw MSE
                if 'mse_weighted' in loss_dict:
                    epoch_losses['mse_weighted'] = epoch_losses.get('mse_weighted', 0.0) + loss_dict['mse_weighted'].item()
                epoch_losses['mse_vm'] += loss_dict.get('mse_vm', 0.0) if isinstance(loss_dict.get('mse_vm', 0.0), float) else loss_dict.get('mse_vm', torch.tensor(0.0)).item()
                epoch_losses['mse_va'] += loss_dict.get('mse_va', 0.0) if isinstance(loss_dict.get('mse_va', 0.0), float) else loss_dict.get('mse_va', torch.tensor(0.0)).item()
                epoch_losses['power_violation'] += loss_dict['power_violation'].item()
                epoch_losses['voltage_violation'] += loss_dict['voltage_violation'].item()
                
                # Update progress bar with running averages
                if self.is_physics_informed:
                    # Display: total, mse (raw), p_vio, v_viol (clean, single line)
                    # Note: total uses weighted MSE (learnable uncertainty), so total can be < raw MSE
                    # This is normal - total = weighted_MSE + weighted_power + weighted_voltage + regularization
                    avg_mse = epoch_losses['mse']/(batch_idx+1)
                    pbar.set_postfix(OrderedDict([
                        ('total', f"{epoch_losses['total_loss']/(batch_idx+1):.7f}"),
                        ('mse', f"{avg_mse:.7f}"),  # Raw MSE (actual prediction error)
                        ('p_vio', f"{epoch_losses['power_violation']/(batch_idx+1):.7f}"),
                        ('v_viol', f"{epoch_losses['voltage_violation']/(batch_idx+1):.7f}")
                    ]))
                else:
                    pbar.set_postfix(mse=f"{epoch_losses['mse']/(batch_idx+1):.7f}")
        
        # Return the average of all loss components
        num_batches = len(val_loader)
        
        return {
            'loss': epoch_losses['total_loss'] / num_batches,
            'mse': epoch_losses['mse'] / num_batches,
            'mse_vm': epoch_losses['mse_vm'] / num_batches,  # ETH Zurich: Separate VM loss
            'mse_va': epoch_losses['mse_va'] / num_batches,  # ETH Zurich: Separate VA loss
            'power_violation': epoch_losses['power_violation'] / num_batches,
            'voltage_violation': epoch_losses['voltage_violation'] / num_batches
        }