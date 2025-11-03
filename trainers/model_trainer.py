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
        epoch_losses = {'total_loss': 0, 'mse': 0, 'power_violation': 0, 'voltage_violation': 0}
        # --- END CORRECTION ---
        
        # Track adaptive lambdas for monitoring
        adaptive_lambdas_p = []
        adaptive_lambdas_v = []
        
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
                    outputs = self.model(features, adjacency_input)
                    loss_dict = self.criterion(outputs, targets, ybus)
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
                outputs = self.model(features, adjacency_input)
                loss_dict = self.criterion(outputs, targets, ybus)
                total_loss = loss_dict['total_loss'] / accumulation_steps  # Scale loss for accumulation

                total_loss.backward()
                
                # Only step optimizer after accumulating gradients
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update running totals for the epoch (use unscaled loss for reporting)
            epoch_losses['total_loss'] += loss_dict['total_loss'].item()
            epoch_losses['mse'] += loss_dict['mse'].item()
            epoch_losses['power_violation'] += loss_dict['power_violation'].item()
            epoch_losses['voltage_violation'] += loss_dict['voltage_violation'].item()
            
            # Track adaptive lambdas if available
            if hasattr(self.criterion, '_adaptive_lambda_p'):
                adaptive_lambdas_p.append(self.criterion._adaptive_lambda_p)
            if hasattr(self.criterion, '_adaptive_lambda_v'):
                adaptive_lambdas_v.append(self.criterion._adaptive_lambda_v)

            # Update progress bar with running averages (7 decimal places for small values)
            if self.is_physics_informed:
                # Calculate current average lambda for display (if available)
                current_lambda_p = adaptive_lambdas_p[-1] if adaptive_lambdas_p else self.criterion.lambda_p
                current_lambda_v = adaptive_lambdas_v[-1] if adaptive_lambdas_v else self.criterion.lambda_v
                
                # Display WEIGHTED violations so math adds up: total = mse + weighted_p + weighted_v
                avg_p_viol = epoch_losses['power_violation']/(batch_idx+1)
                avg_v_viol = epoch_losses['voltage_violation']/(batch_idx+1)
                weighted_p = current_lambda_p * avg_p_viol
                weighted_v = current_lambda_v * avg_v_viol
                
                # Use OrderedDict to ensure display order: total, mse, weighted_p, weighted_v
                pbar.set_postfix(OrderedDict([
                    ('total', f"{epoch_losses['total_loss']/(batch_idx+1):.7f}"),
                    ('mse', f"{epoch_losses['mse']/(batch_idx+1):.7f}"),
                    ('λp×Pviol', f"{weighted_p:.7f}"),
                    ('λv×Vviol', f"{weighted_v:.7f}")
                ]))
            else:
                pbar.set_postfix(mse=f"{epoch_losses['mse']/(batch_idx+1):.7f}")
            
            # Periodic memory cleanup for large systems (reduced frequency)
            if batch_idx % 100 == 0 and self.config.NUM_BUSES >= 57:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # --- END CORRECTION ---

        # --- START CORRECTION: Return the average of all loss components ---
        num_batches = len(train_loader)
        
        # Calculate average lambdas for this epoch (to be printed later)
        avg_lambda_p = np.mean(adaptive_lambdas_p) if adaptive_lambdas_p else None
        avg_lambda_v = np.mean(adaptive_lambdas_v) if adaptive_lambdas_v else None
        
        return {
            'loss': epoch_losses['total_loss'] / num_batches,
            'mse': epoch_losses['mse'] / num_batches,
            'power_violation': epoch_losses['power_violation'] / num_batches,
            'voltage_violation': epoch_losses['voltage_violation'] / num_batches,
            'adaptive_lambda_p': avg_lambda_p,
            'adaptive_lambda_v': avg_lambda_v
        }
        # --- END CORRECTION ---

    def _val_epoch(self, val_loader):
        self.model.eval()
        # --- START CORRECTION: Initialize trackers for each loss component ---
        epoch_losses = {'total_loss': 0, 'mse': 0, 'power_violation': 0, 'voltage_violation': 0}
        # --- END CORRECTION ---
        
        # Track adaptive lambdas for monitoring
        adaptive_lambdas_p = []
        adaptive_lambdas_v = []
        
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
                
                # --- START CORRECTION: Process the dictionary of losses ---
                loss_dict = self.criterion(outputs, targets, ybus)
                
                # Update running totals for the epoch
                epoch_losses['total_loss'] += loss_dict['total_loss'].item()
                epoch_losses['mse'] += loss_dict['mse'].item()
                epoch_losses['power_violation'] += loss_dict['power_violation'].item()
                epoch_losses['voltage_violation'] += loss_dict['voltage_violation'].item()
                
                # Track adaptive lambdas if available
                if hasattr(self.criterion, '_adaptive_lambda_p'):
                    adaptive_lambdas_p.append(self.criterion._adaptive_lambda_p)
                if hasattr(self.criterion, '_adaptive_lambda_v'):
                    adaptive_lambdas_v.append(self.criterion._adaptive_lambda_v)

                # Update progress bar with running averages (7 decimal places for small values)
                if self.is_physics_informed:
                    # Calculate current average lambda for display (if available)
                    current_lambda_p = adaptive_lambdas_p[-1] if adaptive_lambdas_p else self.criterion.lambda_p
                    current_lambda_v = adaptive_lambdas_v[-1] if adaptive_lambdas_v else self.criterion.lambda_v
                    
                    # Display WEIGHTED violations so math adds up: total = mse + weighted_p + weighted_v
                    avg_p_viol = epoch_losses['power_violation']/(batch_idx+1)
                    avg_v_viol = epoch_losses['voltage_violation']/(batch_idx+1)
                    weighted_p = current_lambda_p * avg_p_viol
                    weighted_v = current_lambda_v * avg_v_viol
                    
                    # Use OrderedDict to ensure display order: total, mse, weighted_p, weighted_v
                    pbar.set_postfix(OrderedDict([
                        ('total', f"{epoch_losses['total_loss']/(batch_idx+1):.7f}"),
                        ('mse', f"{epoch_losses['mse']/(batch_idx+1):.7f}"),
                        ('λp×Pviol', f"{weighted_p:.7f}"),
                        ('λv×Vviol', f"{weighted_v:.7f}")
                    ]))
                else:
                    pbar.set_postfix(mse=f"{epoch_losses['mse']/(batch_idx+1):.7f}")
        
        # --- START CORRECTION: Return the average of all loss components ---
        num_batches = len(val_loader)
        
        # Calculate average lambdas for this epoch (to be printed later)
        avg_lambda_p = np.mean(adaptive_lambdas_p) if adaptive_lambdas_p else None
        avg_lambda_v = np.mean(adaptive_lambdas_v) if adaptive_lambdas_v else None
        
        return {
            'loss': epoch_losses['total_loss'] / num_batches,
            'mse': epoch_losses['mse'] / num_batches,
            'power_violation': epoch_losses['power_violation'] / num_batches,
            'voltage_violation': epoch_losses['voltage_violation'] / num_batches,
            'adaptive_lambda_p': avg_lambda_p,
            'adaptive_lambda_v': avg_lambda_v
        }
        # --- END CORRECTION ---