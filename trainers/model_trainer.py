# In trainers/model_trainer.py

import torch
from tqdm import tqdm
from .base_trainer import BaseTrainer
from torch_geometric.utils import to_dense_adj

class PowerSystemTrainer(BaseTrainer):
    """
    Specific trainer for power system models. Implements the logic for a single
    training and validation epoch.
    """
    def __init__(self, model, criterion, optimizer, config, device, is_physics_informed=True):
        # The __init__ from BaseTrainer is called, which sets up self.current_epoch
        super().__init__(model, criterion, optimizer, config, device)
        self.is_physics_informed = is_physics_informed
    
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
        
        # Calculate gradient accumulation steps based on system size
        accumulation_steps = self._get_gradient_accumulation_steps()
        effective_batch_size = self.config.BATCH_SIZE * accumulation_steps
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}/{self.config.NUM_EPOCHS} [Train] (eff_batch={effective_batch_size})")

        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device)
            ybus = batch['ybus_matrix'].to(self.device)
            
            adjacency_batch = batch['adjacency']
            if isinstance(adjacency_batch, list):
                dense_adj_list = [to_dense_adj(adj.to(self.device), max_num_nodes=self.config.NUM_BUSES).squeeze(0) 
                                  for adj in adjacency_batch]
                adjacency_input = torch.stack(dense_adj_list, dim=0)
            else:
                adjacency_input = adjacency_batch.to(self.device)

            outputs = self.model(features, adjacency_input)
            
            loss_dict = self.criterion(outputs, targets, ybus)
            total_loss = loss_dict['total_loss'] / accumulation_steps  # Scale loss for accumulation

            total_loss.backward()
            
            # Only step optimizer after accumulating gradients
            if (batch_idx + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
           # Update running totals for the epoch
            epoch_losses['total_loss'] += total_loss.item() * accumulation_steps
            epoch_losses['mse'] += loss_dict['mse'].item()
            epoch_losses['power_violation'] += loss_dict['power_violation'].item()
            epoch_losses['voltage_violation'] += loss_dict['voltage_violation'].item()

            # Update the progress bar with detailed metrics for the current batch
            if self.is_physics_informed:
                pbar.set_postfix(
                    mse=f"{loss_dict['mse'].item():.4f}", 
                    p_viol=f"{loss_dict['power_violation'].item():.4f}", 
                    v_viol=f"{loss_dict['voltage_violation'].item():.4f}",
                    acc_steps=f"{accumulation_steps}"
                )
            else:
                pbar.set_postfix(
                    mse=f"{loss_dict['mse'].item():.4f}",
                    acc_steps=f"{accumulation_steps}"
                )
            # --- END CORRECTION ---

        # --- START CORRECTION: Return the average of all loss components ---
        num_batches = len(train_loader)
        return {
            'loss': epoch_losses['total_loss'] / num_batches,
            'mse': epoch_losses['mse'] / num_batches,
            'power_violation': epoch_losses['power_violation'] / num_batches,
            'voltage_violation': epoch_losses['voltage_violation'] / num_batches
        }
        # --- END CORRECTION ---

    def _val_epoch(self, val_loader):
        self.model.eval()
        # --- START CORRECTION: Initialize trackers for each loss component ---
        epoch_losses = {'total_loss': 0, 'mse': 0, 'power_violation': 0, 'voltage_violation': 0}
        # --- END CORRECTION ---
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch}/{self.config.NUM_EPOCHS} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                features = batch['features'].to(self.device)
                targets = batch['targets'].to(self.device)
                ybus = batch['ybus_matrix'].to(self.device)
                
                adjacency_batch = batch['adjacency']
                if isinstance(adjacency_batch, list):
                    dense_adj_list = [to_dense_adj(adj.to(self.device), max_num_nodes=self.config.NUM_BUSES).squeeze(0)
                                      for adj in adjacency_batch]
                    adjacency_input = torch.stack(dense_adj_list, dim=0)
                else:
                    adjacency_input = adjacency_batch.to(self.device)

                outputs = self.model(features, adjacency_input)
                
                # --- START CORRECTION: Process the dictionary of losses ---
                loss_dict = self.criterion(outputs, targets, ybus)
                
                # Update running totals for the epoch
                epoch_losses['total_loss'] += loss_dict['total_loss'].item()
                epoch_losses['mse'] += loss_dict['mse'].item()
                epoch_losses['power_violation'] += loss_dict['power_violation'].item()
                epoch_losses['voltage_violation'] += loss_dict['voltage_violation'].item()

                # Update the progress bar with detailed metrics for the current batch
                if self.is_physics_informed:
                    pbar.set_postfix(
                        mse=f"{loss_dict['mse'].item():.4f}", 
                        p_viol=f"{loss_dict['power_violation'].item():.4f}", 
                        v_viol=f"{loss_dict['voltage_violation'].item():.4f}"
                    )
                else:
                    pbar.set_postfix(
                        mse=f"{loss_dict['mse'].item():.4f}"
                    )
                # --- END CORRECTION ---
        
        # --- START CORRECTION: Return the average of all loss components ---
        num_batches = len(val_loader)
        return {
            'loss': epoch_losses['total_loss'] / num_batches,
            'mse': epoch_losses['mse'] / num_batches,
            'power_violation': epoch_losses['power_violation'] / num_batches,
            'voltage_violation': epoch_losses['voltage_violation'] / num_batches
        }
        # --- END CORRECTION ---