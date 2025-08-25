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
    def __init__(self, model, criterion, optimizer, config, device):
        # The __init__ from BaseTrainer is called, which sets up self.current_epoch
        super().__init__(model, criterion, optimizer, config, device)

    def _train_epoch(self, train_loader):
        self.model.train()
        # --- START CORRECTION: Initialize trackers for each loss component ---
        epoch_losses = {'total_loss': 0, 'mse': 0, 'power_violation': 0, 'voltage_violation': 0}
        # --- END CORRECTION ---
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}/{self.config.NUM_EPOCHS} [Train]")

        for batch in pbar:
            self.optimizer.zero_grad()
            
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device)
            ybus = batch['ybus_matrix'].to(self.device)
            time_carbon = batch['time_carbon_coeffs'].to(self.device)
            time_energy = batch['time_energy_coeffs'].to(self.device)
            
            adjacency_batch = batch['adjacency']
            if isinstance(adjacency_batch, list):
                dense_adj_list = [to_dense_adj(adj.to(self.device), max_num_nodes=self.config.NUM_BUSES).squeeze(0) 
                                  for adj in adjacency_batch]
                adjacency_input = torch.stack(dense_adj_list, dim=0)
            else:
                adjacency_input = adjacency_batch.to(self.device)

            outputs = self.model(features, adjacency_input)
            
            loss_dict = self.criterion(outputs, targets, ybus, time_carbon, time_energy)
            total_loss = loss_dict['total_loss']

            total_loss.backward()
            self.optimizer.step()
            
           # Update running totals for the epoch
            epoch_losses['total_loss'] += total_loss.item()
            epoch_losses['mse'] += loss_dict['mse'].item()
            epoch_losses['power_violation'] += loss_dict['power_violation'].item()
            epoch_losses['voltage_violation'] += loss_dict['voltage_violation'].item()

            # Update the progress bar with detailed metrics for the current batch
            pbar.set_postfix(
                mse=f"{loss_dict['mse'].item():.4f}", 
                p_viol=f"{loss_dict['power_violation'].item():.4f}", 
                v_viol=f"{loss_dict['voltage_violation'].item():.4f}"
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
                time_carbon = batch['time_carbon_coeffs'].to(self.device)
                time_energy = batch['time_energy_coeffs'].to(self.device)
                
                adjacency_batch = batch['adjacency']
                if isinstance(adjacency_batch, list):
                    dense_adj_list = [to_dense_adj(adj.to(self.device), max_num_nodes=self.config.NUM_BUSES).squeeze(0)
                                      for adj in adjacency_batch]
                    adjacency_input = torch.stack(dense_adj_list, dim=0)
                else:
                    adjacency_input = adjacency_batch.to(self.device)

                outputs = self.model(features, adjacency_input)
                
                # --- START CORRECTION: Process the dictionary of losses ---
                loss_dict = self.criterion(outputs, targets, ybus, time_carbon, time_energy)
                
                # Update running totals for the epoch
                epoch_losses['total_loss'] += loss_dict['total_loss'].item()
                epoch_losses['mse'] += loss_dict['mse'].item()
                epoch_losses['power_violation'] += loss_dict['power_violation'].item()
                epoch_losses['voltage_violation'] += loss_dict['voltage_violation'].item()

                # Update the progress bar with detailed metrics for the current batch
                pbar.set_postfix(
                    mse=f"{loss_dict['mse'].item():.4f}", 
                    p_viol=f"{loss_dict['power_violation'].item():.4f}", 
                    v_viol=f"{loss_dict['voltage_violation'].item():.4f}"
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