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
        total_loss = 0
        # The f-string here will now work because self.current_epoch is set by the BaseTrainer's train() method
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
            
            loss = self.criterion(outputs, targets, ybus, time_carbon, time_energy)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(train_loss=loss.item())

        return {'loss': total_loss / len(train_loader)}

    def _val_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        # The f-string here will also work correctly
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
                
                loss = self.criterion(outputs, targets, ybus, time_carbon, time_energy)
                total_loss += loss.item()
                pbar.set_postfix(val_loss=loss.item())
        
        return {'loss': total_loss / len(val_loader)}