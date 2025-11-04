"""
Ensemble Methods for Uncertainty Quantification.

Implements ensemble training and inference for epistemic uncertainty estimation.
Based on: Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty using Deep Ensembles" (NeurIPS 2017)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path


class EnsembleModel:
    """
    Manages an ensemble of models for uncertainty quantification.
    
    Epistemic Uncertainty: Variance across ensemble members indicates model uncertainty.
    Different models will disagree on uncertain predictions, giving higher variance.
    """
    
    def __init__(self, model_class: type, model_kwargs: Dict, num_ensemble: int = 5, 
                 ensemble_seeds: Optional[List[int]] = None):
        """
        Initialize ensemble.
        
        Args:
            model_class: Model class (e.g., AdaptivePIGCN, PIGCLSTM)
            model_kwargs: Arguments to pass to model constructor
            num_ensemble: Number of models in ensemble (default: 5)
            ensemble_seeds: List of random seeds for each model (if None, generates automatically)
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.num_ensemble = num_ensemble
        self.models: List[nn.Module] = []
        
        # Generate seeds if not provided
        if ensemble_seeds is None:
            ensemble_seeds = [42 + i * 100 for i in range(num_ensemble)]
        self.ensemble_seeds = ensemble_seeds
        
    def train_ensemble(self, train_loader, val_loader, trainer_class, trainer_kwargs: Dict,
                      device: torch.device, save_dir: str) -> List[str]:
        """
        Train all models in the ensemble with different random seeds.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            trainer_class: Trainer class (e.g., ModelTrainer)
            trainer_kwargs: Arguments for trainer constructor
            device: Device to train on
            save_dir: Directory to save ensemble models
            
        Returns:
            List of paths to saved model checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        model_paths = []
        
        for i, seed in enumerate(self.ensemble_seeds):
            print(f"\n{'='*80}")
            print(f"Training Ensemble Member {i+1}/{self.num_ensemble} (seed={seed})")
            print(f"{'='*80}")
            
            # Set random seed for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            
            # Create model with different initialization
            model = self.model_class(**self.model_kwargs)
            model = model.to(device)
            
            # Create trainer
            trainer = trainer_class(model=model, **trainer_kwargs)
            
            # Train model
            trainer.train(train_loader, val_loader)
            
            # Save model
            model_path = os.path.join(save_dir, f"ensemble_member_{i+1}_seed_{seed}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'seed': seed,
                'ensemble_member': i+1,
            }, model_path)
            model_paths.append(model_path)
            
            # Store model for inference
            self.models.append(model)
            
            print(f"✓ Saved ensemble member {i+1} to {model_path}")
        
        return model_paths
    
    def load_ensemble(self, model_paths: List[str], device: torch.device):
        """
        Load pre-trained ensemble models.
        
        Args:
            model_paths: List of paths to saved model checkpoints
            device: Device to load models on
        """
        self.models = []
        for i, path in enumerate(model_paths):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Ensemble model not found: {path}")
            
            # Create model
            model = self.model_class(**self.model_kwargs)
            model = model.to(device)
            
            # Load weights
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.models.append(model)
            print(f"✓ Loaded ensemble member {i+1} from {path}")
    
    def predict_ensemble(self, features: torch.Tensor, adjacency: torch.Tensor,
                        device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions from all ensemble members.
        
        Args:
            features: Input features [batch, buses, features] or [batch, seq_len, buses, features]
            adjacency: Adjacency matrix [batch, buses, buses]
            device: Device to run inference on
            
        Returns:
            mean_pred: Mean prediction across ensemble [batch, buses, features]
            std_pred: Standard deviation across ensemble [batch, buses, features]
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                pred = model(features.to(device), adjacency.to(device))
                predictions.append(pred.cpu())
        
        # Stack predictions: [num_ensemble, batch, buses, features]
        predictions_tensor = torch.stack(predictions, dim=0)
        
        # Compute mean and std across ensemble dimension
        mean_pred = torch.mean(predictions_tensor, dim=0)  # [batch, buses, features]
        std_pred = torch.std(predictions_tensor, dim=0)    # [batch, buses, features]
        
        return mean_pred, std_pred
    
    def get_epistemic_uncertainty(self, features: torch.Tensor, adjacency: torch.Tensor,
                                  device: torch.device) -> torch.Tensor:
        """
        Get epistemic uncertainty (variance across ensemble).
        
        Args:
            features: Input features
            adjacency: Adjacency matrix
            device: Device to run inference on
            
        Returns:
            epistemic_uncertainty: Standard deviation across ensemble [batch, buses, features]
        """
        _, std_pred = self.predict_ensemble(features, adjacency, device)
        return std_pred

