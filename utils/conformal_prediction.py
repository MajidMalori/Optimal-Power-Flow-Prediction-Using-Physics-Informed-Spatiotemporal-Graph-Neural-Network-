"""
Conformal Prediction for Guaranteed Prediction Intervals.

Implements conformal prediction with guaranteed coverage probability.
Based on: Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction" (2021)

Key Property: Distribution-free, finite-sample valid prediction intervals.
For confidence level 1-α, intervals contain true value with probability ≥ 1-α.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional
from scipy.stats import norm


class ConformalPredictor:
    """
    Conformal Prediction for uncertainty quantification with statistical guarantees.
    
    Method:
    1. Compute nonconformity scores on calibration set
    2. Compute quantile of scores
    3. Generate prediction intervals with guaranteed coverage
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize conformal predictor.
        
        Args:
            confidence_level: Desired coverage probability (e.g., 0.95 for 95% intervals)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.quantile_value: Optional[float] = None
        self.calibration_scores: Optional[np.ndarray] = None
        
    def calibrate(self, predictions: torch.Tensor, targets: torch.Tensor,
                  mean_pred: torch.Tensor, std_pred: torch.Tensor) -> Dict[str, float]:
        """
        Calibrate conformal predictor on calibration set.
        
        Args:
            predictions: Ensemble mean predictions [n_samples, n_buses, n_features]
            targets: True values [n_samples, n_buses, n_features]
            mean_pred: Mean predictions (same as predictions, for consistency) [n_samples, n_buses, n_features]
            std_pred: Standard deviation (uncertainty) [n_samples, n_buses, n_features]
            
        Returns:
            Dictionary with calibration statistics
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()
        if isinstance(mean_pred, torch.Tensor):
            mean_pred = mean_pred.numpy()
        if isinstance(std_pred, torch.Tensor):
            std_pred = std_pred.numpy()
        
        # Compute nonconformity scores: |target - prediction| / uncertainty
        # Higher score = more nonconforming = prediction is farther from target relative to uncertainty
        errors = np.abs(targets - mean_pred)  # [n_samples, n_buses, n_features]
        std_pred = np.maximum(std_pred, 1e-6)  # Avoid division by zero
        scores = errors / std_pred  # [n_samples, n_buses, n_features]
        
        # Flatten to get all scores
        scores_flat = scores.flatten()
        
        # Compute quantile for conformal prediction
        # Use (n+1)(1-α)/n quantile for finite-sample validity
        n = len(scores_flat)
        quantile_idx = int(np.ceil((n + 1) * (1 - self.alpha)))
        if quantile_idx >= n:
            quantile_idx = n - 1
        
        self.quantile_value = np.quantile(scores_flat, quantile_idx / n)
        self.calibration_scores = scores_flat
        
        # Compute calibration statistics
        stats = {
            'quantile_value': self.quantile_value,
            'num_calibration_samples': n,
            'confidence_level': self.confidence_level,
            'alpha': self.alpha,
        }
        
        return stats
    
    def predict_intervals(self, mean_pred: torch.Tensor, std_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate prediction intervals with guaranteed coverage.
        
        Args:
            mean_pred: Mean predictions [batch, buses, features]
            std_pred: Standard deviation (uncertainty) [batch, buses, features]
            
        Returns:
            lower_bounds: Lower bounds of prediction intervals [batch, buses, features]
            upper_bounds: Upper bounds of prediction intervals [batch, buses, features]
        """
        if self.quantile_value is None:
            raise ValueError("Must calibrate before generating prediction intervals. Call calibrate() first.")
        
        # Convert to numpy if needed
        if isinstance(mean_pred, torch.Tensor):
            mean_pred = mean_pred.numpy()
        if isinstance(std_pred, torch.Tensor):
            std_pred = std_pred.numpy()
        
        # Ensure std_pred is positive
        std_pred = np.maximum(std_pred, 1e-6)
        
        # Compute prediction intervals: [mean - q*std, mean + q*std]
        lower_bounds = mean_pred - self.quantile_value * std_pred
        upper_bounds = mean_pred + self.quantile_value * std_pred
        
        return torch.from_numpy(lower_bounds), torch.from_numpy(upper_bounds)
    
    def compute_coverage(self, lower_bounds: torch.Tensor, upper_bounds: torch.Tensor,
                        targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute empirical coverage rate (should be ≥ confidence_level).
        
        Args:
            lower_bounds: Lower bounds of intervals [n_samples, n_buses, n_features]
            upper_bounds: Upper bounds of intervals [n_samples, n_buses, n_features]
            targets: True values [n_samples, n_buses, n_features]
            
        Returns:
            Dictionary with coverage statistics
        """
        # Convert to numpy
        if isinstance(lower_bounds, torch.Tensor):
            lower_bounds = lower_bounds.numpy()
        if isinstance(upper_bounds, torch.Tensor):
            upper_bounds = upper_bounds.numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()
        
        # Check if targets are within intervals
        within_interval = (targets >= lower_bounds) & (targets <= upper_bounds)
        
        # Compute coverage rate
        coverage_rate = np.mean(within_interval)
        
        # Compute interval width (efficiency)
        interval_width = np.mean(upper_bounds - lower_bounds)
        
        # Per-feature coverage
        per_feature_coverage = np.mean(within_interval, axis=(0, 1))  # [n_features]
        
        stats = {
            'coverage_rate': coverage_rate,
            'target_coverage': self.confidence_level,
            'interval_width_mean': interval_width,
            'per_feature_coverage': per_feature_coverage.tolist(),
            'coverage_valid': coverage_rate >= self.confidence_level - 0.05,  # Allow 5% tolerance
        }
        
        return stats
    
    def get_interval_width(self, lower_bounds: torch.Tensor, upper_bounds: torch.Tensor) -> torch.Tensor:
        """
        Get width of prediction intervals.
        
        Args:
            lower_bounds: Lower bounds [batch, buses, features]
            upper_bounds: Upper bounds [batch, buses, features]
            
        Returns:
            interval_width: Width of intervals [batch, buses, features]
        """
        if isinstance(lower_bounds, torch.Tensor):
            return upper_bounds - lower_bounds
        else:
            return torch.from_numpy(upper_bounds - lower_bounds)


class ConformalEnsemblePredictor:
    """
    Combined Ensemble + Conformal Prediction for comprehensive uncertainty quantification.
    
    Combines:
    - Ensemble methods (epistemic uncertainty)
    - Conformal prediction (guaranteed prediction intervals)
    - Learnable uncertainty (aleatoric uncertainty, already implemented in loss function)
    """
    
    def __init__(self, ensemble_model, confidence_level: float = 0.95):
        """
        Initialize combined predictor.
        
        Args:
            ensemble_model: EnsembleModel instance
            confidence_level: Desired coverage probability
        """
        self.ensemble_model = ensemble_model
        self.conformal_predictor = ConformalPredictor(confidence_level=confidence_level)
        
    def calibrate(self, features: torch.Tensor, adjacency: torch.Tensor, targets: torch.Tensor,
                 device: torch.device) -> Dict[str, float]:
        """
        Calibrate conformal predictor using ensemble predictions.
        
        Args:
            features: Input features [n_samples, buses, features]
            adjacency: Adjacency matrices [n_samples, buses, buses]
            targets: True values [n_samples, buses, features]
            device: Device to run inference on
            
        Returns:
            Calibration statistics
        """
        # Get ensemble predictions
        mean_pred, std_pred = self.ensemble_model.predict_ensemble(features, adjacency, device)
        
        # Calibrate conformal predictor
        stats = self.conformal_predictor.calibrate(
            predictions=mean_pred,
            targets=targets,
            mean_pred=mean_pred,
            std_pred=std_pred
        )
        
        return stats
    
    def predict_with_intervals(self, features: torch.Tensor, adjacency: torch.Tensor,
                              device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Get predictions with conformal prediction intervals.
        
        Args:
            features: Input features [batch, buses, features]
            adjacency: Adjacency matrices [batch, buses, buses]
            device: Device to run inference on
            
        Returns:
            Dictionary with:
                - 'mean': Mean prediction [batch, buses, features]
                - 'std': Standard deviation (epistemic uncertainty) [batch, buses, features]
                - 'lower': Lower bound of prediction interval [batch, buses, features]
                - 'upper': Upper bound of prediction interval [batch, buses, features]
                - 'width': Interval width [batch, buses, features]
        """
        # Get ensemble predictions
        mean_pred, std_pred = self.ensemble_model.predict_ensemble(features, adjacency, device)
        
        # Generate conformal prediction intervals
        lower_bounds, upper_bounds = self.conformal_predictor.predict_intervals(mean_pred, std_pred)
        
        # Compute interval width
        interval_width = self.conformal_predictor.get_interval_width(lower_bounds, upper_bounds)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'lower': lower_bounds,
            'upper': upper_bounds,
            'width': interval_width,
        }

