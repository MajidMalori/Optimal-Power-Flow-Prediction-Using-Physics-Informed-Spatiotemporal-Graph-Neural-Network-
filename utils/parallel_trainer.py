"""
Parallel training utilities for concurrent model training across multiple bus systems.
Optimized for Vast.ai high-memory, multi-core environments.
"""

import os
import torch
import logging
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Queue
import threading
import time
import copy
import gc
from typing import Dict, List, Tuple, Any, Optional

from utils.data_loader import load_power_system_data, create_data_loaders
from utils.metrics import PowerSystemLoss
from utils.optimization import (setup_hyperparameter_bounds, create_model_kwargs, 
                               generate_run_name, process_optimization_params, 
                               calculate_objective_score)
from utils.evaluation import evaluate_model
from trainers.model_trainer import PowerSystemTrainer


class ParallelBusSystemTrainer:
    """
    Manages parallel training across multiple bus systems simultaneously.
    Designed for high-memory environments like Vast.ai.
    """
    
    def __init__(self, base_config, models_to_test, bus_systems_to_test, device):
        self.base_config = base_config
        self.models_to_test = models_to_test
        self.bus_systems_to_test = bus_systems_to_test
        self.device = device
        
        # Pre-load all data to avoid repeated loading
        self.preloaded_data = {}
        self.model_class_map = base_config.get_model_class_map()
        self.model_config_map = base_config.model_config_map
        
        # Thread-safe result storage
        self.results_lock = threading.Lock()
        self.all_results = []
        
        # Progress tracking
        self.progress_queue = Queue()
        self.total_tasks = len(bus_systems_to_test) * len(models_to_test)
        self.completed_tasks = 0
        
    def preload_all_data(self):
        """Pre-load data for all bus systems to avoid repeated I/O."""
        print("🔄 Pre-loading data for all bus systems...")
        
        for num_buses in self.bus_systems_to_test:
            case_name = f"case{num_buses}"
            try:
                print(f"📊 Loading data for {num_buses}-bus system...")
                data_tuple = load_power_system_data(self.base_config, case_name)
                self.preloaded_data[num_buses] = data_tuple
                print(f"✅ Data loaded for {num_buses}-bus system")
            except FileNotFoundError as e:
                print(f"❌ Failed to load data for {num_buses}-bus: {e}")
                self.preloaded_data[num_buses] = None
        
        print(f"🎯 Pre-loaded data for {len([k for k, v in self.preloaded_data.items() if v is not None])} bus systems")
    
    def train_single_model_on_bus_system(self, model_name: str, num_buses: int, 
                                       mosoa_params: Dict) -> Dict[str, Any]:
        """
        Train a single model on a specific bus system.
        This function is designed to be called in parallel.
        """
        try:
            # Get pre-loaded data
            if num_buses not in self.preloaded_data or self.preloaded_data[num_buses] is None:
                return {'error': f'No data available for {num_buses}-bus system'}
            
            data_tuple = self.preloaded_data[num_buses]
            _features, _adjacency, _ybus_matrices, _targets, _energy_coeffs, _carbon_coeffs, _renewable_fractions, _normalizer = data_tuple
            
            # Get model configuration
            model_config = self.model_config_map[model_name]
            is_sequential = self.base_config.is_sequential_model(model_name)
            is_physics_informed = self.base_config.is_physics_informed(model_name)
            uses_adaptive_graph = self.base_config.uses_adaptive_graph(model_name)
            
            print(f"🚀 Starting parallel training: {model_name} on {num_buses}-bus system")
            
            # Setup hyperparameter bounds
            param_bounds = setup_hyperparameter_bounds(
                model_name, model_config, num_buses, 
                is_physics_informed, is_sequential, uses_adaptive_graph
            )
            
            # For parallel training, we'll use a simplified hyperparameter search
            # You can integrate full MoSOA here if needed
            best_params = self._get_default_params(param_bounds, model_name, num_buses)
            
            # Create run configuration
            run_config = copy.deepcopy(self.base_config)
            for key, value in best_params.items():
                setattr(run_config, key.upper(), value)
            run_config.NUM_BUSES = num_buses
            
            # Create data loaders with parallel workers
            loaders = create_data_loaders(
                _features, _adjacency, _ybus_matrices, _targets, 
                _energy_coeffs, _carbon_coeffs, _renewable_fractions, run_config, 
                is_static=(not is_sequential)
            )
            train_loader, val_loader, test_loader = loaders
            
            # Create model
            model_kwargs = create_model_kwargs(
                model_config, best_params, num_buses, is_sequential, uses_adaptive_graph
            )
            
            # Use a separate device for each parallel training (if multiple GPUs available)
            model_device = self._get_device_for_parallel_training()
            model = self.model_class_map[model_name](**model_kwargs).to(model_device)
            
            # Create loss function and optimizer
            criterion = PowerSystemLoss(
                config=run_config, 
                normalizer=_normalizer, 
                is_gcn=(not is_physics_informed)
            ).to(model_device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=run_config.LEARNING_RATE)
            
            # Create trainer
            trainer = PowerSystemTrainer(model, criterion, optimizer, run_config, model_device, is_physics_informed)
            
            # Train the model
            print(f"🔥 Training {model_name} on {num_buses}-bus system...")
            trainer.train(train_loader, val_loader)
            
            # Evaluate model
            val_metrics = evaluate_model(model, val_loader, model_device, run_config, _normalizer, is_sequential)
            test_metrics = evaluate_model(model, test_loader, model_device, run_config, _normalizer, is_sequential)
            
            # Calculate objective score
            total_loss = calculate_objective_score(val_metrics, run_config, is_physics_informed)
            
            # Prepare results
            result = {
                'model_name': model_name,
                'num_buses': num_buses,
                'is_physics_informed': is_physics_informed,
                'is_sequential': is_sequential,
                'best_params': best_params,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'total_loss': total_loss,
                'training_history': trainer.get_training_history(),
                'model_state': model.state_dict(),
                'success': True
            }
            
            print(f"✅ Completed training: {model_name} on {num_buses}-bus system (Loss: {total_loss:.6f})")
            
            # Update progress
            with self.results_lock:
                self.completed_tasks += 1
                progress = (self.completed_tasks / self.total_tasks) * 100
                print(f"📈 Overall Progress: {self.completed_tasks}/{self.total_tasks} ({progress:.1f}%)")
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Failed training {model_name} on {num_buses}-bus: {str(e)}"
            print(error_msg)
            logging.error(error_msg, exc_info=True)
            return {
                'model_name': model_name,
                'num_buses': num_buses,
                'error': str(e),
                'success': False
            }
        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def _get_default_params(self, param_bounds: Dict, model_name: str, num_buses: int) -> Dict:
        """Get reasonable default parameters for quick parallel training."""
        params = {}
        
        for param_name, (min_val, max_val) in param_bounds.items():
            if param_name.lower() == 'hidden_dim':
                # Use adaptive hidden dimensions
                if num_buses <= 33:
                    params[param_name] = 64
                elif num_buses <= 57:
                    params[param_name] = 72
                else:
                    params[param_name] = 96
            elif param_name.lower() == 'num_gc_layers':
                params[param_name] = 3  # Good default for most models
            elif param_name.lower() == 'rnn_layers':
                params[param_name] = 2  # Good default for sequential models
            elif param_name.lower() == 'sequence_length':
                params[param_name] = 10  # Good default for sequential models
            elif param_name.lower() == 'embedding_dim':
                params[param_name] = 16  # Good default for adaptive models
            elif param_name.lower() == 'phi':
                params[param_name] = 0.5  # Middle value for adaptive parameter
            else:
                # Use middle value for other parameters
                params[param_name] = (min_val + max_val) / 2
        
        return params
    
    def _get_device_for_parallel_training(self):
        """Get appropriate device for parallel training."""
        if torch.cuda.is_available():
            # For now, use the same device. In multi-GPU setup, this could be optimized
            return self.device
        else:
            return 'cpu'
    
    def train_all_parallel(self, max_workers: int = 4) -> List[Dict[str, Any]]:
        """
        Train all models on all bus systems in parallel.
        
        Args:
            max_workers: Maximum number of parallel workers (adjust based on GPU memory)
        """
        print(f"🚀 Starting parallel training with {max_workers} workers")
        print(f"📊 Total tasks: {self.total_tasks}")
        
        # Pre-load all data first
        self.preload_all_data()
        
        # Create tasks for parallel execution
        tasks = []
        for num_buses in self.bus_systems_to_test:
            if self.preloaded_data.get(num_buses) is not None:
                mosoa_params = self.base_config._ModelConfig.get_adaptive_mosoa_params(num_buses)
                for model_name in self.models_to_test:
                    tasks.append((model_name, num_buses, mosoa_params))
        
        print(f"📋 Created {len(tasks)} training tasks")
        
        # Execute tasks in parallel using ThreadPoolExecutor
        # ThreadPoolExecutor is better for I/O bound tasks and GPU operations
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.train_single_model_on_bus_system, model_name, num_buses, mosoa_params): 
                (model_name, num_buses) 
                for model_name, num_buses, mosoa_params in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                model_name, num_buses = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.get('success', False):
                        print(f"✅ Completed: {model_name} on {num_buses}-bus")
                    else:
                        print(f"❌ Failed: {model_name} on {num_buses}-bus")
                        
                except Exception as e:
                    print(f"❌ Exception in {model_name} on {num_buses}-bus: {e}")
                    results.append({
                        'model_name': model_name,
                        'num_buses': num_buses,
                        'error': str(e),
                        'success': False
                    })
        
        print(f"🏁 Parallel training completed!")
        print(f"✅ Successful: {len([r for r in results if r.get('success', False)])}")
        print(f"❌ Failed: {len([r for r in results if not r.get('success', False)])}")
        
        return results


class ParallelHyperparameterOptimizer:
    """
    Parallel hyperparameter optimization using multiple processes.
    Each process handles a subset of the MoSOA population.
    """
    
    def __init__(self, base_config, model_name, num_buses, data_tuple):
        self.base_config = base_config
        self.model_name = model_name
        self.num_buses = num_buses
        self.data_tuple = data_tuple
        
        # Model characteristics
        self.is_sequential = base_config.is_sequential_model(model_name)
        self.is_physics_informed = base_config.is_physics_informed(model_name)
        self.uses_adaptive_graph = base_config.uses_adaptive_graph(model_name)
        
        # Model configuration
        self.model_config = base_config.model_config_map[model_name]
        self.model_class_map = base_config.get_model_class_map()
    
    def parallel_objective_evaluation(self, param_sets: List[np.ndarray], 
                                    param_keys: List[str], max_workers: int = 4) -> List[float]:
        """
        Evaluate multiple parameter sets in parallel.
        
        Args:
            param_sets: List of parameter arrays to evaluate
            param_keys: Parameter names
            max_workers: Number of parallel workers
        
        Returns:
            List of objective scores
        """
        print(f"🔄 Evaluating {len(param_sets)} parameter sets in parallel...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._evaluate_single_params, params, param_keys)
                for params in param_sets
            ]
            
            results = []
            for i, future in enumerate(as_completed(futures)):
                try:
                    score = future.result()
                    results.append(score)
                    print(f"✅ Completed evaluation {i+1}/{len(param_sets)} (Score: {score:.6f})")
                except Exception as e:
                    print(f"❌ Failed evaluation {i+1}: {e}")
                    results.append(float('inf'))
        
        return results
    
    def _evaluate_single_params(self, params_array: np.ndarray, param_keys: List[str]) -> float:
        """Evaluate a single parameter set."""
        try:
            params = process_optimization_params(param_keys, params_array)
            
            # Create run configuration
            run_config = copy.deepcopy(self.base_config)
            for key, value in params.items():
                setattr(run_config, key.upper(), value)
            run_config.NUM_BUSES = self.num_buses
            
            # Unpack data
            _features, _adjacency, _ybus_matrices, _targets, _energy_coeffs, _carbon_coeffs, _renewable_fractions, _normalizer = self.data_tuple
            
            # Create data loaders
            loaders = create_data_loaders(
                _features, _adjacency, _ybus_matrices, _targets, 
                _energy_coeffs, _carbon_coeffs, _renewable_fractions, run_config, 
                is_static=(not self.is_sequential)
            )
            train_loader, val_loader, test_loader = loaders
            
            # Create model
            model_kwargs = create_model_kwargs(
                self.model_config, params, self.num_buses, 
                self.is_sequential, self.uses_adaptive_graph
            )
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = self.model_class_map[self.model_name](**model_kwargs).to(device)
            
            # Create loss and optimizer
            criterion = PowerSystemLoss(
                config=run_config, 
                normalizer=_normalizer, 
                is_gcn=(not self.is_physics_informed)
            ).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=run_config.LEARNING_RATE)
            
            # Train model
            trainer = PowerSystemTrainer(model, criterion, optimizer, run_config, device, self.is_physics_informed)
            trainer.train(train_loader, val_loader)
            
            # Evaluate
            val_metrics = evaluate_model(model, val_loader, device, run_config, _normalizer, self.is_sequential)
            total_loss = calculate_objective_score(val_metrics, run_config, self.is_physics_informed)
            
            return total_loss
            
        except Exception as e:
            logging.error(f"Parameter evaluation failed: {e}", exc_info=True)
            return float('inf')
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
