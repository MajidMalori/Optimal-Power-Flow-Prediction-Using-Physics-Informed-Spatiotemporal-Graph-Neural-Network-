"""
Optimization utilities for hyperparameter tuning and algorithm implementation.
Contains MoSOA (Multi-objective Seagull Optimization Algorithm) and related functions.
"""

import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Callable, Any


def _init_positions(num_agents: int, dim: int, upper_bound: np.ndarray, lower_bound: np.ndarray) -> np.ndarray:
    """Initialize positions for optimization agents within bounds."""
    if isinstance(upper_bound, (int, float)): 
        upper_bound = np.full(dim, upper_bound)
    if isinstance(lower_bound, (int, float)): 
        lower_bound = np.full(dim, lower_bound)
    
    positions = np.zeros((num_agents, dim))
    for i in range(dim):
        positions[:, i] = np.random.uniform(lower_bound[i], upper_bound[i], num_agents)
    return positions


def soa(num_agents: int, max_iter: int, lower_bound: np.ndarray, upper_bound: np.ndarray, 
        dim: int, objective_func: Callable) -> Tuple[float, np.ndarray, List[float], List[Dict]]:
    """
    Enhanced Seagull Optimization Algorithm for hyperparameter tuning.
    
    Args:
        num_agents: Number of seagull agents
        max_iter: Maximum number of iterations
        lower_bound: Lower bounds for parameters
        upper_bound: Upper bounds for parameters
        dim: Dimension of parameter space
        objective_func: Function to minimize
        
    Returns:
        Tuple of (best_score, best_position, convergence_history, iteration_details)
    """
    print("\nStarting Enhanced Seagull Optimization Algorithm for Hyperparameter Tuning...")
    
    best_position, best_score = np.zeros(dim), float('inf')
    positions = _init_positions(num_agents, dim, upper_bound, lower_bound)
    convergence_curve = []
    iteration_details = []  # Track best configuration per iteration
    
    # Algorithm parameters
    lambda_uncertainty, lambda_beta, beta_max = 5.0, 5.0, 2.0
    
    pbar = tqdm(range(max_iter), desc="MoSOA Progress")
    for l in pbar:
        # Evaluate fitness for all agents
        fitness_all = [objective_func(np.clip(p, lower_bound, upper_bound)) for p in positions]
        valid_fitness = [(f, i) for i, f in enumerate(fitness_all) if f is not None and f != float('inf')]
        
        # Track iteration details - best configuration in THIS iteration
        iteration_best_score = float('inf')
        iteration_best_position = None
        
        # Update best solution
        if valid_fitness:
            current_best_score_iter, best_agent_idx = min(valid_fitness, key=lambda item: item[0])
            iteration_best_score = current_best_score_iter
            iteration_best_position = positions[best_agent_idx].copy()
            
            if current_best_score_iter < best_score:
                best_score = current_best_score_iter
                best_position = positions[best_agent_idx].copy()
        
        # Store iteration details (best configuration found in this iteration)
        iteration_details.append({
            'iteration': l + 1,
            'best_score': iteration_best_score,
            'best_position': iteration_best_position.copy() if iteration_best_position is not None else None,
            'global_best_score': best_score,
            'num_valid_evaluations': len(valid_fitness)
        })
        
        convergence_curve.append(best_score)
        
        # Calculate diversity measure
        sigma = np.std([f for f, _ in valid_fitness]) if valid_fitness else 1e-9
        if sigma == 0: 
            sigma = 1e-9
        
        # Update algorithm parameters
        fc = 2 - l * (2 / max_iter)
        beta = beta_max * np.exp(-lambda_beta * (l / max_iter))
        
        # Update positions for all agents
        for i in range(num_agents):
            # Time and uncertainty factors
            time_factor = 1 - np.sin((np.pi / 2) * (l / max_iter))
            uncertainty_factor = 1 / (1 + lambda_uncertainty * sigma)
            
            # Seagull algorithm parameters
            A1 = 1.0 * time_factor * uncertainty_factor
            b = 1.0 * (1 - 2 / (1 + np.exp((2 * l) / max_iter))) + -1.0
            rand_ll = (fc - 1) * np.random.rand() + 1

            # Update position using seagull migration behavior
            D_alphs = fc * positions[i, :] + A1 * (best_position - positions[i, :])
            X1 = D_alphs * np.exp(b * rand_ll) * np.cos(rand_ll * 2 * np.pi) + best_position

            # Add random component from other seagulls
            P_rand = positions[np.random.randint(0, num_agents), :]
            new_position = X1 + beta * (P_rand - positions[i, :])
            
            # CRITICAL: Ensure positions stay within bounds
            positions[i, :] = np.clip(new_position, lower_bound, upper_bound)

        pbar.set_description(f"MoSOA Iteration {l+1}/{max_iter} | Best MSE: {best_score:.6f}")
        
        # Add spacing between iterations by updating progress bar with newline
        if l < max_iter - 1:  # Don't add space after the last iteration
            pbar.write("")  # Use pbar.write() instead of print() to avoid interference
    
    return best_score, best_position, convergence_curve, iteration_details


def setup_hyperparameter_bounds(model_name: str, model_config: Any, num_buses: int, 
                               is_physics_informed: bool, is_sequential: bool, 
                               uses_adaptive_graph: bool) -> Dict[str, Tuple[float, float]]:
    """
    Setup hyperparameter bounds for optimization based on model characteristics.
    
    Args:
        model_name: Name of the model
        model_config: Model configuration object
        num_buses: Number of buses in the system
        is_physics_informed: Whether model uses physics-informed loss
        is_sequential: Whether model is sequential (LSTM/GRU)
        uses_adaptive_graph: Whether model uses adaptive graph features
        
    Returns:
        Dictionary of parameter bounds
    """
    # Use adaptive scaling for hidden dimensions based on system size
    hidden_range = (model_config.get_hidden_dim_range(num_buses) 
                   if hasattr(model_config, 'get_hidden_dim_range') 
                   else model_config.HIDDEN_DIM_RANGE)
    
    param_bounds = {
        'HIDDEN_DIM': hidden_range, 
        'NUM_GC_LAYERS': model_config.NUM_GC_LAYERS_RANGE
    }
    
    # Add physics-informed parameters
    if is_physics_informed: 
        param_bounds['LAMBDA_P'] = (1.0, 50.0)
        param_bounds['LAMBDA_V'] = (1.0, 50.0)
    
    # Add sequential model parameters
    if is_sequential: 
        param_bounds.update({
            'SEQUENCE_LENGTH': model_config.SEQUENCE_LENGTH_RANGE, 
            'RNN_LAYERS': model_config.RNN_LAYERS_RANGE
        })
    
    # Add adaptive graph parameters
    if uses_adaptive_graph: 
        param_bounds.update({
            'EMBEDDING_DIM': model_config.EMBEDDING_DIM_RANGE, 
            'PHI': model_config.PHI_RANGE
        })
    
    return param_bounds


def create_model_kwargs(model_config: Any, params: Dict[str, Any], num_buses: int, 
                       is_sequential: bool, uses_adaptive_graph: bool) -> Dict[str, Any]:
    """
    Create model keyword arguments from optimized parameters.
    
    Args:
        model_config: Model configuration object
        params: Optimized parameters dictionary
        num_buses: Number of buses in the system
        is_sequential: Whether model is sequential
        uses_adaptive_graph: Whether model uses adaptive graph features
        
    Returns:
        Dictionary of model keyword arguments
    """
    model_kwargs = {
        'feature_dim': model_config.FEATURE_DIM,
        'hidden_dim': int(params['HIDDEN_DIM']),
        'num_gc_layers': int(params['NUM_GC_LAYERS']),
        'num_buses': num_buses,
        'dropout': model_config.DROPOUT
    }
    
    if is_sequential:
        model_kwargs['rnn_layers'] = int(params['RNN_LAYERS'])
    
    if uses_adaptive_graph:
        model_kwargs.update({
            'embedding_dim': int(params['EMBEDDING_DIM']),
            'phi': float(params['PHI'])
        })
    
    return model_kwargs


def generate_run_name(model_name: str, params: Dict[str, Any], num_buses: int, 
                     is_sequential: bool) -> str:
    """Generate a descriptive run name based on model and parameters."""
    run_name = f"run_{model_name}_B{num_buses}_H{params.get('HIDDEN_DIM', 'N/A')}_GC{params.get('NUM_GC_LAYERS', 'N/A')}"
    
    if is_sequential:
        run_name += f"_SL{params.get('SEQUENCE_LENGTH', 'N/A')}_R{params.get('RNN_LAYERS', 'N/A')}"
    
    return run_name


def process_optimization_params(param_keys: List[str], param_values: np.ndarray) -> Dict[str, Any]:
    """
    Process optimization parameters, converting integers where needed.
    
    Args:
        param_keys: List of parameter names
        param_values: Array of parameter values
        
    Returns:
        Dictionary of processed parameters
    """
    params = {key: val for key, val in zip(param_keys, param_values)}
    
    # Convert specific parameters to integers
    for k in ['HIDDEN_DIM', 'NUM_GC_LAYERS', 'SEQUENCE_LENGTH', 'RNN_LAYERS', 'EMBEDDING_DIM']:
        if k in params:
            params[k] = int(round(params[k]))
    
    return params


def calculate_objective_score(metrics: Dict[str, float], config: Any, is_physics_informed: bool) -> float:
    """
    Calculate objective score for optimization based on model type.
    
    Args:
        metrics: Dictionary of evaluation metrics
        config: Configuration object with lambda values
        is_physics_informed: Whether model is physics-informed
        
    Returns:
        Total objective score to minimize
    """
    if is_physics_informed:
        total_loss = (metrics['mse'] + 
                     config.LAMBDA_P * metrics['power_violation'] + 
                     config.LAMBDA_V * metrics['voltage_violation'])
    else:
        # For non-physics-informed models, only use MSE
        total_loss = metrics['mse']
    
    return total_loss
