"""
Optimization utilities for hyperparameter tuning and algorithm implementation.
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


def mosoa_optimizer(num_agents: int, max_iterations: int, lower_bound: np.ndarray, upper_bound: np.ndarray,
                   dim: int, objective_func: Callable) -> Tuple[float, np.ndarray, List[float], List[Dict]]:
    """
    Modified Seagull Optimization Algorithm (MoSOA) for hyperparameter optimization.
    
    Args:
        num_agents: Number of seagull agents
        max_iterations: Maximum number of iterations
        lower_bound: Lower bounds for parameters
        upper_bound: Upper bounds for parameters
        dim: Dimension of parameter space
        objective_func: Function to minimize
        
    Returns:
        Tuple of (best_score, best_position, convergence_history, iteration_details)
    """
    
    # --- 1. Initialization ---
    best_position = np.zeros(dim)
    best_score = float('inf')
    
    positions = _init_positions(num_agents, dim, upper_bound, lower_bound)
    convergence_curve = []
    iteration_details = []
    
    # MoSOA-specific parameters from the paper
    v_max, v_min = 1.0, 0.0
    u = 1.0
    w_max, w_min = 0.9, 0.2
    beta_max = 1.0
    lambda_val = 2.0
    fc_min, fc_max = 0.0, 2.0
    
    # --- 2. Main Optimization Loop ---
    pbar = tqdm(range(max_iterations), desc="MoSOA Progress")
    for l in pbar:
        # --- 2a. Fitness Evaluation and Global Best Update ---
        fitness_all = np.full(num_agents, np.inf)
        for i in range(num_agents):
            positions[i, :] = np.clip(positions[i, :], lower_bound, upper_bound)
            fitness = objective_func(positions[i, :])
            fitness_all[i] = fitness
            
            if fitness < best_score:
                best_score = fitness
                best_position = positions[i, :].copy()

        # --- 2b. Calculate Adaptive Parameters ---
        f_max, f_min, f_avg = np.max(fitness_all), np.min(fitness_all), np.mean(fitness_all)
        sigma = np.std(fitness_all)
        
        M = 1.0 if (f_avg - f_min) == 0 else (f_max - f_avg) / (f_avg - f_min)
        fc_ada = fc_min + M * (fc_max - fc_min) + (sigma * np.random.randn())
        A = fc_ada * (1 - np.sin((np.pi / 2) * (l / max_iterations)))
        
        v = v_max * (1 - l / max_iterations)
        w = (w_max - w_min) * (1 - np.cos(np.pi / 2 * (l / max_iterations))) + w_min
        beta = beta_max * np.exp(-lambda_val * (l / max_iterations))
        
        # --- 2c. Update Agent Positions ---
        for i in range(num_agents):
            B = 2 * (A**2) * np.random.rand()
            Ms = B * (best_position - positions[i, :])
            Ds = np.abs(Ms)
            
            k = np.random.uniform(0, 2 * np.pi)
            r = u * np.exp(k * v)
            spiral_attack = Ds * r * np.cos(2 * np.pi * k)

            rand_agent_idx = np.random.randint(0, num_agents)
            p_rand = positions[rand_agent_idx, :]
            perturbation = beta * (p_rand - positions[i, :])
            
            positions[i, :] = spiral_attack + (w * best_position) + perturbation
        
        # Track iteration details
        iteration_details.append({
            'iteration': l + 1,
            'best_score': best_score,
            'best_position': best_position.copy(),
            'global_best_score': best_score,
            'num_valid_evaluations': num_agents
        })
        
        convergence_curve.append(best_score)
        pbar.set_description(f"MoSOA Iteration {l+1}/{max_iterations} | Best Score: {best_score:.6e}")

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
    
    # For sequential models, use system-size-dependent ranges to prevent OOM on large systems
    if is_sequential and hasattr(model_config, 'get_sequential_ranges'):
        seq_ranges = model_config.get_sequential_ranges(num_buses)
        hidden_range = seq_ranges['hidden_dim']
        sequence_range = seq_ranges['sequence_length']
        rnn_layers_range = seq_ranges['rnn_layers']
    else:
        sequence_range = model_config.SEQUENCE_LENGTH_RANGE if is_sequential else None
        rnn_layers_range = model_config.RNN_LAYERS_RANGE if is_sequential else None
    
    param_bounds = {
        'HIDDEN_DIM': hidden_range, 
        'NUM_GC_LAYERS': model_config.NUM_GC_LAYERS_RANGE
    }
    
    # Add physics-informed parameters
    if is_physics_informed: 
        param_bounds['LAMBDA_P'] = (1.0, 50.0)
        param_bounds['LAMBDA_V'] = (1.0, 50.0)
    
    # Add sequential model parameters with adaptive ranges
    if is_sequential: 
        param_bounds.update({
            'SEQUENCE_LENGTH': sequence_range, 
            'RNN_LAYERS': rnn_layers_range
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


def trial_based_search(num_trials: int, lower_bound: np.ndarray, upper_bound: np.ndarray,
                      dim: int, objective_func: Callable, 
                      search_strategy: str = 'random') -> Tuple[float, np.ndarray, List[float], List[Dict]]:
    """
    Trial-based hyperparameter search with fixed budget.
    
    Args:
        num_trials: Total number of training runs
        lower_bound: Lower bounds for parameters
        upper_bound: Upper bounds for parameters
        dim: Dimension of parameter space
        objective_func: Function that trains model and returns validation loss
        search_strategy: 'random' or 'latin_hypercube' sampling
        
    Returns:
        Tuple of (best_score, best_position, convergence_history, trial_details)
    """
    
    best_position = np.zeros(dim)
    best_score = float('inf')
    convergence_curve = []
    trial_details = []
    
    # Convert bounds to arrays if they're scalars
    if isinstance(upper_bound, (int, float)):
        upper_bound = np.full(dim, upper_bound)
    if isinstance(lower_bound, (int, float)):
        lower_bound = np.full(dim, lower_bound)
    
    # Generate all trial positions at once
    if search_strategy == 'latin_hypercube':
        # Latin Hypercube Sampling for better space coverage
        positions = _latin_hypercube_sampling(num_trials, dim, lower_bound, upper_bound)
    else:
        # Random sampling (default)
        positions = np.zeros((num_trials, dim))
        for i in range(dim):
            positions[:, i] = np.random.uniform(lower_bound[i], upper_bound[i], num_trials)
    
    # Run trials sequentially (ONE training run per trial)
    pbar = tqdm(range(num_trials), desc="Trial Progress")
    for trial_idx in pbar:
        current_params = positions[trial_idx]
        
        # Clip to ensure bounds
        current_params = np.clip(current_params, lower_bound, upper_bound)
        
        # ONE full model training run
        try:
            score = objective_func(current_params)
        except Exception as e:
            print(f"Trial {trial_idx + 1} failed: {e}")
            score = float('inf')
        
        # Update best if this trial is better
        if score < best_score:
            best_score = score
            best_position = current_params.copy()
        
        # Track convergence (best score so far)
        convergence_curve.append(best_score)
        
        # Store trial details
        trial_details.append({
            'trial': trial_idx + 1,
            'score': score,
            'position': current_params.copy(),
            'is_best': (score == best_score),
            'global_best_score': best_score
        })
        
        pbar.set_description(f"Trial {trial_idx + 1}/{num_trials} | Best Score: {best_score:.6f}")
    
    return best_score, best_position, convergence_curve, trial_details


def _latin_hypercube_sampling(num_samples: int, dim: int, 
                              lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
    """
    Latin Hypercube Sampling for better coverage of parameter space.
    
    This ensures samples are more evenly distributed than pure random sampling.
    """
    # Generate Latin Hypercube samples in [0, 1]
    samples = np.zeros((num_samples, dim))
    
    for d in range(dim):
        # Divide [0,1] into num_samples equal intervals
        intervals = np.linspace(0, 1, num_samples + 1)
        # Randomly sample within each interval
        samples[:, d] = np.random.uniform(intervals[:-1], intervals[1:])
        # Shuffle to avoid correlation between dimensions
        np.random.shuffle(samples[:, d])
    
    # Scale to actual bounds
    for d in range(dim):
        samples[:, d] = samples[:, d] * (upper_bound[d] - lower_bound[d]) + lower_bound[d]
    
    return samples
