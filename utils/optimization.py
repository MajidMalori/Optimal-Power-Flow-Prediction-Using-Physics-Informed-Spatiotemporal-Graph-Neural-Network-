"""
Optimization utilities for hyperparameter tuning and algorithm implementation.
"""

import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Callable, Any


def _format_value(val: float) -> str:
    """Format a value to 6-7 significant figures for concise display."""
    if isinstance(val, (np.integer, int)):
        return str(int(val))
    elif abs(val) < 0.01 or abs(val) > 1000:
        return f"{val:.5g}"  # Scientific notation for very small/large
    else:
        return f"{val:.6g}"  # 6 significant figures


def _init_positions(num_agents: int, dim: int, upper_bound: np.ndarray, lower_bound: np.ndarray) -> np.ndarray:
    """Initialize positions for optimization agents within bounds."""
    if isinstance(upper_bound, (int, float)): 
        upper_bound = np.full(dim, upper_bound)
    if isinstance(lower_bound, (int, float)): 
        lower_bound = np.full(dim, lower_bound)
    
    # Vectorized initialization: generate all positions at once using broadcasting
    # Shape: (num_agents, dim) - each row is an agent, each column is a dimension
    lower_expanded = np.tile(lower_bound, (num_agents, 1))  # (num_agents, dim)
    upper_expanded = np.tile(upper_bound, (num_agents, 1))  # (num_agents, dim)
    positions = np.random.uniform(lower_expanded, upper_expanded, size=(num_agents, dim))
    return positions


def mosoa_optimizer(num_agents: int, max_iterations: int, lower_bound: np.ndarray, upper_bound: np.ndarray,
                   dim: int, objective_func: Callable, param_keys: List[str] = None) -> Tuple[float, np.ndarray, List[float], List[Dict]]:
    """
    Modified Seagull Optimization Algorithm (MoSOA) for hyperparameter optimization.
    
    Args:
        num_agents: Number of seagull agents
        max_iterations: Maximum number of iterations
        lower_bound: Lower bounds for parameters
        upper_bound: Upper bounds for parameters
        dim: Dimension of parameter space
        objective_func: Function to minimize
        param_keys: Optional list of parameter names for display
        
    Returns:
        Tuple of (best_score, best_position, convergence_history, iteration_details)
    """
    
    # --- 1. Initialization ---
    # Convert bounds to numpy arrays if they're lists (robustness)
    lower_bound = np.asarray(lower_bound, dtype=np.float64)
    upper_bound = np.asarray(upper_bound, dtype=np.float64)
    
    # Validate bounds
    if len(lower_bound) != dim or len(upper_bound) != dim:
        raise ValueError(f"Bounds dimension mismatch: lower_bound={len(lower_bound)}, upper_bound={len(upper_bound)}, dim={dim}")
    if np.any(lower_bound >= upper_bound):
        raise ValueError("Lower bounds must be strictly less than upper bounds")
    
    # Initialize best_position to middle of bounds (not zeros) to avoid invalid models
    best_position = (lower_bound + upper_bound) / 2.0
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
    for l in range(max_iterations):
        # --- 2a. Fitness Evaluation and Global Best Update ---
        fitness_all = np.full(num_agents, np.inf)
        for i in range(num_agents):
            original_pos = positions[i, :].copy()
            positions[i, :] = np.clip(positions[i, :], lower_bound, upper_bound)
            was_clipped = not np.allclose(original_pos, positions[i, :])
            if was_clipped and param_keys:
                clipped_dims = np.where(~np.isclose(original_pos, positions[i, :]))[0]
                # Log first time clipping happens for each dimension (avoid spam)
            
            # Validate parameters before evaluation to prevent invalid model initialization
            try:
                fitness = objective_func(positions[i, :])
                # Only update if fitness is valid (not NaN or inf)
                if np.isfinite(fitness):
                    fitness_all[i] = fitness
                    
                    if fitness < best_score:
                        best_score = fitness
                        best_position = positions[i, :].copy()
                else:
                    # Invalid fitness - keep previous best
                    fitness_all[i] = np.inf
            except (ValueError, RuntimeError) as e:
                # Model initialization failed - skip this position
                fitness_all[i] = np.inf
                continue

        # --- 2b. Calculate Adaptive Parameters ---
        f_max, f_min, f_avg = np.max(fitness_all), np.min(fitness_all), np.mean(fitness_all)
        sigma = np.std(fitness_all)
        
        M = 1.0 if (f_avg - f_min) == 0 else (f_max - f_avg) / (f_avg - f_min)
        fc_ada = fc_min + M * (fc_max - fc_min) + (sigma * np.random.randn())
        A = fc_ada * (1 - np.sin((np.pi / 2) * (l / max_iterations)))
        
        # Tanh-based decay for v (Eq. 22 in MOSOA paper)
        # v decays non-linearly from v_max to v_min
        # Using the formula: v = v_max * tanh(|1 - l/max_iterations|)
        v = v_max * np.tanh(np.abs(1 - l / max_iterations))
        w = (w_max - w_min) * (1 - np.cos(np.pi / 2 * (l / max_iterations))) + w_min
        beta = beta_max * np.exp(-lambda_val * (l / max_iterations))
        
        # --- 2c. Update Agent Positions (Vectorized) ---
        # Vectorized update for all agents simultaneously
        num_agents_actual = positions.shape[0]
        
        # B: (num_agents,) - random values for each agent
        B = 2 * (A**2) * np.random.rand(num_agents_actual)
        # Ms: (num_agents, dim) - difference vectors
        Ms = B[:, np.newaxis] * (best_position - positions)  # Broadcasting
        Ds = np.abs(Ms)  # (num_agents, dim)
        
        # k: (num_agents,) - random angles for each agent
        k = np.random.uniform(0, 2 * np.pi, size=num_agents_actual)
        r = u * np.exp(k * v)  # (num_agents,)
        # Spiral attack: (num_agents, dim) - element-wise multiplication with broadcasting
        spiral_attack = Ds * r[:, np.newaxis] * np.cos(2 * np.pi * k)[:, np.newaxis]
        
        # Random agent selection for perturbation: (num_agents,)
        rand_agent_indices = np.random.randint(0, num_agents_actual, size=num_agents_actual)
        p_rand = positions[rand_agent_indices, :]  # (num_agents, dim)
        perturbation = beta * (p_rand - positions)  # (num_agents, dim)
        
        # Update all positions at once: (num_agents, dim)
        positions = spiral_attack + (w * best_position) + perturbation
        
        # Track iteration details
        iteration_details.append({
            'iteration': l + 1,
            'best_score': best_score,
            'best_position': best_position.copy(),
            'global_best_score': best_score,
            'num_valid_evaluations': num_agents
        })
        
        convergence_curve.append(best_score)
        
        # Format hyperparameters concisely for display
        if param_keys:
            params_str = ", ".join([f"{k}={_format_value(best_position[i])}" for i, k in enumerate(param_keys)])
            print(f"MoSOA iter {l+1}/{max_iterations} | Score: {best_score:.6g} | {params_str}")
        else:
            print(f"MoSOA iter {l+1}/{max_iterations} | Score: {best_score:.6g}")

    return best_score, best_position, convergence_curve, iteration_details


def pso_optimizer(num_agents: int, max_iterations: int, lower_bound: np.ndarray, upper_bound: np.ndarray,
                 dim: int, objective_func: Callable, param_keys: List[str] = None) -> Tuple[float, np.ndarray, List[float], List[Dict]]:
    """
    Particle Swarm Optimization (PSO) for benchmarking.
    """
    # --- 1. Initialization ---
    lower_bound = np.asarray(lower_bound, dtype=np.float64)
    upper_bound = np.asarray(upper_bound, dtype=np.float64)
    
    positions = _init_positions(num_agents, dim, upper_bound, lower_bound)
    velocities = np.zeros((num_agents, dim))
    
    # Personal bests
    pbest_positions = positions.copy()
    pbest_scores = np.full(num_agents, np.inf)
    
    # Global best
    gbest_position = np.zeros(dim)
    gbest_score = float('inf')
    
    convergence_curve = []
    iteration_details = []
    
    # PSO parameters
    w = 0.7   # Inertia weight
    c1 = 1.5  # Cognitive weight
    c2 = 1.5  # Social weight
    
    # --- 2. Main Loop ---
    for l in range(max_iterations):
        # Evaluate fitness
        for i in range(num_agents):
            # Clip positions to bounds
            positions[i, :] = np.clip(positions[i, :], lower_bound, upper_bound)
            
            try:
                fitness = objective_func(positions[i, :])
                if np.isfinite(fitness):
                    # Update Personal Best
                    if fitness < pbest_scores[i]:
                        pbest_scores[i] = fitness
                        pbest_positions[i, :] = positions[i, :].copy()
                        
                    # Update Global Best
                    if fitness < gbest_score:
                        gbest_score = fitness
                        gbest_position = positions[i, :].copy()
            except Exception:
                continue
        
        # Update Velocities and Positions
        r1 = np.random.rand(num_agents, dim)
        r2 = np.random.rand(num_agents, dim)
        
        velocities = (w * velocities + 
                     c1 * r1 * (pbest_positions - positions) + 
                     c2 * r2 * (gbest_position - positions))
        
        positions = positions + velocities
        
        convergence_curve.append(gbest_score)
        
        if param_keys:
             print(f"PSO iter {l+1}/{max_iterations} | Score: {gbest_score:.6g}")
        else:
             print(f"PSO iter {l+1}/{max_iterations} | Score: {gbest_score:.6g}")

    return gbest_score, gbest_position, convergence_curve, iteration_details


def gwo_optimizer(num_agents: int, max_iterations: int, lower_bound: np.ndarray, upper_bound: np.ndarray,
                 dim: int, objective_func: Callable, param_keys: List[str] = None) -> Tuple[float, np.ndarray, List[float], List[Dict]]:
    """
    Grey Wolf Optimizer (GWO) for benchmarking.
    """
    # --- 1. Initialization ---
    lower_bound = np.asarray(lower_bound, dtype=np.float64)
    upper_bound = np.asarray(upper_bound, dtype=np.float64)
    
    positions = _init_positions(num_agents, dim, upper_bound, lower_bound)
    
    # Hierarchy of wolves
    alpha_pos = np.zeros(dim)
    alpha_score = float('inf')
    
    beta_pos = np.zeros(dim)
    beta_score = float('inf')
    
    delta_pos = np.zeros(dim)
    delta_score = float('inf')
    
    convergence_curve = []
    iteration_details = []
    
    # --- 2. Main Loop ---
    for l in range(max_iterations):
        # Evaluate fitness
        for i in range(num_agents):
            positions[i, :] = np.clip(positions[i, :], lower_bound, upper_bound)
            
            try:
                fitness = objective_func(positions[i, :])
                if np.isfinite(fitness):
                    # Update Alpha, Beta, Delta
                    if fitness < alpha_score:
                        # Shift down
                        delta_score, delta_pos = beta_score, beta_pos.copy()
                        beta_score, beta_pos = alpha_score, alpha_pos.copy()
                        alpha_score, alpha_pos = fitness, positions[i, :].copy()
                    elif fitness < beta_score:
                        delta_score, delta_pos = beta_score, beta_pos.copy()
                        beta_score, beta_pos = fitness, positions[i, :].copy()
                    elif fitness < delta_score:
                        delta_score, delta_pos = fitness, positions[i, :].copy()
            except Exception:
                continue
                
        # Update positions
        a = 2 - l * (2 / max_iterations)  # Linearly decreases from 2 to 0
        
        for i in range(num_agents):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha
                
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta
                
                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta
                
                positions[i, j] = (X1 + X2 + X3) / 3
        
        convergence_curve.append(alpha_score)
        
        if param_keys:
             print(f"GWO iter {l+1}/{max_iterations} | Score: {alpha_score:.6g}")
        else:
             print(f"GWO iter {l+1}/{max_iterations} | Score: {alpha_score:.6g}")

    return alpha_score, alpha_pos, convergence_curve, iteration_details


# --- Benchmark Functions ---

def rastrigin_function(x: np.ndarray) -> float:
    """
    Rastrigin benchmark function.
    Global minimum at x=0 with f(x)=0.
    Range: [-5.12, 5.12]
    """
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock_function(x: np.ndarray) -> float:
    """
    Rosenbrock benchmark function.
    Global minimum at x=1 with f(x)=0.
    Range: [-5, 10] or wider.
    """
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


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
    
    # Use adaptive NUM_GC_LAYERS range based on system size
    gc_layers_range = (model_config.get_num_gc_layers_range(num_buses)
                      if hasattr(model_config, 'get_num_gc_layers_range')
                      else model_config.NUM_GC_LAYERS_RANGE)
    
    param_bounds = {
        'HIDDEN_DIM': hidden_range, 
        'NUM_GC_LAYERS': gc_layers_range,
        # LEARNING_RATE removed from hyperparameter tuning - now controlled by adaptive scheduler
    }
    
    # The model automatically learns optimal loss weights via backpropagation
    
    # Add sequential model parameters with adaptive ranges
    if is_sequential: 
        param_bounds.update({
            'SEQUENCE_LENGTH': sequence_range, 
            'RNN_LAYERS': rnn_layers_range
        })
    
    # Add adaptive graph parameters
    if uses_adaptive_graph:
        # Use adaptive EMBEDDING_DIM range based on system size
        embedding_range = (model_config.get_embedding_dim_range(num_buses)
                          if hasattr(model_config, 'get_embedding_dim_range')
                          else model_config.EMBEDDING_DIM_RANGE)
        param_bounds.update({
            'EMBEDDING_DIM': embedding_range, 
            'PHI': model_config.PHI_RANGE
        })
    
    return param_bounds


def create_model_kwargs(model_config: Any, params: Dict[str, Any], num_buses: int, 
                       is_sequential: bool, uses_adaptive_graph: bool, model_name: str = None,
                       config: Any = None, normalizer: Any = None) -> Dict[str, Any]:
    """
    Create model keyword arguments from optimized parameters.
    Always assumes OPF mode.
    
    Args:
        model_config: Model configuration object
        params: Optimized parameters dictionary
        num_buses: Number of buses in the system
        is_sequential: Whether model is sequential
        uses_adaptive_graph: Whether model uses adaptive graph features
        model_name: Name of the model (for logging)
        config: Main config object (for generator constraints)
        normalizer: PowerSystemNormalizer (for generator constraints)
        
    Returns:
        Dictionary of model keyword arguments
    """
    input_dim = getattr(model_config, 'INPUT_DIM', 10)  # Default to 10 measurements
    model_kwargs = {
        'feature_dim': input_dim,  # Input dimension (10 measurements), NOT output dimension (2 voltages)
        'hidden_dim': int(params['HIDDEN_DIM']),
        'num_gc_layers': int(params['NUM_GC_LAYERS']),
        'num_buses': num_buses,
        'dropout': model_config.DROPOUT
    }
    
    # Add config and normalizer for generator constraints (if available)
    if config is not None:
        model_kwargs['config'] = config
    if normalizer is not None:
        model_kwargs['normalizer'] = normalizer
    
    if is_sequential:
        model_kwargs['rnn_layers'] = int(params['RNN_LAYERS'])
    
    if uses_adaptive_graph:
        model_kwargs.update({
            'embedding_dim': int(params['EMBEDDING_DIM']),
            'phi': float(params['PHI'])
        })
    
    # Always use heteroscedastic mode (no flag needed)
    
    # Twin heads removed: Not compatible with OPF mode (different bus types have different unknowns)
    
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
    Validates and ensures parameters are within valid ranges to prevent invalid model initialization.
    
    Args:
        param_keys: List of parameter names
        param_values: Array of parameter values
        
    Returns:
        Dictionary of processed parameters with validated values
    """
    params = {key: val for key, val in zip(param_keys, param_values)}
    
    # Define minimum values for each parameter type to prevent invalid model initialization
    min_values = {
        'HIDDEN_DIM': 16,      # Minimum hidden dimension for valid neural network layers
        'NUM_GC_LAYERS': 1,    # Must have at least 1 graph convolution layer
        'SEQUENCE_LENGTH': 1,  # Minimum sequence length
        'RNN_LAYERS': 1,       # Must have at least 1 RNN layer
        'EMBEDDING_DIM': 4,    # Minimum embedding dimension for adaptive graph
    }
    
    # Convert specific parameters to integers with validation
    for k in ['HIDDEN_DIM', 'NUM_GC_LAYERS', 'SEQUENCE_LENGTH', 'RNN_LAYERS', 'EMBEDDING_DIM']:
        if k in params:
            # Handle NaN, inf, and invalid values
            val = params[k]
            if np.isnan(val) or np.isinf(val) or val <= 0:
                # Use minimum value if invalid
                params[k] = min_values.get(k, 1)
            else:
                # Round and ensure minimum value
                rounded = int(round(val))
                params[k] = max(rounded, min_values.get(k, 1))
    
    # Validate PHI (mixing coefficient) - must be between 0 and 1
    if 'PHI' in params:
        phi_val = params['PHI']
        if np.isnan(phi_val) or np.isinf(phi_val):
            params['PHI'] = 0.5  # Default to balanced mixing
        else:
            params['PHI'] = np.clip(float(phi_val), 0.0, 1.0)
    
    # LEARNING_RATE validation removed - now controlled by adaptive scheduler (ReduceLROnPlateau)
    
    return params


def format_params_concise(params: Dict[str, Any]) -> str:
    """Format parameters dictionary concisely for display."""
    formatted_parts = []
    for k, v in params.items():
        if isinstance(v, (int, np.integer)):
            formatted_parts.append(f"{k}={v}")
        elif isinstance(v, (float, np.floating)):
            formatted_parts.append(f"{k}={_format_value(v)}")
        else:
            formatted_parts.append(f"{k}={v}")
    return ", ".join(formatted_parts)


def calculate_objective_score(metrics: Dict[str, float], config: Any, is_physics_informed: bool) -> float:
    """
    Calculate objective score for MoSOA optimization.
    
    CRITICAL: We MUST use Validation MSE (Denormalized, Physical Units) as the objective.
    We cannot use 'total_loss' because it contains learnable uncertainty weights (sigmas).
    If we optimize 'total_loss', MoSOA will pick hyperparameters that inflate sigma 
    to cheat the loss function, rather than improving accuracy.
    
    The loss function L = MSE/(2σ²) + log(σ) can be minimized by making σ→∞,
    which makes the first term go to zero while log(σ) grows slowly.
    This results in a numerically low "Total Loss" but terrible predictions.
    
    Validation MSE (denormalized) measures actual prediction error in physical units
    (per-unit for voltages/power, radians for angles), which is what we actually care about.
    
    Args:
        metrics: Dictionary of evaluation metrics from the VALIDATION set.
        config: Configuration object.
        is_physics_informed: Boolean flag (not used, but kept for compatibility).
        
    Returns:
        float: The objective score to minimize (Validation MSE in physical units).
    """
    # ALWAYS use the denormalized MSE (physical units) as the gold standard.
    # This allows fair comparison between PI and non-PI models, 
    # and between models with different learned uncertainties.
    
    if 'mse' in metrics:
        return metrics['mse']  # This is the denormalized MSE from evaluation.py
    elif 'mse_score' in metrics:
        return metrics['mse_score']
    else:
        # Fallback (should not happen if evaluation.py is correct)
        raise ValueError("Validation metrics missing 'mse'. Cannot compute objective. "
                        "Metrics keys: " + str(list(metrics.keys())))


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
    
    # Convert bounds to numpy arrays if they're lists or scalars (robustness)
    lower_bound = np.asarray(lower_bound, dtype=np.float64)
    upper_bound = np.asarray(upper_bound, dtype=np.float64)
    
    # Handle scalar bounds (expand to array)
    if lower_bound.ndim == 0:
        lower_bound = np.full(dim, lower_bound.item())
    if upper_bound.ndim == 0:
        upper_bound = np.full(dim, upper_bound.item())
    
    # Validate bounds
    if len(lower_bound) != dim or len(upper_bound) != dim:
        raise ValueError(f"Bounds dimension mismatch: lower_bound={len(lower_bound)}, upper_bound={len(upper_bound)}, dim={dim}")
    if np.any(lower_bound >= upper_bound):
        raise ValueError("Lower bounds must be strictly less than upper bounds")
    
    best_position = np.zeros(dim)
    best_score = float('inf')
    convergence_curve = []
    trial_details = []
    
    # Generate all trial positions at once
    if search_strategy == 'latin_hypercube':
        # Latin Hypercube Sampling for better space coverage
        positions = _latin_hypercube_sampling(num_trials, dim, lower_bound, upper_bound)
    else:
        # Random sampling (default) - vectorized
        lower_expanded = np.tile(lower_bound, (num_trials, 1))  # (num_trials, dim)
        upper_expanded = np.tile(upper_bound, (num_trials, 1))  # (num_trials, dim)
        positions = np.random.uniform(lower_expanded, upper_expanded, size=(num_trials, dim))
    
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
    # Generate Latin Hypercube samples in [0, 1] - vectorized
    samples = np.zeros((num_samples, dim))
    
    # Vectorized interval generation: (num_samples, dim)
    intervals_start = np.linspace(0, 1, num_samples + 1)[:-1]  # (num_samples,)
    intervals_end = np.linspace(0, 1, num_samples + 1)[1:]  # (num_samples,)
    
    # Generate samples for all dimensions at once
    for d in range(dim):
        # Randomly sample within each interval
        samples[:, d] = np.random.uniform(intervals_start, intervals_end)
        # Shuffle to avoid correlation between dimensions
        np.random.shuffle(samples[:, d])
    
    # Vectorized scaling to actual bounds: (num_samples, dim)
    bound_ranges = upper_bound - lower_bound  # (dim,)
    bound_ranges_expanded = np.tile(bound_ranges, (num_samples, 1))  # (num_samples, dim)
    lower_expanded = np.tile(lower_bound, (num_samples, 1))  # (num_samples, dim)
    samples = samples * bound_ranges_expanded + lower_expanded
    
    return samples
