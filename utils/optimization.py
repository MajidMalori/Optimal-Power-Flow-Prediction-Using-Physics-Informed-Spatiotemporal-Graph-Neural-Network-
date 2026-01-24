import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Callable, Any

def _format_value(val: float) -> str:
    if isinstance(val, (np.integer, int)): return str(int(val))
    if abs(val) < 0.01 or abs(val) > 1000: return f"{val:.5g}"
    return f"{val:.6g}"

def _init_positions(num_agents: int, dim: int, ub: np.ndarray, lb: np.ndarray) -> np.ndarray:
    if isinstance(ub, (int, float)): ub = np.full(dim, ub)
    if isinstance(lb, (int, float)): lb = np.full(dim, lb)
    return np.random.uniform(np.tile(lb, (num_agents, 1)), np.tile(ub, (num_agents, 1)), size=(num_agents, dim))

def mosoa_optimizer(num_agents: int, max_iter: int, lb: np.ndarray, ub: np.ndarray, dim: int, obj_func: Callable, param_keys: List[str] = None) -> Tuple[float, np.ndarray, List[float], List[Dict]]:
    lb, ub = np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64)
    if len(lb) != dim or len(ub) != dim: raise ValueError(f"Bounds mismatch: lb={len(lb)}, ub={len(ub)}, dim={dim}")
    if np.any(lb >= ub): raise ValueError("Lower bounds must be < upper bounds")
    
    best_pos, best_score = (lb + ub) / 2.0, float('inf')
    pos = _init_positions(num_agents, dim, ub, lb)
    curve, details = [], []
    v_max, v_min, u, w_max, w_min, beta_max, lambda_val, fc_min, fc_max = 1.0, 0.0, 1.0, 0.9, 0.2, 1.0, 2.0, 0.0, 2.0
    
    for l in range(max_iter):
        try:
            from utils.shutdown_flag import get_shutdown
            if get_shutdown(): raise KeyboardInterrupt("Shutdown signal")
        except (ImportError, AttributeError): pass
        if l > 0: print()
        
        fit = np.full(num_agents, np.inf)
        for i in range(num_agents):
            pos[i, :] = np.clip(pos[i, :], lb, ub)
            try:
                val = obj_func(pos[i, :])
                if np.isfinite(val):
                    fit[i] = val
                    if val < best_score: best_score, best_pos = val, pos[i, :].copy()
            except (ValueError, RuntimeError): pass

        f_max, f_min, f_avg, sigma = np.max(fit), np.min(fit), np.mean(fit), np.std(fit)
        M = 1.0 if (f_avg - f_min) == 0 else (f_max - f_avg) / (f_avg - f_min)
        fc = fc_min + M * (fc_max - fc_min) + (sigma * np.random.randn())
        A = fc * (1 - np.sin((np.pi / 2) * (l / max_iter)))
        v = v_max * np.tanh(np.abs(1 - l / max_iter))
        w = (w_max - w_min) * (1 - np.cos(np.pi / 2 * (l / max_iter))) + w_min
        beta = beta_max * np.exp(-lambda_val * (l / max_iter))
        
        B = 2 * (A**2) * np.random.rand(pos.shape[0])
        Ms = B[:, np.newaxis] * (best_pos - pos)
        r = u * np.exp(np.random.uniform(0, 2 * np.pi, size=pos.shape[0]) * v)
        spiral = np.abs(Ms) * r[:, np.newaxis] * np.cos(2 * np.pi * np.random.uniform(0, 2 * np.pi, size=pos.shape[0]))[:, np.newaxis]
        pert = beta * (pos[np.random.randint(0, pos.shape[0], size=pos.shape[0]), :] - pos)
        pos = spiral + (w * best_pos) + pert
        
        details.append({'iteration': l+1, 'best_score': best_score, 'best_position': best_pos.copy()})
        curve.append(best_score)
        
    return best_score, best_pos, curve, details

def setup_hyperparameter_bounds(model_name: str, cfg: Any, nb: int, is_pi: bool, is_seq: bool, use_adapt: bool) -> Dict[str, Tuple[float, float]]:
    hr = cfg.get_hidden_dim_range(nb) if hasattr(cfg, 'get_hidden_dim_range') else cfg.HIDDEN_DIM_RANGE
    sr, rr = (None, None)
    if is_seq:
        if hasattr(cfg, 'get_sequential_ranges'):
            r = cfg.get_sequential_ranges(nb)
            hr, sr, rr = r['hidden_dim'], r['sequence_length'], r['rnn_layers']
        else: sr, rr = cfg.SEQUENCE_LENGTH_RANGE, cfg.RNN_LAYERS_RANGE
    
    gcr = cfg.get_num_gc_layers_range(nb) if hasattr(cfg, 'get_num_gc_layers_range') else cfg.NUM_GC_LAYERS_RANGE
    bounds = {'HIDDEN_DIM': hr, 'NUM_GC_LAYERS': gcr}
    if is_seq: bounds.update({'SEQUENCE_LENGTH': sr, 'RNN_LAYERS': rr})
    if use_adapt:
        er = cfg.get_embedding_dim_range(nb) if hasattr(cfg, 'get_embedding_dim_range') else cfg.EMBEDDING_DIM_RANGE
        bounds.update({'EMBEDDING_DIM': er, 'PHI': cfg.PHI_RANGE})
    return bounds

def create_model_kwargs(cfg: Any, params: Dict[str, Any], nb: int, is_seq: bool, use_adapt: bool, model_name: str = None, config: Any = None, normalizer: Any = None, is_physics_informed: bool = False) -> Dict[str, Any]:
    kwargs = {'feature_dim': getattr(cfg, 'INPUT_DIM', 10), 'hidden_dim': int(params['HIDDEN_DIM']), 'num_gc_layers': int(params['NUM_GC_LAYERS']), 'num_buses': nb, 'dropout': cfg.DROPOUT}
    if is_physics_informed:
        if config: kwargs['config'] = config
        if normalizer: kwargs['normalizer'] = normalizer
    if is_seq: kwargs['rnn_layers'] = int(params['RNN_LAYERS'])
    if use_adapt: kwargs.update({'embedding_dim': int(params['EMBEDDING_DIM']), 'phi': float(params['PHI']), 'physics_informed': is_physics_informed, 'use_batch_norm': is_physics_informed})
    return kwargs

def generate_run_name(model: str, p: Dict[str, Any], nb: int, is_seq: bool) -> str:
    name = f"run_{model}_B{nb}_H{p.get('HIDDEN_DIM', 'N/A')}_GC{p.get('NUM_GC_LAYERS', 'N/A')}"
    return name + f"_SL{p.get('SEQUENCE_LENGTH', 'N/A')}_R{p.get('RNN_LAYERS', 'N/A')}" if is_seq else name

def process_optimization_params(keys: List[str], vals: np.ndarray) -> Dict[str, Any]:
    p = {k: v for k, v in zip(keys, vals)}
    mins = {'HIDDEN_DIM': 16, 'NUM_GC_LAYERS': 1, 'SEQUENCE_LENGTH': 1, 'RNN_LAYERS': 1, 'EMBEDDING_DIM': 4}
    for k in ['HIDDEN_DIM', 'NUM_GC_LAYERS', 'SEQUENCE_LENGTH', 'RNN_LAYERS', 'EMBEDDING_DIM']:
        if k in p:
            v = p[k]
            p[k] = mins.get(k, 1) if np.isnan(v) or np.isinf(v) or v <= 0 else max(int(round(v)), mins.get(k, 1))
    if 'PHI' in p: p['PHI'] = 0.5 if np.isnan(p['PHI']) or np.isinf(p['PHI']) else np.clip(float(p['PHI']), 0.0, 1.0)
    return p

def format_params_concise(params: Dict[str, Any]) -> str:
    return ", ".join([f"{k}={v}" if isinstance(v, (int, np.integer)) else f"{k}={_format_value(v)}" if isinstance(v, (float, np.floating)) else f"{k}={v}" for k, v in params.items()])

def calculate_objective_score(metrics: Dict[str, float], config: Any, is_pi: bool) -> float:
    if is_pi and 'total_loss' in metrics: return metrics['total_loss']
    if 'mse' in metrics: return metrics['mse']
    if 'mse_score' in metrics: return metrics['mse_score']
    raise ValueError(f"Missing mse in metrics: {list(metrics.keys())}")

def trial_based_search(trials: int, lb: np.ndarray, ub: np.ndarray, dim: int, obj_func: Callable, strategy: str = 'random') -> Tuple[float, np.ndarray, List[float], List[Dict]]:
    lb, ub = np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64)
    if lb.ndim == 0: lb = np.full(dim, lb.item())
    if ub.ndim == 0: ub = np.full(dim, ub.item())
    if len(lb) != dim or len(ub) != dim or np.any(lb >= ub): raise ValueError("Bounds mismatch")
    
    best_pos, best_score, curve, details = np.zeros(dim), float('inf'), [], []
    pos = _latin_hypercube_sampling(trials, dim, lb, ub) if strategy == 'latin_hypercube' else np.random.uniform(np.tile(lb, (trials, 1)), np.tile(ub, (trials, 1)), size=(trials, dim))
    
    pbar = tqdm(range(trials), desc="Trial Progress")
    for i in pbar:
        curr = np.clip(pos[i], lb, ub)
        try: score = obj_func(curr)
        except Exception as e: print(f"Trial {i+1} failed: {e}"); score = float('inf')
        
        if score < best_score: best_score, best_pos = score, curr.copy()
        curve.append(best_score)
        details.append({'trial': i+1, 'score': score, 'position': curr.copy(), 'is_best': (score==best_score)})
        pbar.set_description(f"Trial {i+1}/{trials} | Best: {best_score:.6f}")
        
    return best_score, best_pos, curve, details

def _latin_hypercube_sampling(n: int, dim: int, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    samples = np.zeros((n, dim))
    starts, ends = np.linspace(0, 1, n + 1)[:-1], np.linspace(0, 1, n + 1)[1:]
    for d in range(dim):
        samples[:, d] = np.random.uniform(starts, ends)
        np.random.shuffle(samples[:, d])
    return samples * np.tile(ub - lb, (n, 1)) + np.tile(lb, (n, 1))
