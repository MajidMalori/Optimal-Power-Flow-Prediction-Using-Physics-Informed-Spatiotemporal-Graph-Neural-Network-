from typing import Dict, Set, Tuple

import numpy as np

from src.benchmarks.benchmark_state import BenchmarkState
from src.benchmarks.warmstart_protocol import SolverRunResult
from src.constants import FeatureIndices


def _features_to_arrays(state: BenchmarkState):
    x = np.array(state.features, dtype=float)
    p_load = x[:, FeatureIndices.P_LOAD]
    q_load = x[:, FeatureIndices.Q_LOAD]
    p_gen = x[:, FeatureIndices.P_CONV]
    p_ren = x[:, FeatureIndices.P_REN]
    q_ren = x[:, FeatureIndices.Q_REN]
    vm_guess = x[:, FeatureIndices.VM] + 1.0
    va_guess = x[:, FeatureIndices.VA]
    return p_load, q_load, p_gen, p_ren, q_ren, vm_guess, va_guess


def _edge_set(state: BenchmarkState) -> Set[Tuple[int, int]]:
    return {(int(u), int(v)) for u, v in state.active_edges}


def run_all_methods_for_state(
    state: BenchmarkState,
    max_iter: int = 100,
    tolerance: float = 1e-5,
    load_network_fn=None,
    evaluator_cls=None,
    pred_vm=None,
    pred_va=None,
) -> Dict[str, SolverRunResult]:
    """
    Real NR execution for speed benchmark using existing evaluator.
    Supports model prediction warmstart or falls back to feature-derived guess.
    """
    import time
    print(f"[DEBUG] speed_runtime: Starting for state {state.sample_id}")
    
    start_time = time.time()
    if load_network_fn is None:
        from src.processing.topology import load_network as load_network_fn
    if evaluator_cls is None:
        from src.benchmarks.warm_start_evaluator import WarmStartEvaluator as evaluator_cls

    print(f"[DEBUG] speed_runtime: Loading network...")
    net = load_network_fn(state.case_name)
    print(f"[DEBUG] speed_runtime: Creating evaluator...")
    evaluator = evaluator_cls(net=net, case_name=state.case_name, max_iter=max_iter, tolerance=tolerance)
    
    load_time = time.time() - start_time
    print(f"[DEBUG] speed_runtime: Setup completed in {load_time:.2f}s")
    p_load, q_load, p_gen, p_ren, q_ren, vm_guess, va_guess = _features_to_arrays(state)
    if pred_vm is None:
        pred_vm = vm_guess
    if pred_va is None:
        pred_va = va_guess
    active_edges = _edge_set(state)

    # Speed pillar focuses on convergence/time/iterations; targets are placeholders.
    target_vm = pred_vm.copy()
    target_va = pred_va.copy()

    print(f"[DEBUG] speed_runtime: Calling evaluator.evaluate_sample...")
    eval_start = time.time()
    out = evaluator.evaluate_sample(
        p_load=p_load,
        q_load=q_load,
        p_gen=p_gen,
        p_ren=p_ren,
        q_ren=q_ren,
        active_edges=active_edges,
        pred_vm=pred_vm,
        pred_va=pred_va,
        target_vm=target_vm,
        target_va=target_va,
    )
    eval_elapsed = time.time() - eval_start
    print(f"[DEBUG] speed_runtime: evaluator.evaluate_sample completed in {eval_elapsed:.2f}s")

    return {
        "flat": SolverRunResult("flat", bool(out["flat"]["success"]), float(out["flat"]["time_ms"]), int(out["flat"]["iterations"])),
        "dc": SolverRunResult("dc", bool(out["dc"]["success"]), float(out["dc"]["time_ms"]), int(out["dc"]["iterations"])),
        "warmstart": SolverRunResult(
            "warmstart",
            bool(out["results"]["success"]),
            float(out["results"]["time_ms"]),
            int(out["results"]["iterations"]),
        ),
    }
