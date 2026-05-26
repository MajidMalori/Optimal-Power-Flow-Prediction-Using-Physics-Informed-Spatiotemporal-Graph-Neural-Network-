from typing import Dict, Tuple

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


def _edge_set(state: BenchmarkState):
    return {(int(u), int(v)) for u, v in state.active_edges}


def _assess_feasibility(converged: bool, valid: bool, flags: Dict[str, bool]) -> Tuple[bool, float]:
    if not converged or not valid:
        return False, 0.0
    n = len(flags) if flags else 0
    if n == 0:
        return True, 1.0
    violated = sum(1 for v in flags.values() if v)
    is_feasible = violated == 0
    rate = 1.0 - (violated / n)
    return is_feasible, rate


def run_all_methods_for_state_feasibility(
    state: BenchmarkState,
    max_iter: int = 100,
    tolerance: float = 1e-5,
    load_network_fn=None,
    evaluator_cls=None,
    validate_outputs_fn=None,
    pred_vm=None,
    pred_va=None,
) -> Dict[str, SolverRunResult]:
    if load_network_fn is None:
        from src.processing.topology import load_network as load_network_fn
    if evaluator_cls is None:
        from src.benchmarks.warm_start_evaluator import WarmStartEvaluator as evaluator_cls
    if validate_outputs_fn is None:
        from src.processing.validation import validate_power_flow_outputs as validate_outputs_fn

    net = load_network_fn(state.case_name)
    evaluator = evaluator_cls(net=net, case_name=state.case_name, max_iter=max_iter, tolerance=tolerance)
    p_load, q_load, p_gen, p_ren, q_ren, vm_guess, va_guess = _features_to_arrays(state)
    if pred_vm is None:
        pred_vm = vm_guess
    if pred_va is None:
        pred_va = va_guess
    active_edges = _edge_set(state)
    target_vm = pred_vm.copy()
    target_va = pred_va.copy()

    # run base evaluator and request per-method solved nets
    raw = evaluator.evaluate_sample(
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
        include_nets=True,
    )

    out = {}
    method_map = {"flat": "flat", "dc": "dc", "results": "warmstart"}
    for raw_method, out_method in method_map.items():
        converged = bool(raw[raw_method]["success"])
        solved_net = raw[raw_method].get("net")
        valid = False
        flags = {}
        if converged and solved_net is not None:
            valid, _reason, flags = validate_outputs_fn(solved_net, {}, case_name=state.case_name)
        is_feasible, rate = _assess_feasibility(converged, valid, flags)
        out[out_method] = SolverRunResult(
            method=out_method,
            converged=converged,
            time_ms=float(raw[raw_method]["time_ms"]),
            iterations=int(raw[raw_method]["iterations"]),
            is_feasible=is_feasible,
            constraint_satisfaction_rate=rate,
        )
    return out
