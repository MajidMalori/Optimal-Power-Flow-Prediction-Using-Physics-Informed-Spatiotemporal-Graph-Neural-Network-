import time
import copy
import numpy as np
import pandapower as pp
from src.constants import SYSTEM_PHYSICS

class WarmStartEvaluator:
    """
    Evaluates pandapower's Newton-Raphson solver using different initializations.
    """
    def __init__(self, net, case_name, max_iter=100, tolerance=1e-5):
        self.base_net = net
        self.case_name = case_name
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.base_mva = SYSTEM_PHYSICS.get(case_name, SYSTEM_PHYSICS['default'])['base_mva']

    def evaluate_sample(
        self,
        p_load,
        q_load,
        p_gen,
        p_ren,
        q_ren,
        active_edges,
        pred_vm,
        pred_va,
        target_vm,
        target_va,
        include_nets: bool = False,
    ):
        """
        Runs NR solver with Flat, DC, and Neural Warm Start.
        """
        # Create isolated network for injection
        net = copy.deepcopy(self.base_net)
        
        # Override grid state from features to match the test sample
        if len(net.load) > 0 and p_load is not None:
            load_buses = net.load.bus.values
            net.load.p_mw = p_load[load_buses] * self.base_mva
            net.load.q_mvar = q_load[load_buses] * self.base_mva
            
        if len(net.gen) > 0 and p_gen is not None:
             gen_buses = net.gen.bus.values
             net.gen.p_mw = p_gen[gen_buses] * self.base_mva

        if len(net.sgen) > 0 and p_ren is not None and q_ren is not None:
             sgen_buses = net.sgen.bus.values
             net.sgen.p_mw = p_ren[sgen_buses] * self.base_mva
             net.sgen.q_mvar = q_ren[sgen_buses] * self.base_mva

        if active_edges is not None:
             for idx, row in net.line.iterrows():
                  u, v = int(row.from_bus), int(row.to_bus)
                  if (u, v) not in active_edges and (v, u) not in active_edges:
                       net.line.loc[idx, 'in_service'] = False

        results = {}
        # Avoid pandapower's noisy prints internally
        pp.set_user_pf_options(net, numba=False)

        methods = ["flat", "dc", "results"]
        # Randomize order to avoid library initialization bias appearing in the first method
        import random
        original_order = methods.copy()
        random.shuffle(methods)
        print(f"[DEBUG] warm_start_evaluator: Method order: {methods} (original: {original_order})")

        for i, init_type in enumerate(methods):
            print(f"[DEBUG] warm_start_evaluator: Running method {i+1}/3: {init_type}")
            net_test = copy.deepcopy(net)
            
            if init_type == "results":
                # Pandapower docs: for robust custom warm starts, prefer init="auto"
                # with init_vm_pu / init_va_degree arrays. Sanitize the neural guesses
                # to avoid non-physical starts (e.g., Vm==0 on unobserved buses).
                vm_init = np.asarray(pred_vm, dtype=float).copy()
                va_init_deg = np.rad2deg(np.asarray(pred_va, dtype=float).copy())

                # Replace invalid / extreme values with neutral fallback.
                vm_bad = (~np.isfinite(vm_init)) | (vm_init < 0.8) | (vm_init > 1.2)
                va_bad = ~np.isfinite(va_init_deg)
                vm_init[vm_bad] = 1.0
                va_init_deg[va_bad] = 0.0

                # Anchor slack buses to ext_grid setpoints.
                slack_buses = set(net_test.ext_grid.bus.values)
                for i, bus_idx in enumerate(net_test.bus.index):
                    if bus_idx in slack_buses:
                        vm_init[i] = net_test.ext_grid.vm_pu.values[0] if len(net_test.ext_grid) > 0 else 1.0
                        va_init_deg[i] = net_test.ext_grid.va_degree.values[0] if len(net_test.ext_grid) > 0 else 0.0

                solver_init = "auto"
            else:
                solver_init = init_type
                
            try:
                print(f"[DEBUG] warm_start_evaluator: Starting pp.runpp with init={solver_init}")
                start_time = time.perf_counter()
                
                # Add timeout protection around pandapower solver (Unix-only)
                import signal
                import multiprocessing
                
                has_alarm = hasattr(signal, 'alarm') and hasattr(signal, 'SIGALRM')
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"pandapower solver timed out after {self.max_iter * 2} seconds")
                
                if has_alarm:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(self.max_iter * 2)
                
                try:
                    if init_type == "results":
                        pp.runpp(
                            net_test,
                            algorithm="nr",
                            init=solver_init,
                            init_vm_pu=vm_init,
                            init_va_degree=va_init_deg,
                            max_iteration=self.max_iter,
                            tolerance_mva=self.tolerance,
                        )
                    else:
                        pp.runpp(net_test, algorithm="nr", init=solver_init, max_iteration=self.max_iter, tolerance_mva=self.tolerance)
                    
                    if has_alarm:
                        signal.alarm(0)
                    solve_time = (time.perf_counter() - start_time) * 1000  # ms
                    print(f"[DEBUG] warm_start_evaluator: pp.runpp completed in {solve_time:.2f}ms")
                    
                except TimeoutError as e:
                    solve_time = (time.perf_counter() - start_time) * 1000
                    print(f"[DEBUG] warm_start_evaluator: pp.runpp TIMED OUT after {solve_time:.2f}ms - {e}")
                    raise pp.LoadflowNotConverged(f"Solver timeout: {e}")
                
                iterations = net_test._ppc['iterations']
                
                final_vm = net_test.res_bus.sort_index().vm_pu.values
                final_va = np.deg2rad(net_test.res_bus.sort_index().va_degree.values)
                
                # Verify indices match for MAE calculation
                target_vm_sorted = target_vm[net_test.bus.index] if hasattr(target_vm, '__getitem__') else target_vm
                target_va_sorted = target_va[net_test.bus.index] if hasattr(target_va, '__getitem__') else target_va

                mae_vm = np.mean(np.abs(final_vm - target_vm))
                mae_va = np.mean(np.abs(final_va - target_va))
                
                success = True
            except pp.LoadflowNotConverged:
                solve_time = np.nan
                iterations = self.max_iter
                mae_vm = np.nan
                success = False
            except Exception:
                # Numerical failures (e.g., singular Jacobian factorization) should
                # be counted as non-converged samples, not crash the whole benchmark.
                solve_time = np.nan
                iterations = self.max_iter
                mae_vm = np.nan
                success = False

            results[init_type] = {
                "time_ms": solve_time,
                "iterations": iterations,
                "mae_vm": mae_vm,
                "success": success
            }
            if include_nets:
                results[init_type]["net"] = net_test if success else None
            
        return results
