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
        random.shuffle(methods)

        for init_type in methods:
            net_test = copy.deepcopy(net)
            
            if init_type == "results":
                # 1. First, we must create/pre-allocate the res_bus table using a dummy run
                # max_iteration=0 creates the tables but doesn't solve.
                try:
                    pp.runpp(net_test, algorithm="nr", init="flat", max_iteration=0)
                except Exception: pass
                
                # 2. Inject prediction into the results table
                # Ensure we match by index
                slack_buses = set(net_test.ext_grid.bus.values)
                for i, bus_idx in enumerate(net_test.bus.index):
                    if bus_idx in slack_buses:
                        # Force Slack Bus to exact physical setpoints (reference)
                        # This prevents the solver from wasting iterations rotating the entire grid's phase
                        net_test.res_bus.loc[bus_idx, 'vm_pu'] = net_test.ext_grid.vm_pu.values[0] if len(net_test.ext_grid) > 0 else 1.0
                        net_test.res_bus.loc[bus_idx, 'va_degree'] = net_test.ext_grid.va_degree.values[0] if len(net_test.ext_grid) > 0 else 0.0
                    else:
                        net_test.res_bus.loc[bus_idx, 'vm_pu'] = pred_vm[i]
                        net_test.res_bus.loc[bus_idx, 'va_degree'] = np.rad2deg(pred_va[i])
                
                solver_init = "results" # Tells pandapower to use the res_bus table as starting point
            else:
                solver_init = init_type
                
            try:
                start_time = time.perf_counter()
                pp.runpp(net_test, algorithm="nr", init=solver_init, max_iteration=self.max_iter, tolerance_mva=self.tolerance)
                solve_time = (time.perf_counter() - start_time) * 1000  # ms
                
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

            results[init_type] = {
                "time_ms": solve_time,
                "iterations": iterations,
                "mae_vm": mae_vm,
                "success": success
            }
            if include_nets:
                results[init_type]["net"] = net_test if success else None
            
        return results
