import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top
import networkx as nx
from scipy import sparse

def load_network(case_name: str) -> pp.pandapowerNet:
    """Loads a pandapower network based on its name."""
    print(f"\n----- Loading Base Test Case: {case_name} -----")
    if case_name == "case33": return pn.case33bw()
    if case_name == "case57": return pn.case57()
    if case_name == "case118": return pn.case118()
    raise ValueError(f"Unknown test case: {case_name}")

def configure_renewables(net: pp.pandapowerNet, renewable_fraction_for_run: float, config: dict) -> pp.pandapowerNet:
    """Adds renewable static generators (sgen) to the network for a specific fraction."""
    num_buses = len(net.bus)
    num_renewables = int(num_buses * renewable_fraction_for_run)
    slack_buses = set(net.ext_grid.bus)
    possible_buses = list(set(net.bus.index) - slack_buses)
    
    net.sgen.drop(net.sgen.index, inplace=True)
    
    if len(possible_buses) < num_renewables:
        print(f"Warning: Not enough non-slack buses. Using {len(possible_buses)} of {num_renewables} requested.")
        num_renewables = len(possible_buses)
        
    if num_renewables == 0:
        print("Configuring network with 0 renewable generators.")
        return net

    renewable_buses = np.random.choice(possible_buses, size=num_renewables, replace=False)
    
    if 'type' not in net.sgen.columns: net.sgen['type'] = pd.Series(dtype=str)
        
    for bus_idx in renewable_buses:
        gen_type = np.random.choice(['solar', 'wind'])
        pp.create_sgen(net, bus=bus_idx, p_mw=0, q_mvar=0, name=f"{gen_type.capitalize()}@{bus_idx}", type=gen_type)
        
    print(f"Configured {len(net.sgen)} renewable generators for a {renewable_fraction_for_run*100:.0f}% fraction.")
    return net

def apply_n1_contingency(net: pp.pandapowerNet) -> int:
    """Randomly takes one active line out of service if it doesn't cause islanding."""
    active_lines = net.line.index[net.line.in_service]
    if not active_lines.any(): return None
    
    for line_to_drop in np.random.permutation(active_lines.values):
        net.line.loc[line_to_drop, 'in_service'] = False
        if nx.is_connected(top.create_nxgraph(net, include_trafos=True)):
            return line_to_drop
        net.line.loc[line_to_drop, 'in_service'] = True
    return None

def restore_contingency(net: pp.pandapowerNet, dropped_line_idx: int):
    """Restores a line that was previously taken out of service."""
    if dropped_line_idx is not None:
        net.line.loc[dropped_line_idx, 'in_service'] = True


def calculate_ybus_from_net(net: pp.pandapowerNet) -> np.ndarray:
    """
    Extract Ybus matrix directly from Pandapower's internal representation.
    This is the PROVEN method from validate_simple.py that passes all tests.
    
    CRITICAL: Returns Ybus in PER-UNIT system (p.u.), not physical units (Siemens).
    This is required for correct physics-informed loss calculation.

    Args:
        net: The pandapower network object (AFTER power flow).

    Returns:
        A dense numpy array representing the Ybus matrix in per-unit, ordered by external bus indices (0 to N-1).
    """
    # Ensure power flow has been run and _ppc exists
    if net._ppc is None or 'internal' not in net._ppc or 'Ybus' not in net._ppc['internal']:
        try:
            pp.runpp(net, algorithm='nr', calculate_voltage_angles=True)
        except:
            # Fallback: manually build _ppc if power flow fails
            from pandapower.pd2ppc import _pd2ppc
            from pandapower.pf.makeYbus import makeYbus
            _pd2ppc(net)
            baseMVA, bus, gen, branch = net._ppc["baseMVA"], net._ppc["bus"], net._ppc["gen"], net._ppc["branch"]
            Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
            net._ppc['internal']['Ybus'] = Ybus

    # Extract Ybus from Pandapower's internal representation
    ppc = net._ppc
    ybus_int = ppc['internal']['Ybus']
    
    # Map external bus indices to internal indices
    bus_lookup = net._pd2ppc_lookups['bus']
    n_bus = len(net.bus)
    
    # Create permutation array to reorder from internal to external indices
    perm = np.zeros(n_bus, dtype=int)
    for ext_idx in net.bus.index:
        if ext_idx in bus_lookup:
            perm[ext_idx] = bus_lookup[ext_idx]
            
    # Convert sparse to dense if needed
    if isinstance(ybus_int, np.ndarray):
        ybus_int_dense = ybus_int
    else:
        ybus_int_dense = ybus_int.toarray()
        
    # Reorder to external bus indices (0 to N-1)
    ybus = ybus_int_dense[np.ix_(perm, perm)]
    
    # print(f"  [DEBUG] Pandapower Ybus extraction: shape={ybus.shape}, diagonal[0]={ybus[0,0]:.6f}")
    
    return ybus


def calculate_adjacency_matrix(net: pp.pandapowerNet) -> np.ndarray:
    """Calculate adjacency matrix from network topology."""
    num_buses = len(net.bus)
    adj_matrix = np.zeros((num_buses, num_buses), dtype=np.float32)
    
    # Add edges from lines
    for _, line in net.line.iterrows():
        from_bus = int(line['from_bus'])
        to_bus = int(line['to_bus'])
        adj_matrix[from_bus, to_bus] = 1.0
        adj_matrix[to_bus, from_bus] = 1.0
    
    return adj_matrix

def identify_bus_types(net: pp.pandapowerNet) -> np.ndarray:
    """
    Identify bus types for Optimal Power Flow (OPF) from pandapower network state.
    Bus types are determined AFTER power flow solution (pandapower decides).
    
    - Slack bus: Bus with ext_grid (reference bus, V and θ known/specified)
    - PV bus: Bus with gen (generator with voltage control, V known, P specified)
    - PQ bus: Bus with load or sgen only (load bus, V and θ unknown)
    
    Note: A bus type can change dynamically (e.g., if gen hits Q limits, becomes PQ),
    but we use static classification based on network elements for simplicity.
    
    Returns:
        bus_types: Array of bus type codes [0=PQ, 1=PV, 2=Slack] for each bus
    """
    num_buses = len(net.bus)
    bus_types = np.zeros(num_buses, dtype=np.int32)  # Default: PQ bus
    
    # Identify slack buses (external grid) - these are always slack
    slack_buses = set(net.ext_grid.bus.values)
    for bus_idx in slack_buses:
        bus_types[bus_idx] = 2  # Slack bus
    
    # Identify PV buses (conventional generators with voltage control)
    # PV buses have gen connected (not ext_grid, not just load/sgen)
    # but we use static classification for training data consistency
    gen_buses = set(net.gen.bus.values)
    for bus_idx in gen_buses:
        if bus_idx not in slack_buses:  # Don't override slack
            bus_types[bus_idx] = 1  # PV bus
    
    return bus_types

def create_opf_targets(net: pp.pandapowerNet, bus_types: np.ndarray) -> np.ndarray:
    """
    Create OPF-style targets based on bus type (predict only unknowns):
    - PQ bus: Predict [V, θ] (unknowns)
    - PV bus: Predict [Q, θ] (unknowns, V is known/specified)
    - Slack bus: Predict [P, Q] (unknowns, V and θ are known/specified)
    
    **UNIT CONSISTENCY FIX (Option A):**
    - V: In per-unit (vm_pu) - standard for power systems
    - θ: In radians - standard for power systems  
    - P, Q: In MW/MVAR (NOT per-unit) - for consistency with features
    
    This ensures features and targets use the same units for power,
    making normalization consistent and avoiding information loss.
    
    Args:
        net: Pandapower network AFTER power flow solution
        bus_types: Array of bus type codes [0=PQ, 1=PV, 2=Slack] from identify_bus_types()
    
    Returns:
        targets: Array [num_buses, 2] with unknowns for each bus
    """
    num_buses = len(net.bus)
    targets = np.zeros((num_buses, 2), dtype=np.float64)
    
    # Get power flow results
    vm_pu = net.res_bus.vm_pu.values
    va_rad = np.deg2rad(net.res_bus.va_degree.values)
    
    # Get power injections (net injection = generation - load)
    ext_grid_p_by_bus = net.res_ext_grid.groupby(net.ext_grid.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)
    ext_grid_q_by_bus = net.res_ext_grid.groupby(net.ext_grid.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0)
    gen_p_by_bus = net.res_gen.groupby(net.gen.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)
    gen_q_by_bus = net.res_gen.groupby(net.gen.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0)
    sgen_p_by_bus = net.res_sgen.groupby(net.sgen.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)
    sgen_q_by_bus = net.res_sgen.groupby(net.sgen.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0)
    load_p_by_bus = net.res_load.groupby(net.load.bus).p_mw.sum().reindex(net.bus.index, fill_value=0)
    load_q_by_bus = net.res_load.groupby(net.load.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0)
    
    # Net power injection at each bus (generation - load) in MW/MVAR
    p_inj_mw = (ext_grid_p_by_bus + gen_p_by_bus + sgen_p_by_bus - load_p_by_bus).values
    q_inj_mvar = (ext_grid_q_by_bus + gen_q_by_bus + sgen_q_by_bus - load_q_by_bus).values
    
    # **UNIT CONSISTENCY FIX**: Store power in MW/MVAR (same as features)
    # Previously converted to p.u., causing unit mismatch with features
    # Now both features and targets use MW/MVAR for power values
    
    # Create targets based on bus type (only unknowns)
    for bus_idx in range(num_buses):
        if bus_types[bus_idx] == 0:  # PQ bus: unknowns = [V, θ]
            targets[bus_idx, 0] = vm_pu[bus_idx]  # Voltage magnitude in per-unit
            targets[bus_idx, 1] = va_rad[bus_idx]  # Voltage angle in radians
        elif bus_types[bus_idx] == 1:  # PV bus: unknowns = [Q, θ]
            targets[bus_idx, 0] = q_inj_mvar[bus_idx]  # Reactive power in MVAR (not p.u.)
            targets[bus_idx, 1] = va_rad[bus_idx]  # Voltage angle in radians
        else:  # Slack bus: unknowns = [P, Q]
            targets[bus_idx, 0] = p_inj_mw[bus_idx]  # Active power in MW (not p.u.)
            targets[bus_idx, 1] = q_inj_mvar[bus_idx]  # Reactive power in MVAR (not p.u.)
    
    return targets
