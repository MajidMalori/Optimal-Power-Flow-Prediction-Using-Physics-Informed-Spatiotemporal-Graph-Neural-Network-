import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top
from src.constants import TargetIndices


def load_network(case_name: str) -> pp.pandapowerNet:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*dtypes could not be corrected.*')
        if case_name == "case33": return pn.case33bw()
        if case_name == "case57": return pn.case57()
        if case_name == "case118": return pn.case118()
    raise ValueError(f"Unknown test case: {case_name}")

def configure_renewables(net: pp.pandapowerNet, renewable_fraction_for_run: float, _config: dict = None) -> pp.pandapowerNet:
    num_buses = len(net.bus)
    num_renewables = int(num_buses * renewable_fraction_for_run)
    slack_buses = set(net.ext_grid.bus)
    possible_buses = list(set(net.bus.index) - slack_buses)
    
    net.sgen.drop(net.sgen.index, inplace=True)
    
    if not possible_buses or num_renewables == 0:
        return net
    
    num_renewables = min(num_renewables, len(possible_buses))
    renewable_buses = np.random.choice(possible_buses, size=num_renewables, replace=False)
    
    if 'type' not in net.sgen.columns: net.sgen['type'] = pd.Series(dtype=str)
        
    for bus_idx in renewable_buses:
        gen_type = np.random.choice(['solar', 'wind'])
        pp.create_sgen(net, bus=bus_idx, p_mw=0, q_mvar=0, name=f"{gen_type.capitalize()}@{bus_idx}", type=gen_type)
        
    return net

def apply_configuration_switch(net: pp.pandapowerNet) -> dict:
    """
    Simulates a 'Configuration Change' (Switching Event) for both radial and meshed grids.
    1. Identifies tie-lines (Normally Open).
    2. If no NO lines exist (Case 57/118), it creates one by opening a line in a cycle.
    3. Closes a NO line to create a loop.
    4. Finds a different line in that loop and opens it.
    """
    # 1. Identify Tie-Lines (Normally Open)
    no_lines = net.line.index[~net.line.in_service].values
    
    # 2. If no NO lines exist, we must create one by opening a cycle
    # This prevents the 'straight line' topology on meshed systems like Case 57
    if len(no_lines) == 0:
        try:
            g = top.create_nxgraph(net, include_trafos=False)
            simple_g = nx.Graph(g)
            loops = nx.cycle_basis(simple_g)
            if not loops: return None
            
            # Pick a random loop and open a line in it
            loop_nodes = loops[np.random.randint(len(loops))]
            u, v = loop_nodes[0], loop_nodes[1]
            mask = ((net.line.from_bus == u) & (net.line.to_bus == v)) | \
                   ((net.line.from_bus == v) & (net.line.to_bus == u))
            indices = net.line.index[mask].values
            if len(indices) == 0: return None
            
            line_to_open = indices[0]
            net.line.loc[line_to_open, 'in_service'] = False
            # When a normally open (NO) line is available, return it as the switching action
            return {'closed_idx': int(line_to_open), 'opened_idx': int(line_to_open)} # Self-loop switch
        except Exception: return None

    line_to_close = np.random.choice(no_lines)
    net.line.loc[line_to_close, 'in_service'] = True
    
    # 3. Find the resulting loop using NetworkX
    try:
        g = top.create_nxgraph(net, include_trafos=True)
        # convert to simple graph for cycle_basis
        simple_g = nx.Graph(g)
        loops = list(nx.cycle_basis(simple_g))
        if not loops:
            net.line.loc[line_to_close, 'in_service'] = False
            return None
            
        loop_nodes = loops[0] # Take the first loop formed
        
        # Find all lines that connect nodes in this loop
        loop_lines = []
        for i in range(len(loop_nodes)):
            u, v = loop_nodes[i], loop_nodes[(i+1)%len(loop_nodes)]
            # Find the line index connecting u and v
            mask = ((net.line.from_bus == u) & (net.line.to_bus == v)) | \
                   ((net.line.from_bus == v) & (net.line.to_bus == u))
            indices = net.line.index[mask].values
            if len(indices) > 0:
                loop_lines.extend(indices)
        
        # 4. Filter: Avoid opening the same line that was just closed to ensure a topology change
        possible_to_open = [idx for idx in loop_lines if idx != line_to_close]
        
        if not possible_to_open:
            net.line.loc[line_to_close, 'in_service'] = False
            return None
            
        line_to_open = np.random.choice(possible_to_open)
        
        # 5. Open the second line
        net.line.loc[line_to_open, 'in_service'] = False
        
        return {
            'closed_idx': int(line_to_close),
            'opened_idx': int(line_to_open)
        }
    except Exception:
        # Emergency rollback
        net.line.loc[line_to_close, 'in_service'] = False
        return None

def restore_configuration(net: pp.pandapowerNet, switch_info: dict):
    """Reverts a switching event back to the base configuration."""
    if switch_info:
        net.line.loc[switch_info['closed_idx'], 'in_service'] = False
        net.line.loc[switch_info['opened_idx'], 'in_service'] = True

def calculate_ybus_from_net(net: pp.pandapowerNet) -> np.ndarray:
    if net._ppc is None or 'internal' not in net._ppc or 'Ybus' not in net._ppc['internal']:
        try:
            pp.runpp(net, algorithm='nr', calculate_voltage_angles=True)
        except Exception: # Catch any exception from runpp
            # If runpp fails, try to manually create Ybus
            from pandapower.pd2ppc import _pd2ppc
            from pandapower.pf.makeYbus import makeYbus
            
            ppc, ppci = _pd2ppc(net)
            Ybus, _, _ = makeYbus(ppci["baseMVA"], ppci["bus"], ppci["branch"])
            net._ppc['internal']['Ybus'] = Ybus

    ppc = net._ppc
    ybus_int = ppc['internal']['Ybus']
    bus_lookup = net._pd2ppc_lookups['bus']
    n_bus = len(net.bus)
    
    perm = np.zeros(n_bus, dtype=int)
    for ext_idx in net.bus.index:
        if ext_idx in bus_lookup:
            perm[ext_idx] = bus_lookup[ext_idx]
            
    ybus_int_dense = ybus_int if isinstance(ybus_int, np.ndarray) else ybus_int.toarray()
    return ybus_int_dense[np.ix_(perm, perm)]

def calculate_adjacency_matrix(net: pp.pandapowerNet) -> np.ndarray:
    num_buses = len(net.bus)
    adj_matrix = np.zeros((num_buses, num_buses), dtype=np.float32)
    active_lines = net.line[net.line.in_service]
    
    for _, line in active_lines.iterrows():
        from_bus, to_bus = int(line['from_bus']), int(line['to_bus'])
        adj_matrix[from_bus, to_bus] = 1.0
        adj_matrix[to_bus, from_bus] = 1.0
    
    return adj_matrix

def identify_bus_types(net: pp.pandapowerNet) -> np.ndarray:
    num_buses = len(net.bus)
    bus_types = np.zeros(num_buses, dtype=np.int32)
    
    slack_buses = set(net.ext_grid.bus.values)
    for bus_idx in slack_buses:
        bus_types[bus_idx] = 2
    
    gen_buses = set(net.gen.bus.values)
    for bus_idx in gen_buses:
        if bus_idx not in slack_buses:
            bus_types[bus_idx] = 1
    
    return bus_types

def create_opf_targets(net: pp.pandapowerNet, _bus_types: np.ndarray) -> np.ndarray:
    num_buses = len(net.bus)
    targets = np.zeros((num_buses, 10), dtype=np.float32)
    
    load_p = net.res_load.groupby(net.load.bus).p_mw.sum().reindex(net.bus.index, fill_value=0).values
    load_q = net.res_load.groupby(net.load.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0).values
    ext_p = net.res_ext_grid.groupby(net.ext_grid.bus).p_mw.sum().reindex(net.bus.index, fill_value=0).values
    ext_q = net.res_ext_grid.groupby(net.ext_grid.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0).values
    gen_p = net.res_gen.groupby(net.gen.bus).p_mw.sum().reindex(net.bus.index, fill_value=0).values
    gen_q = net.res_gen.groupby(net.gen.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0).values
    sgen_p = net.res_sgen.groupby(net.sgen.bus).p_mw.sum().reindex(net.bus.index, fill_value=0).values
    sgen_q = net.res_sgen.groupby(net.sgen.bus).q_mvar.sum().reindex(net.bus.index, fill_value=0).values
    
    targets[:, TargetIndices.P_LOAD] = load_p
    targets[:, TargetIndices.Q_LOAD] = load_q
    targets[:, TargetIndices.P_EXT_GRID] = ext_p
    targets[:, TargetIndices.Q_EXT_GRID] = ext_q
    targets[:, TargetIndices.P_CONV] = gen_p
    targets[:, TargetIndices.Q_CONV] = gen_q
    targets[:, TargetIndices.P_REN] = sgen_p
    targets[:, TargetIndices.Q_REN] = sgen_q
    targets[:, TargetIndices.VM] = net.res_bus.vm_pu.values
    targets[:, TargetIndices.VA] = np.deg2rad(net.res_bus.va_degree.values)
    
    return targets
