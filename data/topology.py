import numpy as np
import pandas as pd
import warnings
import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top
import networkx as nx

def load_network(case_name: str) -> pp.pandapowerNet:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*dtypes could not be corrected.*')
        if case_name == "case33": return pn.case33bw()
        if case_name == "case57": return pn.case57()
        if case_name == "case118": return pn.case118()
    raise ValueError(f"Unknown test case: {case_name}")

def configure_renewables(net: pp.pandapowerNet, renewable_fraction_for_run: float, config: dict) -> pp.pandapowerNet:
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

def apply_n1_contingency(net: pp.pandapowerNet) -> int:
    active_lines = net.line.index[net.line.in_service]
    if not active_lines.any(): return None
    
    for line_to_drop in np.random.permutation(active_lines.values):
        net.line.loc[line_to_drop, 'in_service'] = False
        if nx.is_connected(top.create_nxgraph(net, include_trafos=True)):
            return line_to_drop
        net.line.loc[line_to_drop, 'in_service'] = True
    return None

def restore_contingency(net: pp.pandapowerNet, dropped_line_idx: int):
    if dropped_line_idx is not None:
        net.line.loc[dropped_line_idx, 'in_service'] = True

def calculate_ybus_from_net(net: pp.pandapowerNet) -> np.ndarray:
    if net._ppc is None or 'internal' not in net._ppc or 'Ybus' not in net._ppc['internal']:
        try:
            pp.runpp(net, algorithm='nr', calculate_voltage_angles=True)
        except:
            from pandapower.pd2ppc import _pd2ppc
            from pandapower.pf.makeYbus import makeYbus
            _pd2ppc(net)
            baseMVA, bus, gen, branch = net._ppc["baseMVA"], net._ppc["bus"], net._ppc["gen"], net._ppc["branch"]
            Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
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

def create_opf_targets(net: pp.pandapowerNet, bus_types: np.ndarray) -> np.ndarray:
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
    
    targets[:, 0] = load_p
    targets[:, 1] = load_q
    targets[:, 2] = ext_p
    targets[:, 3] = ext_q
    targets[:, 4] = gen_p
    targets[:, 5] = gen_q
    targets[:, 6] = sgen_p
    targets[:, 7] = sgen_q
    targets[:, 8] = net.res_bus.vm_pu.values
    targets[:, 9] = np.deg2rad(net.res_bus.va_degree.values)
    
    return targets
