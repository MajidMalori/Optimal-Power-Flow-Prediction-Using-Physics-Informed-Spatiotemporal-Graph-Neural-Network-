import pytest
import torch
import numpy as np
from src.models.physics_loss import PhysicsLoss

@pytest.fixture
def dummy_physics_data():
    num_nodes = 5
    num_branches = 4
    batch_size = 2
    
    # Dummy Ybus (Zeroed for pure balance check)
    ybus = torch.zeros(num_nodes, num_nodes, dtype=torch.complex128)
    
    # Branch data
    branch_from = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    branch_to = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    branch_max_s_pu = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    
    # Predictions (all 0 deviation -> perfectly 1.0 p.u.)
    vm_pred = torch.zeros(batch_size, num_nodes)
    va_pred = torch.zeros(batch_size, num_nodes)
    
    # Targets
    # Index 0: P_load, 1: Q_load, 2: P_gen, 3: Q_gen, 4: P_ext, 5: Q_ext, 6: P_ren, 7: Q_ren
    # P_net = P_gen + P_ext + P_ren - P_load
    targets = torch.zeros(batch_size, num_nodes, 10)
    # Perfectly balance everything at 0
    
    topology_ids = torch.zeros(batch_size, dtype=torch.long)
    
    return {
        "ybus": ybus,
        "branch_from": branch_from,
        "branch_to": branch_to,
        "branch_max_s_pu": branch_max_s_pu,
        "vm_pred": vm_pred,
        "va_pred": va_pred,
        "targets": targets,
        "topology_ids": topology_ids,
        "num_nodes": num_nodes,
        "num_branches": num_branches,
        "batch_size": batch_size
    }

def test_physics_evaluation_metrics_all_pass(dummy_physics_data):
    physics = PhysicsLoss(
        ybus=dummy_physics_data["ybus"],
        branch_from=dummy_physics_data["branch_from"],
        branch_to=dummy_physics_data["branch_to"],
        branch_max_s_pu=dummy_physics_data["branch_max_s_pu"],
        v_min=0.95,
        v_max=1.05
    )
    
    res = physics.evaluate_constraints(
        vm_pred=dummy_physics_data["vm_pred"],
        va_pred=dummy_physics_data["va_pred"],
        targets=dummy_physics_data["targets"],
        topology_ids=dummy_physics_data["topology_ids"],
        p_tol=0.01
    )
    
    # Since everything is 0/1.0, everything should be satisfied
    assert res["p_satisfied"] == dummy_physics_data["batch_size"] * dummy_physics_data["num_nodes"]
    assert res["v_satisfied"] == dummy_physics_data["batch_size"] * dummy_physics_data["num_nodes"]
    assert res["s_satisfied"] == dummy_physics_data["batch_size"] * dummy_physics_data["num_branches"]
    assert res["feasible_samples"] == dummy_physics_data["batch_size"]

def test_physics_evaluation_metrics_voltage_violation(dummy_physics_data):
    physics = PhysicsLoss(
        ybus=dummy_physics_data["ybus"],
        branch_from=dummy_physics_data["branch_from"],
        branch_to=dummy_physics_data["branch_to"],
        branch_max_s_pu=dummy_physics_data["branch_max_s_pu"],
        v_min=0.95,
        v_max=1.05
    )
    
    # Force a voltage violation (1.1 p.u. -> 0.1 deviation)
    vm_pred = dummy_physics_data["vm_pred"].clone()
    vm_pred[0, 0] = 0.1 
    
    res = physics.evaluate_constraints(
        vm_pred=vm_pred,
        va_pred=dummy_physics_data["va_pred"],
        targets=dummy_physics_data["targets"],
        topology_ids=dummy_physics_data["topology_ids"],
        p_tol=0.01
    )
    
    # 1 node in 1 sample violates voltage
    assert res["v_satisfied"] == (dummy_physics_data["batch_size"] * dummy_physics_data["num_nodes"]) - 1
    # Sample 0 is no longer feasible
    assert res["feasible_samples"] == dummy_physics_data["batch_size"] - 1

def test_physics_evaluation_metrics_power_violation(dummy_physics_data):
    physics = PhysicsLoss(
        ybus=dummy_physics_data["ybus"],
        branch_from=dummy_physics_data["branch_from"],
        branch_to=dummy_physics_data["branch_to"],
        branch_max_s_pu=dummy_physics_data["branch_max_s_pu"]
    )
    
    # Force a power violation (Target says load=1.0, Gen=0.0 -> residual=1.0)
    targets = dummy_physics_data["targets"].clone()
    targets[1, 2, 0] = 1.0 # Sample 1, Node 2, P_load = 1.0
    
    res = physics.evaluate_constraints(
        vm_pred=dummy_physics_data["vm_pred"],
        va_pred=dummy_physics_data["va_pred"],
        targets=targets,
        topology_ids=dummy_physics_data["topology_ids"],
        p_tol=0.01
    )
    
    # P balance fails for 1 node
    assert res["p_satisfied"] == (dummy_physics_data["batch_size"] * dummy_physics_data["num_nodes"]) - 1
    # Sample 1 is no longer feasible
    assert res["feasible_samples"] == dummy_physics_data["batch_size"] - 1
