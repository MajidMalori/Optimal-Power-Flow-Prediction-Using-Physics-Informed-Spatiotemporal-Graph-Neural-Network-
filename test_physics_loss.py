import torch
import numpy as np
import os
import sys

# Ensure the script can find project modules
# This allows you to run the script from the MOOPS directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.data_loader import load_power_system_data, create_data_loaders
from utils.metrics import PowerSystemLoss
from config import Config

def run_physics_loss_test():
    """
    Tests if the ground-truth data perfectly satisfies the physics constraints.
    It simulates a perfect model prediction by setting `prediction = target`
    and expects the physics-based loss components to be zero.
    """
    print("="*80)
    print("STARTING PHYSICS-INFORMED LOSS VERIFICATION TEST")
    print("="*80)

    # --- 1. Setup Configuration ---
    # We use a small test case and a small batch size for speed.
    test_case = "case33"
    config = Config()
    config.BATCH_SIZE = 16
    config.NUM_BUSES = 33 # Explicitly set for the test case
    
    # These lambda values are used as weights in the loss function
    config.LAMBDA_P = 10.0 
    config.LAMBDA_V = 10.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Data ---
    # This step verifies that the data loading pipeline correctly handles the
    # new time-varying Ybus matrices.
    try:
        print(f"\n[Step 1/4] Loading data for '{test_case}'...")
        data_tuple = load_power_system_data(config, test_case)
        features, adjacency, ybus_matrices, targets, energy_coeffs, carbon_coeffs, normalizer = data_tuple
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"\n[CRITICAL ERROR] Data files not found: {e}")
        print("Please run 'python data/gen_meas_best.py' first to generate the dataset.")
        return
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred during data loading: {e}")
        return

    # --- 3. Create DataLoader ---
    # We only need one batch to perform the test.
    print("\n[Step 2/4] Creating data loader...")
    # We test with is_static=True, as it's the simplest case (1-to-1 mapping)
    loaders = create_data_loaders(features, adjacency, ybus_matrices, targets, energy_coeffs, carbon_coeffs, config, is_static=True)
    _, _, test_loader = loaders
    
    try:
        batch = next(iter(test_loader))
        print("Successfully fetched one batch of data.")
    except StopIteration:
        print("\n[CRITICAL ERROR] DataLoader is empty. Cannot perform test.")
        return

    # --- 4. Instantiate Loss Function and Run Test ---
    print("\n[Step 3/4] Initializing the loss function...")
    criterion = PowerSystemLoss(config=config, normalizer=normalizer, is_gcn=False).to(device)
    print("Loss function initialized.")

    print("\n[Step 4/4] Performing the physics validation...")
    
    # Move all necessary data from the batch to the correct device
    targets_norm = batch['targets'].to(device)
    ybus_batch = batch['ybus_matrix'].to(device)
    
    # THE CORE OF THE TEST: Simulate a perfect prediction
    # If the model's output is identical to the target, the loss should only
    # come from physical inconsistencies in the target data itself.
    predictions_norm = targets_norm.clone()

    # Denormalize to get physical values, which are used for physics calculations
    num_buses = config.NUM_BUSES
    predictions_phys = normalizer.denormalize(predictions_norm, num_buses)

    # --- Calculate individual loss components ---
    # We call the internal methods of the loss class to inspect each part.
    
    # a) Data Loss (MSE)
    data_loss = torch.nn.functional.mse_loss(predictions_norm, targets_norm)

    # b) Physics Loss: Power Balance Violation (returns a tensor of shape [batch_size])
    power_balance_violation = criterion._compute_power_balance_violation(predictions_phys, ybus_batch)

    # c) Physics Loss: Voltage Limit Violation (returns a tensor of shape [batch_size])
    voltage_limit_violation = criterion._compute_voltage_limit_violation(predictions_phys)

    # --- 5. Report Results ---
    print("\n" + "-"*40)
    print("TEST RESULTS")
    print("-"*40)

    # --- START CORRECTION: Aggregate batch results before checking ---
    # Aggregate the batch violations into a single scalar value (mean) for reporting.
    mean_power_balance_violation = torch.mean(power_balance_violation)
    mean_voltage_limit_violation = torch.mean(voltage_limit_violation)

    # Check Data Loss
    print(f"1. Data Loss (MSE): {data_loss.item():.6e}")
    if torch.allclose(data_loss, torch.tensor(0.0)):
        print("   [PASS] Data loss is zero, as expected for a perfect prediction.")
    else:
        print("   [FAIL] Data loss is non-zero. This should not happen.")

    # Check Power Balance Loss
    print(f"2. Mean Power Balance Violation: {mean_power_balance_violation.item():.6e}")
    if torch.allclose(mean_power_balance_violation, torch.tensor(0.0), atol=1e-6):
        print("   [PASS] Power balance is satisfied. The Ybus matrix is correctly synchronized with the target state.")
    else:
        print("   [FAIL] Power balance is violated. There is a mismatch between the target state and the provided Ybus matrix.")

    # Check Voltage Limit Loss
    print(f"3. Mean Voltage Limit Violation: {mean_voltage_limit_violation.item():.6e}")
    if torch.allclose(mean_voltage_limit_violation, torch.tensor(0.0), atol=1e-6):
        print("   [PASS] Voltage limits are satisfied by the ground-truth data.")
    else:
        print("   [FAIL] Ground-truth data violates the predefined voltage limits.")
    # --- END CORRECTION ---

    print("-"*40)
    print("\nTest finished.")
    print("="*80)


if __name__ == '__main__':
    run_physics_loss_test()