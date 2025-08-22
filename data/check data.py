# In c:\Users\Admin\Desktop\Sus\power_monitoring-orig 6(Final)\power_monitoring-orig\data\check data.py

import numpy as np
import os

# Define the path to the file you want to check
file_path = r'c:\Users\Admin\Desktop\Sus\power_monitoring-orig 6(Final)\power_monitoring-orig\data\case33_adjacency_frac0.0.npy'

print(f"Loading file: {file_path}")

try:
    # --- THIS IS THE FIX ---
    # Add allow_pickle=True to correctly load the object array
    adjacency_data = np.load(file_path, allow_pickle=True)
    
    # Now you can inspect the loaded data
    print("\nFile loaded successfully!")
    print(f"Data type: {type(adjacency_data)}")
    print(f"Data shape: {adjacency_data.shape}")
    print("\nData content (first 5x5 slice):")
    # This might need adjustment if the data isn't a simple 2D array
    if isinstance(adjacency_data, np.ndarray) and adjacency_data.ndim >= 2:
        print(adjacency_data[:5, :5])
    else:
        print(adjacency_data)

except FileNotFoundError:
    print(f"\nERROR: File not found at path: {file_path}")
except Exception as e:
    print(f"\nAn error occurred: {e}")