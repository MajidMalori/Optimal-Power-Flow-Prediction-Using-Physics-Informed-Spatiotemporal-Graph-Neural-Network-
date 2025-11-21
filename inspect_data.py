import numpy as np
import os

file_path = r"c:\Users\Bernard\Documents\Python\Dynamic-ANN-DSSE-main\Physics_Informed_Machine_Learning\data\train\case33_adjacency_frac0.0_20251105_093013.npy"

try:
    data = np.load(file_path, allow_pickle=True)
    print(f"Loaded data type: {type(data)}")
    print(f"Data shape: {data.shape}")
    if len(data) > 0:
        print(f"First element type: {type(data[0])}")
        print(f"First element: {data[0]}")
except Exception as e:
    print(f"Error loading file: {e}")
