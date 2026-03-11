# Required Paper Corrections: Aligning with Actual Implementation

This document outlines the discrepancies between the `paper.txt` manuscript and the actual codebase implementation. The goal is to update the paper to accurately reflect what the code actually does, rather than changing the code to match the theoretical claims.

Please review the following sections and update the paper accordingly.

## 1. Hyperparameter Optimization (MoSOA)
**What the paper claims:**
The paper claims to use a "Modified Seagull Optimization Algorithm (MoSOA)" for adaptive multi-objective hyperparameter tuning (optimizing learning rate, hidden layer dimensions, graph convolution depth, etc.).

**What the code actually does:**
There is no MoSOA, Seagull Optimization, or any dynamic hyperparameter tuning algorithm implemented in the codebase. All hyperparameters (learning rate, hidden dimensions, number of layers, batch size) are statically defined in `configs/training.yaml` and loaded directly by the training scripts.

**Correction needed:**
Remove all mentions of MoSOA or adaptive hyperparameter tuning from the abstract, methodology (Section 2.5), and conclusion. State instead that hyperparameters were selected and configured statically for all experiments.

## 2. Loss Function Weighting (Kendall's Homoscedastic Uncertainty)
**What the paper claims:**
Section 2.4 claims the network uses "Kendall’s homoscedastic uncertainty weighting" (Equation 42) to automatically and adaptively balance heterogeneous loss components (power balance, voltage limits, branch capacity) with learnable variance parameters ($\sigma_i^2$).

**What the code actually does:**
The loss function uses static, manually assigned penalty weights. In `configs/training.yaml` and `src/models/physics_loss.py`, the physics losses are weighted by scalar constants (`lambda_power_balance = 0.1`, `lambda_voltage_limit = 0.01`, `lambda_branch_capacity = 0.01`). There are no learnable uncertainty parameters in the loss formulation.

**Correction needed:**
Remove the description of Kendall's homoscedastic uncertainty weighting and Equation 42. Replace it with a description of a standard weighted penalty summation method, specifying that static weights were empirically determined and applied to the physics-informed loss terms.

## 3. Spatiotemporal GNN Architecture (Sequential vs. Simultaneous GCLSTM)
**What the paper claims:**
The paper defines a "Graph Convolution LSTM (GCLSTM)" mathematically (Equations 25-30) as having simultaneous spatial and temporal feature learning, where graph convolutions are embedded directly inside the LSTM cell's gating mechanisms.

**What the code actually does:**
The implementation uses a *decoupled, successive* spatiotemporal architecture. As seen in `src/models/pi_gclstm.py` and `pi_resnet_gclstm.py`, the input sequence is first processed spatially timestamp-by-timestamp using standard `GCNConv` blocks. The resulting spatial embeddings are flattened and then passed through an unmodified PyTorch `nn.LSTM` sequence model. The spatial and temporal learning stages happen in two consecutive steps, not simultaneously within the LSTM gates.

**Correction needed:**
Update Section 2.3.3 to describe a "Successive Spatiotemporal GNN" architecture. Remove the integrated GCLSTM gating equations (25-30) and clarify that the model employs graph convolutional layers for spatial feature extraction, followed sequentially by standard LSTM/GRU layers for temporal modeling.

## 4. Adaptive GCN Adjacency Matrix
**What the paper claims:**
Section 2.3.1 outlines a "Self-Adaptive GCN" that uses a learnable residual adjacency matrix, which dynamically blends the physical topology prior with data-driven interactions learned through softmax/attention over node embeddings.

**What the code actually does:**
The codebase uses `GCNConv` from `torch_geometric` utilizing exactly the physical grid topology (the post-contingency, dynamically updated edge connections). It does not learn or infer any hidden connections or construct residual adjacency matrices. Message passing occurs strictly along the physical transmission lines.

**Correction needed:**
Remove equations 13-15 detailing the "learnable adaptive adjacency". Clarify that the dynamic GCN strictly leverages the real-time physical topology (adjacency matrix reflecting current breaker/switch states) without introducing a learnable residual graph structure.

## 5. Multi-Objective Function (Carbon Emissions and Power Loss)
**What the paper claims:**
The methodology suggests the learning formulation simultaneously minimizes active power loss and carbon emissions directly within the objective function (Equations 4 and 6).

**What the code actually does:**
The models are trained solely to predict OPF system states (Voltage Magnitude and Voltage Angle) using Mean Squared Error (MSE), supplemented by physics-informed penalties (power balance, voltage bounds, thermal limits) in `physics_loss.py`. The neural network does not directly optimize for carbon emissions or active power loss in its backward-pass loss calculation. 

**Correction needed:**
Revise the discussion surrounding Equations 4 and 6. Clarify that the neural network's loss function optimizes state prediction accuracy and physical feasibility. If emission/loss minimization is discussed, it should be framed as an outcome of the underlying optimal power flow problem simulated in the dataset generation, rather than terms directly penalized in the neural network's loss function.
