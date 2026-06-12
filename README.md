# Optimal Power Flow Prediction Using Physics-Informed Spatiotemporal Graph Neural Networks

This repository outlines the detailed study of Graph Neural Network (GNN) architectures for predicting AC Optimal Power Flow (ACOPF) solutions in electrical distribution and transmission grids under dynamic topology changes. The system simulates realistic grid conditions including renewable energy integration, configuration switching (line rerouting), and stochastic weather models. It then trains and benchmarks seven distinct neural architectures ranging from static spatial baselines to physics-informed spatiotemporal models.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Mathematical Formulation](#mathematical-formulation)
  - [AC Power Flow Equations](#ac-power-flow-equations)
  - [Physics-Informed Loss Function](#physics-informed-loss-function)
  - [Graph Convolution](#graph-convolution)
  - [Per-Unit Normalization](#per-unit-normalization)
- [Model Architectures](#model-architectures)
- [Data Pipeline](#data-pipeline)
  - [Test Cases](#test-cases)
  - [Feature Vector](#feature-vector)
  - [Load and Renewable Profiles](#load-and-renewable-profiles)
  - [Configuration Switching](#configuration-switching)
  - [Preprocessing](#preprocessing)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Individual Pipeline Steps](#individual-pipeline-steps)
  - [Parallel Pipeline Execution](#parallel-pipeline-execution)
  - [Makefile Reference](#makefile-reference)
- [Configuration](#configuration)
- [Evaluation and Benchmarking](#evaluation-and-benchmarking)
- [Warm-Start Benchmark Suite](#warm-start-benchmark-suite)
- [Testing](#testing)
- [Experiment Tracking](#experiment-tracking)

---

## Problem Statement

Solving AC Optimal Power Flow is computationally expensive. Classical solvers (Newton-Raphson, Gauss-Seidel) require iterative matrix operations at every timestep, which becomes a bottleneck for real-time grid operation, especially as renewable energy sources introduce high-frequency variability in both generation and topology.

This project trains GNNs to approximate ACOPF solutions directly from the grid state (bus power injections, voltage measurements, and topology) in a single forward pass. The key question is whether encoding physical constraints into the training loss and incorporating temporal dependencies improves prediction accuracy and physical feasibility compared to purely data-driven baselines.

---

## Mathematical Formulation

### AC Power Flow Equations

The AC power flow equations govern the relationship between voltage phasors and power injections at each bus in the network. For bus `i` in an `N`-bus system:

**Complex Voltage Phasor:**

$$V_i = |V_i| \cdot e^{j \cdot \theta_i}$$

where `|V_i|` is the voltage magnitude (p.u.) and `θ_i` is the voltage angle (radians).

**Current Injection (Ohm's Law for Networks):**

$$I_i = \sum_{k=1}^{N} Y_{ik} \cdot V_k$$

where `Y_{ik}` is the `(i, k)` entry of the bus admittance matrix (Ybus). The Ybus encodes the network topology: `Y_{ik} = -y_{ik}` for the mutual admittance between buses `i` and `k`, and `Y_{ii} = Σ_{k≠i} y_{ik}` for the self-admittance.

**Complex Power Injection:**

$$S_i = V_i \cdot I_i^* = P_i + j \cdot Q_i$$

Expanding this gives the standard power balance equations:

**Active Power Balance:**

$$P_i = |V_i| \cdot \sum_{k=1}^{N} |V_k| \cdot (G_{ik}\cos(\theta_i - \theta_k) + B_{ik}\sin(\theta_i - \theta_k))$$

**Reactive Power Balance:**

$$Q_i = |V_i| \cdot \sum_{k=1}^{N} |V_k| \cdot (G_{ik}\sin(\theta_i - \theta_k) - B_{ik}\cos(\theta_i - \theta_k))$$

where `G_{ik} + j·B_{ik} = Y_{ik}` are the real (conductance) and imaginary (susceptance) components of the admittance matrix.

**Net Power Injection at Bus `i`:**

$$\begin{aligned} P_{net,i} &= P_{ext,i} + P_{gen,i} + P_{ren,i} - P_{load,i} \\ Q_{net,i} &= Q_{ext,i} + Q_{gen,i} + Q_{ren,i} - Q_{load,i} \end{aligned}$$

### Physics-Informed Loss Function

The total loss for physics-informed models is:

$$L_{total} = L_{data} + \lambda_P L_{power} + \lambda_V L_{voltage} + \lambda_S L_{branch}$$

Each term is defined below.

#### 1. Data Loss (MSE)

$$L_{data} = \frac{1}{NB} \sum_{i=1}^{N} \sum_{b=1}^{B} [(\hat{V}_{m,b,i} - V_{m,b,i})^2 + (\hat{\theta}_{b,i} - \theta_{b,i})^2]$$

where `V̂m` and `θ̂` are the predicted voltage magnitude deviation and angle, and `Vm`, `θ` are the ground-truth values from pandapower.

#### 2. Power Balance Loss (Equality Constraint)

$$L_{power} = \frac{1}{NB} \sum [(P_{calc,i} - P_{net,true,i})^2 + (Q_{calc,i} - Q_{net,true,i})^2]$$

The calculated power `P_calc` and `Q_calc` are derived from the predicted voltages using the AC power flow equations above. The true net power `P_net_true` comes from the ground-truth target vector (not from the predictions).

In the code, this is computed as:

```python
V_complex = (Vm_pred + 1.0) · exp(j · Va_pred)          # Reconstruct phasor
I_injected = Y_bus @ V_complex                            # Current injection
S_calc = V_complex · conj(I_injected)                     # Complex power
P_calc = Re(S_calc)
Q_calc = Im(S_calc)
```

#### 3. Voltage Limit Loss (Inequality Constraint)

$$L_{voltage} = \frac{1}{NB} \sum [\text{ReLU}(|V_i| - V_{max})^2 + \text{ReLU}(V_{min} - |V_i|)^2]$$

This penalizes predicted voltage magnitudes that fall outside the operational bounds. The bounds are case-specific:

| System    | V_min (p.u.) | V_max (p.u.) | S_base (MVA) |
| :-------- | :----------- | :----------- | :----------- |
| Case 33   | 0.85         | 1.15         | 10           |
| Case 57   | 0.90         | 1.10         | 100          |
| Case 118  | 0.90         | 1.10         | 100          |

#### 4. Branch Capacity Loss (Inequality Constraint)

$$L_{branch} = \frac{1}{LB} \sum [\text{ReLU}(|S_k| - S_{k,max})^2]$$

where the branch apparent power flow is:

$$\begin{aligned} I_k &= Y_{branch,k} (V_{from,k} - V_{to,k}) \\ |S_k| &= |V_{from,k} \cdot I_k^*| \end{aligned}$$

`Y_branch_k` is the series admittance of branch `k`, extracted as `-Y_{bus}[from, to]`. `S_k_max` is the thermal rating of the branch in per-unit.

#### Default Loss Weights

| Parameter | Symbol | Default Value |
| :-------- | :----- | :------------ |
| Power balance weight | λ_P | 0.1 |
| Voltage limit weight | λ_V | 0.01 |
| Branch capacity weight | λ_S | 0.01 |

These are configurable in `configs/training.yaml` under `physics_loss`.

### Graph Convolution

All models use the GCN convolution operator from Kipf & Welling (2017):

$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$

where:
- `Ã = A + I` is the adjacency matrix with added self-loops
- `D̃` is the diagonal degree matrix of `Ã`
- `H^(l)` is the node feature matrix at layer `l`
- `W^(l)` is the learnable weight matrix at layer `l`
- `σ` is a nonlinearity (ReLU)

The adjacency matrix `A` is derived from the Ybus: two buses are connected (`A_{ij} = 1`) if there is an in-service line between them. For dynamic models, `A` changes per timestep based on the current switching state.

### Per-Unit Normalization

All power quantities (MW, MVar) are normalized by the system base `S_base`:

$$x_{pu} = \frac{x_{MW}}{S_{base}}$$

Voltage magnitudes are mean-centered around the nominal value:

$$Vm_{feature} = Vm_{pu} - 1.0$$

Voltage angles are left in radians (already small-scale). Bus degree is normalized by the maximum degree observed in the base topology.

---

## Model Architectures

The project implements seven model architectures, organized into three categories:

### Spatial-Only Models

These process one timestep at a time. Input shape: `(B, N, F)` where `B` is batch size, `N` is number of buses, `F` is number of features.

| # | Model | Class | Physics Loss | Adjacency | Description |
|---|-------|-------|:------------:|:---------:|-------------|
| 1 | StandardGCN | `StandardGCN` | No | Static | Baseline model that uses a fixed adjacency matrix loaded from disk, ignoring any topology changes. |
| 2 | DynamicGCN | `DynamicGCN` | No | Dynamic | Similar to the StandardGCN, but the adjacency matrix is updated for every sample to reflect the current switching state. |
| 3 | PIGCN | `PIGCN` | Yes | Dynamic | Physics-Informed GCN with the same forward pass as Model 2, but the training loss incorporates power balance, voltage limit, and branch capacity penalties. |

### Spatiotemporal Models

These process a sequence of `T` timesteps. Input shape: `(B, T, N, F)`. They predict the state at the final timestep of the sequence.

| # | Model | Class | GCN Type | Temporal Layer | Description |
|---|-------|-------|----------|----------------|-------------|
| 4 | PIGCLSTM | `PIGCLSTM` | Standard GCNConv | LSTM | GCN extracts spatial features at each timestep. The per-node embeddings are then passed through an LSTM to capture temporal dependencies. |
| 5 | PIGCGRU | `PIGCGRU` | Standard GCNConv | GRU | Same as Model 4 but with a GRU instead of LSTM. Fewer parameters, generally faster. |
| 6 | PIResnetGCLSTM | `PIResnetGCLSTM` | Residual GCNConv | LSTM | Uses Residual GCN blocks (two GCNConv layers with a skip connection) to mitigate oversmoothing in deep GNNs. Temporal layer is LSTM. |
| 7 | PIResnetGCGRU | `PIResnetGCGRU` | Residual GCNConv | GRU | Residual GCN blocks with GRU. |

All spatiotemporal models (4–7) use physics-informed loss. The temporal architecture follows this pattern:

```text
For each sample b in batch:
  For each timestep t in sequence:
    h_{b,t} = GCN(X_{b,t}, A_{b,t})            # Spatial embedding [N, H]
  H_b = stack(h_{b,1}, ..., h_{b,T})            # [T, N, H]
  H_b = reshape(H_b) → [N, T, H]               # Per-node sequences
  O_b = RNN(H_b)                                # Temporal processing
  pred_b = Linear(O_b[:, -1, :])                # Last timestep output [N, 2]
```

The output is always `(B, N, 2)`, representing the predicted voltage magnitude deviation and voltage angle for each bus.

### Residual GCN Block

The `ResidualGCNBlock` implements:

```text
identity = shortcut(x)
h = ReLU(GCNConv_1(x, A))
h = GCNConv_2(h, A)
output = ReLU(h + identity)
```

If the input and output dimensions differ, `shortcut` is a linear projection. Otherwise it is the identity function.

### Optimizer and Scheduler

All models use:
- **Optimizer:** Adam (lr = 0.001)
- **Scheduler:** ReduceLROnPlateau (patience = 10, factor = 0.5, monitors `val_loss`)
- **Early Stopping:** patience = 20 epochs on `val_loss`

---

## Data Pipeline

### Test Cases

The project uses three standard IEEE test systems provided by `pandapower`:

| Case | Buses | Lines | Type | Topology | Base MVA |
|------|------:|------:|------|----------|:--------:|
| `case33` (Baran & Wu) | 33 | 37 | Distribution (radial) | Has tie-lines (normally open) | 10 |
| `case57` | 57 | 80 | Transmission (meshed) | No tie-lines | 100 |
| `case118` | 118 | 186 | Transmission (meshed) | No tie-lines | 100 |

### Feature Vector

Each bus at each timestep is described by an 11-dimensional feature vector:

| Index | Feature | Unit | Description |
|:-----:|---------|------|-------------|
| 0 | `P_LOAD` | MW → p.u. | Active power consumed at this bus |
| 1 | `Q_LOAD` | MVar → p.u. | Reactive power consumed at this bus |
| 2 | `P_EXT_GRID` | MW → p.u. | Active power from the external grid (slack bus) |
| 3 | `Q_EXT_GRID` | MVar → p.u. | Reactive power from the external grid |
| 4 | `P_CONV` | MW → p.u. | Active power from conventional generators |
| 5 | `Q_CONV` | MVar → p.u. | Reactive power from conventional generators |
| 6 | `P_REN` | MW → p.u. | Active power from renewable generators (solar/wind) |
| 7 | `Q_REN` | MVar → p.u. | Reactive power from renewable generators |
| 8 | `VM` | p.u. - 1.0 | Voltage magnitude (mean-centered) |
| 9 | `VA` | radians | Voltage angle |
| 10 | `DEGREE` | normalized | Bus degree (number of connected lines / max degree) |

The first 8 features have zero-mean Gaussian sensor noise added during data generation (`voltage_error_std = 0.005`, `power_error_std = 0.01`, `angle_error_std = 0.02`).

### Target Vector

Each bus has a 10-dimensional target vector containing the ground-truth power flow solution from pandapower (indices 0–7 mirror the feature power columns, indices 8–9 are voltage magnitude deviation and angle). The GNN predicts only indices 8 and 9.

### Load and Renewable Profiles

#### Load Profile

A 24-hour "Camel" demand shape. Each hour maps to a scaling factor (0–1):

```text
Hour:   0     1     ...   9    10    ...   18    19    ...   23
Scale:  0.40  0.35  ...  0.90  0.88  ...  1.00  0.98  ...  0.50
```

The profile has two peaks: a morning peak around 9:00–10:00 and an evening global peak at 18:00. A ±5% uniform random perturbation is applied per timestep.

#### Solar Profile

A bell-curve peaking at 12:00 (noon), modulated by a seasonal factor and a stochastic weather model:

$$P_{\text{solar}}(h) = P_{\text{base}}(h) \cdot \text{cloud factor} \cdot \text{season factor}$$

- `season_factor = 0.85 + 0.15 · sin(2π · (day - 80) / 365)`, peaking around the summer solstice.
- `cloud_factor` is sampled from a weather-state-dependent range:

| Weather State   | Cloud Factor Range |
|-----------------|--------------------|
| Clear           | [0.90, 1.00]       |
| Partly Cloudy   | [0.35, 0.85]       |
| Cloudy          | [0.08, 0.35]       |
| Storm           | [0.00, 0.08]       |

#### Wind Profile

A night-peaking coastal profile. Wind output is highest at night (hours 0–4, 20–23) and lowest around midday (hours 11–13). The output is modulated by a weather state:

| Weather State | Wind Speed Range |
|---------------|------------------|
| Calm          | [0.00, 0.20]     |
| Breezy        | [0.20, 0.55]     |
| Windy         | [0.55, 0.90]     |
| Storm         | [0.85, 1.00]     |

#### Weather Model

Weather evolves over time using a first-order Markov chain with separate transition matrices for solar and wind states. For example, the solar transition matrix:

```text
From \ To       Clear   Partly   Cloudy  Storm
Clear           0.65    0.30     0.05    0.00
Partly Cloudy   0.25    0.45     0.25    0.05
Cloudy          0.10    0.30     0.50    0.10
Storm           0.00    0.10     0.40    0.50
```

#### Renewable Reactive Power

Renewable inverters provide reactive power support based on local voltage:
- If `V < 0.98 p.u.`: inject positive Q (capacitive) proportional to the voltage deficit.
- If `V > 1.02 p.u.`: absorb Q (inductive) proportional to the voltage excess.
- Maximum reactive output: `Q_max = 0.33 · P`.

### Configuration Switching

The data generator simulates grid reconfiguration events (switching), where tie-lines are closed and other lines are opened to reroute power flow while maintaining connectivity.

**Radial grids (Case 33):** The network includes pre-defined tie-lines that are normally open. During a switching event, the system closes a randomly selected tie-line, identifies the resulting loop via `networkx.cycle_basis()`, and then opens a different line within that loop.

**Meshed grids (Case 57, 118):** These networks do not have designated tie-lines. The code first identifies a cycle in the existing graph and opens one line to create a "virtual" tie-line, then proceeds with the standard switching procedure.

The probability of a switching event at each timestep is controlled by `configuration_rate` in `configs/data_generation.yaml` (default: 10%).

For each unique topology encountered during simulation, the Ybus matrix is recomputed and stored. Topology states are tracked via integer IDs:
- `topology_id = 0` → base topology (no switching)
- `topology_id = k > 0` → the `k`-th unique switching configuration

### Preprocessing

The preprocessing script (`scripts/preprocess_data.py`):

1. Loads raw `.npy` files from `data/raw/<case>/` (concatenates all renewable fraction runs).
2. Applies per-unit normalization using the case-specific `S_base`.
3. Splits the data chronologically (70% train / 15% validation / 15% test).
4. Saves PyTorch tensors to `data/prep/<case>/`:
   - `train_features.pt`, `val_features.pt`, `test_features.pt`
   - `train_targets.pt`, `val_targets.pt`, `test_targets.pt`
   - `train_topology_ids.pt`, `val_topology_ids.pt`, `test_topology_ids.pt`
   - `ybus_base.pt`, `ybus_contingencies.pt`, `ybus_contingency_timesteps.pt`
   - `adjacency.pt`, `branch_from.pt`, `branch_to.pt`, `branch_max_s_pu.pt`
   - `normalization.json` (stores `S_base`, `max_degree`, and split sizes)

---

## Project Structure

```text
.
├── configs/
│   ├── data_generation.yaml      # Data simulation parameters
│   └── training.yaml             # Model, training, and evaluation settings
├── data/
│   ├── raw/                      # Raw .npy from pandapower simulations
│   └── prep/                     # Normalized PyTorch tensors
├── scripts/
│   ├── generate_data.py          # Data generation (pandapower simulation)
│   ├── preprocess_data.py        # Normalization and train/val/test splitting
│   ├── generate_benchmark_states.py  # Canonical warm-start benchmark state export
│   ├── train.py                  # Model training with Lightning + W&B
│   ├── evaluate.py               # Benchmark evaluation (accuracy, physics, speed)
│   ├── analyze_uncertainty.py    # Test-Time Augmentation uncertainty analysis
│   ├── benchmark_ws_speed.py     # Warm-start speed benchmark
│   ├── benchmark_ws_feasibility.py  # Warm-start feasibility benchmark
│   ├── benchmark_ws_rescue.py    # Warm-start rescue benchmark
│   └── animate_grid_dynamics.py  # Grid topology animation
├── src/
│   ├── constants.py              # Feature indices, physical constants, load profiles
│   ├── benchmarks/
│   │   ├── warm_start_evaluator.py   # Solver-in-the-loop warm-start evaluator
│   │   ├── warmstart_protocol.py     # Shared method/rule definitions
│   │   ├── benchmark_state.py        # Canonical benchmark state schema
│   │   ├── benchmark_dataset.py      # JSONL save/load for benchmark states
│   │   ├── state_builder.py          # Conversion from tensors to canonical states
│   │   ├── speed_runtime.py          # Runtime adapter for speed benchmark
│   │   ├── speed_runner.py           # Speed benchmark aggregation runner
│   │   ├── feasibility_runtime.py    # Runtime adapter for feasibility benchmark
│   │   ├── feasibility_runner.py     # Feasibility benchmark aggregation runner
│   │   ├── rescue_runtime.py         # Runtime adapter for rescue benchmark
│   │   │                              # (flat-fail candidate gate)
│   │   └── rescue_runner.py          # Rescue benchmark aggregation runner
│   ├── models/
│   │   ├── __init__.py           # Model registry
│   │   ├── gcn.py                # Model 1: StandardGCN
│   │   ├── dynamic_gcn.py        # Model 2: DynamicGCN
│   │   ├── pi_gcn.py             # Model 3: PIGCN
│   │   ├── pi_gclstm.py         # Model 4: PIGCLSTM
│   │   ├── pi_gcgru.py          # Model 5: PIGCGRU
│   │   ├── pi_resnet_gclstm.py  # Model 6: PIResnetGCLSTM
│   │   ├── pi_resnet_gcgru.py   # Model 7: PIResnetGCGRU
│   │   ├── layers.py            # ResidualGCNBlock, adjacency normalization
│   │   ├── physics_loss.py      # PhysicsLoss (3 ACOPF constraints)
│   │   └── data_module.py       # Lightning DataModule (spatial and temporal)
│   ├── processing/
│   │   ├── topology.py           # Network loading, switching, Ybus computation
│   │   ├── profiles.py           # Load/solar/wind profiles, weather model
│   │   └── validation.py         # Power flow input/output validation
│   └── visualization/
│       ├── plot_benchmarks.py     # Accuracy, physics gap, efficiency plots
│       ├── plot_uncertainty.py    # Spatial and temporal uncertainty maps
│       ├── plot_data_profile.py   # Load/generation profile visualization
│       ├── plot_switching_heatmap.py
│       └── ...                    # Additional plotting modules
├── tests/
│   ├── test_models.py            # Unit tests for all 7 model architectures
│   ├── test_data_physics.py      # Physics validation of generated data
│   ├── test_preprocessing.py     # Normalization and splitting tests
│   ├── test_topology.py          # Switching event verification
│   ├── test_evaluation.py        # Evaluation pipeline tests
│   ├── test_training_e2e.py      # End-to-end training smoke tests
│   ├── test_warmstart_protocol.py
│   ├── test_benchmark_state.py
│   ├── test_benchmark_dataset.py
│   ├── test_state_builder.py
│   ├── test_benchmark_metrics.py
│   ├── test_speed_runtime.py
│   ├── test_speed_runner.py
│   ├── test_feasibility_runtime.py
│   ├── test_feasibility_runner.py
│   ├── test_rescue_runtime.py
│   └── test_rescue_runner.py
├── reports/                      # Generated plots and CSVs
├── logs/                         # Pipeline execution logs
├── Makefile                      # Convenience targets for all pipeline steps
├── run_pipeline.sh               # Parallel execution of all cases
├── verify_setup.sh               # Quick end-to-end validation
├── requirements-cpu.txt
└── requirements-gpu.txt
```

---

## Installation

### Prerequisites

- Python 3.11.x (recommended; project tested with 3.11.9)
- CUDA 12.x (for GPU training; CPU-only is supported)

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-org>/spatio_temporal_nn.git
cd spatio_temporal_nn

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements-gpu.txt   # For GPU (CUDA 12.x)
# or
pip install -r requirements-cpu.txt   # For CPU-only
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.10.0 | Tensor computation and autograd |
| PyTorch Geometric | 2.7.0 | GCNConv and graph utilities |
| Lightning | 2.6.1 | Training loop, callbacks, logging |
| pandapower | ≥2.14.0 | Power flow simulation and IEEE test cases |
| Weights & Biases | 0.25.0 | Experiment tracking |
| NetworkX | (via pandapower) | Cycle detection for switching |

---

## Usage

### Quick Start

Run the sanity check to verify the full pipeline works end-to-end with minimal data (24 timesteps, 1 epoch):

```bash
bash verify_setup.sh
```

### Individual Pipeline Steps

```bash
# 1. Generate data (10,008 timesteps for Case 33)
python scripts/generate_data.py --case 33 --timesteps 10008

# 2. Preprocess (normalize and split)
python scripts/preprocess_data.py --case 33

# 3. Train all 7 models
python scripts/train.py --case 33 --models all

# 4. Evaluate (benchmark against classical solvers)
python scripts/evaluate.py --case case33

# 5. Uncertainty analysis (Test-Time Augmentation)
python scripts/analyze_uncertainty.py --case case33
```

### Parallel Pipeline Execution

The `run_pipeline.sh` script runs all three cases (33, 57, 118) simultaneously in isolated background processes:

```bash
bash run_pipeline.sh
```

Configuration is controlled by flags at the top of the script:

```bash
# Case Selection
DO_CASE_33=true
DO_CASE_57=true
DO_CASE_118=true

# Execution Controls
DO_GENERATE=true
DO_PREPROCESS=true
DO_TRAIN=true
DO_EVALUATE=true
DO_UNCERTAINTY=true

# Cleanup Controls (run before pipeline starts)
DO_CLEAN_REPORTS=true
DO_CLEAN_LOGS=true
DO_CLEAN_WANDB=true
DO_CLEAN_RAW_DATA=false       # Caution: regeneration is slow
DO_CLEAN_PROCESSED_DATA=true
```

### Makefile Reference

| Target | Description |
|--------|-------------|
| `make gen-33` | Generate 96-timestep data for Case 33 |
| `make gen-full` | Generate 10,008-timestep data for all cases |
| `make prep-33` | Preprocess Case 33 data |
| `make train-33` | Train all models on Case 33 |
| `make eval-33` | Run benchmark evaluation for Case 33 |
| `make unc-33` | Run uncertainty analysis for Case 33 |
| `make full-33` | Full pipeline: generate → preprocess → test → train → evaluate → uncertainty |
| `make full-test` | Full pipeline for all cases (sequential) |
| `make test` | Run all pytest tests |
| `make test-fast` | Run tests, stop on first failure |
| `make clean-all` | Remove all generated data, logs, reports, and caches |
| `make clean-training` | Remove only logs, checkpoints, and W&B data |
| `make clean-reports` | Remove only report directories |
| `make sync` | Upload offline W&B runs to the cloud |
| `make gen-bench-33` | Build canonical benchmark states for Case 33 |
| `make ws-speed-33` | Run warm-start speed benchmark on Case 33 |
| `make ws-feas-33` | Run warm-start feasibility benchmark on Case 33 |
| `make ws-rescue-33` | Run warm-start rescue benchmark on Case 33 |
| `make ws-all-33` | Full warm-start benchmark pipeline for Case 33 |
| `make ws-all-57` | Full warm-start benchmark pipeline for Case 57 |
| `make ws-all-118` | Full warm-start benchmark pipeline for Case 118 |
| `make ws-smoke-33` | 5-sample smoke test for warm-start benchmarks |

---

## Configuration

### `configs/data_generation.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `time_steps` | 10008 | Number of simulation timesteps |
| `renewable_fractions_to_run` | [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] | Renewable penetration levels to simulate |
| `configuration_rate` | 0.10 | Probability of a switching event per timestep |
| `voltage_error_std` | 0.005 | Std. dev. of Gaussian noise on voltage measurements |
| `power_error_std` | 0.01 | Std. dev. of Gaussian noise on power measurements |
| `angle_error_std` | 0.02 | Std. dev. of Gaussian noise on angle measurements |
| `pmu_coverage` | 0.3 | Fraction of buses with PMU-quality measurements |
| `max_energy_utilization_coeff` | 0.98 | Maximum renewable capacity utilization |
| `solar_weather_weights` | [0.3, 0.4, 0.25, 0.05] | Initial weather state probabilities for solar |
| `wind_weather_weights` | [0.15, 0.45, 0.30, 0.10] | Initial weather state probabilities for wind |

### `configs/training.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Training batch size |
| `seq_len` | 4 | Sequence length for spatiotemporal models |
| `in_channels` | 11 | Number of input features per bus |
| `out_channels` | 2 | Number of outputs per bus (Vm deviation, Va) |
| `gcn_hidden` | 64 | Hidden dimension for GCN layers |
| `lstm_hidden` / `gru_hidden` | 64 | Hidden dimension for recurrent layers |
| `num_layers` | 4 | Number of GCN layers or Residual blocks |
| `learning_rate` | 0.001 | Initial learning rate for Adam |
| `lr_patience` | 10 | Epochs before reducing learning rate |
| `lr_factor` | 0.5 | Learning rate reduction factor |
| `max_epochs` | 200 | Maximum training epochs |
| `seed` | 42 | Random seed for reproducibility |
| `lambda_power_balance` | 0.1 | Weight for power balance physics loss |
| `lambda_voltage_limit` | 0.01 | Weight for voltage limit physics loss |
| `lambda_branch_capacity` | 0.01 | Weight for branch capacity physics loss |
| `solver_trials` | 3 | Number of speed trials per classical solver |
| `num_tta_samples` | 10 | Test-Time Augmentation samples for uncertainty |
| `tta_noise_scale` | 0.05 | Proportional noise scale for TTA (5%) |

---

## Evaluation and Benchmarking

The evaluation script (`scripts/evaluate.py`) benchmarks each trained model on the held-out test set along three axes:

1. **Prediction Accuracy:** MAE for voltage magnitude and voltage angle against the pandapower ground truth.
2. **Physical Feasibility:** Constraint satisfaction rates for active power balance (P), reactive power balance (Q), voltage limits (V), and branch capacity (S). A 0.5% tolerance is used for power balance.
3. **Computational Efficiency:** Inference speed compared to four classical `pandapower` solvers:
   - Newton-Raphson (NR)
   - Newton-Raphson + Iwamoto multiplier (NRI)
   - Gauss-Seidel (GS)
   - Backward/Forward Sweep (BFS, radial grids only)

The uncertainty analysis (`scripts/analyze_uncertainty.py`) uses Test-Time Augmentation (TTA) by injecting 5% proportional Gaussian noise into load and renewable power features. Inference is then repeated `num_tta_samples` times, and the variance of these predictions across the augmented inputs is used to measure how sensitive the model is to input perturbations.

Results are saved to `reports/benchmarks/<case>/` and `reports/uncertainty/<case>/`.

---

## Warm-Start Benchmark Suite

This benchmark suite evaluates a specific scientific question:

**Does a learned voltage initializer place Newton-Raphson closer to the convergence region, thereby improving convergence speed, physical quality of converged solutions, and recovery on difficult operating points?**

The design is solver-in-the-loop and mirrors operational usage: at each timestep, the neural initializer and Newton-Raphson receive the same network state.

### Problem Formulation

At timestep `t`, define the grid state:

$$s_t = (G_t, P_t, Q_t, \xi_t)$$

where:
- `G_t` is active topology (graph, line statuses),
- `P_t, Q_t` are active/reactive injections from load/gen/renewables,
- `ξ_t` represents measurement context (noise, PMU sparsity, missingness pattern where applicable).

The initializer predicts bus voltage magnitude and angle:

$$\hat{x}_t = (\hat{V}_{m,t}, \hat{\theta}_t)$$

The AC power-flow solver then solves:

$$F(x_t; s_t) = 0$$

with one of three initialization schemes:
- flat start (`init="flat"`),
- DC start (`init="dc"`),
- neural warm-start (`init="results"` using `\hat{x}_t`).

### Operational Consistency Requirement

For each sample, the same canonical state is used in both paths:

1. **Neural path:** `s_t -> \hat{x}_t`
2. **NR path:** `s_t + init -> x_t^*` (if converged)

This prevents hidden confounders and ensures performance differences come from initialization quality, not data mismatch.

### Canonical State Representation

Benchmark states are exported to:

```text
data/benchmark/<case>/states.jsonl
```

Each state stores:
- identity: `sample_id`, `case_name`, `timestep`
- scenario metadata: `renewable_fraction`, `topology_id`
- per-bus feature tensor (`features`) matching model input schema
- active edges (`active_edges`) used to reconstruct line on/off statuses
- auxiliary metadata (`metadata`)

### Realism Assumptions Carried from Data Pipeline

Benchmark states are intended to preserve the same realism used for model-facing data:
- renewable penetration scheduling (`0.0 ... 1.0`),
- topology switching / contingencies,
- measurement noise assumptions,
- PMU-coverage driven observability assumptions,
- preprocessing-compatible feature layout.

This is the core reason the benchmark state builder is separated from training code but aligned with the same data conventions.

### Benchmark Pillars

#### 1) Speed (`scripts/benchmark_ws_speed.py`)

For each state, run all three initializations and record:
- convergence flag,
- solve time (ms),
- iterations to convergence.

Reported statistics include:
- per-method mean runtime,
- per-method mean iterations,
- success rate,
- warm-start speedup relative to flat start.

#### 2) Feasibility (`scripts/benchmark_ws_feasibility.py`)

Feasibility is evaluated on each method's own converged network result (`flat`, `dc`, `warmstart` separately).

Validation checks include:
- voltage magnitude limits,
- angle and loading plausibility,
- slack/generator/inverter-related operational checks from `validation.py`.

Case-specific voltage limits follow `SYSTEM_PHYSICS`:
- `case33`: `[0.85, 1.15]` p.u.
- `case57`: `[0.90, 1.10]` p.u.
- `case118`: `[0.90, 1.10]` p.u.

#### 3) Rescue (`scripts/benchmark_ws_rescue.py`)

Rescue candidates are defined by **flat-start failure**:
1. run flat first,
2. if flat converges, sample is excluded from rescue pool,
3. if flat fails, evaluate recovery under DC and warm-start on that same state.

This yields a direct answer to: *on hard cases, does warm-start recover more often than classical initialization?*

### Metric Definitions

For `T` evaluated states, let `CT_t^{flat}` and `CT_t^{ws}` be solve times for flat and warm-start:

$$\text{Speedup}_{ws/flat} = \frac{\frac{1}{T}\sum_{t=1}^T CT_t^{flat}}{\frac{1}{T}\sum_{t=1}^T CT_t^{ws}}$$

Feasibility rate:

$$\text{Feasibility Rate}(\%) = \frac{n_f}{n_r} \times 100$$

Constraint satisfaction rate:

$$\text{Constraint Satisfaction}(\%) = \frac{n_s}{n_c} \times 100$$

where:
- `n_f`: feasible solutions,
- `n_r`: total evaluated solutions,
- `n_s`: satisfied constraints,
- `n_c`: total constraints assessed.

Rescue recovery rate (for each method `m` in `{dc, warmstart}`):

$$\text{Recovery Rate}_m(\%) = \frac{n_{rec,m}}{n_{cand}} \times 100$$

where:
- `n_cand`: number of flat-failed rescue candidates,
- `n_rec,m`: number of candidates recovered by method `m`.

### End-to-End Runbook

#### Case-specific full run

```bash
make ws-all-33
make ws-all-57
make ws-all-118
```

#### Fast smoke validation (recommended before full runs)

```bash
make ws-smoke-33
```

#### Stepwise manual execution

```bash
# 1) Build benchmark states
make gen-bench-33

# 2) Speed benchmark
make ws-speed-33

# 3) Feasibility benchmark
make ws-feas-33

# 4) Rescue benchmark
make ws-rescue-33
```

### Output Artifacts

```text
reports/warmstart/
├── speed/<case>/
│   ├── <case>_speed_records.json
│   └── <case>_speed_summary.json
├── feasibility/<case>/
│   ├── <case>_feasibility_records.json
│   └── <case>_feasibility_summary.json
└── rescue/<case>/
    ├── <case>_rescue_records.json
    └── <case>_rescue_summary.json
```

### Reproducibility Notes

- Keep the same prepared dataset version for state generation and benchmarking.
- Compare methods on the exact same canonical states per case.
- Run smoke targets before long runs to validate environment and paths.
- Record Python version and dependency set when reporting benchmark results.

---

## Testing

```bash
make test              # Run all tests
make test-fast         # Stop on first failure
make test-physics      # Data physics validation only
make test-models       # Model architecture tests only
make test-topology     # Switching event verification
make test-e2e          # End-to-end training smoke test
make test-preprocessing # Normalization and splitting tests
```

The test suite covers:
- Forward pass shape checks for all 7 models
- Physics loss computation correctness
- Data generation output validation (voltage bounds, power balance)
- Preprocessing normalization and split ratios
- Topology switching event consistency (bus degree analysis)
- End-to-end training loop (1 epoch smoke test)

---

## Experiment Tracking

Training logs are managed by [Weights & Biases](https://wandb.ai/). By default, logging is in `offline` mode (fast, no network dependency). To sync offline runs to the cloud:

```bash
make sync
```

To train with live cloud logging:

```bash
python scripts/train.py --case 33 --models all --online
```

Each training run is automatically tagged with the model name, case, and a unique session timestamp for easy filtering in the W&B dashboard.
