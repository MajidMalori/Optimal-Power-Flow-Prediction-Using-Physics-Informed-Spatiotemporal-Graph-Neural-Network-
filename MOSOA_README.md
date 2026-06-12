# MoSOA: Modified Seagull Optimization Algorithm

## Overview
The **Modified Seagull Optimization Algorithm (MoSOA)** is a next-generation population-based metaheuristic developed to solve complex, high-dimensional, and non-convex optimization problems. Originally inspired by the natural foraging and tactical behaviors of seagulls (Derviskadic, 2019), the base Seagull Optimization Algorithm (SOA) often suffers from premature convergence and poor exploration-exploitation balance in highly complex landscapes.

This repository implements **MoSOA**, which introduces **four major mathematical modifications** to the standard SOA architecture, specifically designed to optimize the hyperparameters of Physics-Informed Spatio-Temporal Graph Neural Networks (PISTGNN).

---

## 1. The Four Core Mathematical Modifications

To overcome the limitations of the base SOA, MoSOA implements the following four specific architectural enhancements.

### Modification 1: Dynamic Nonlinear Convergence Factor ($a$)
**The Problem:** Base SOA uses a linear decay for its search radius, which rapidly collapses the population diversity regardless of the actual fitness landscape.
**The Solution:** MoSOA introduces a dynamic, fitness-aware nonlinear decay parameter ($\sigma$) that monitors the population's standard deviation. If the population is highly diverse, it explores more; if it is converging, it exploits.

$$\sigma(t) = \sigma_{max} + \frac{\text{std}(Fit)}{\text{mean}(Fit) + \epsilon}$$
$$A(t) = f_c \cdot \left( 1 - \frac{t}{T_{max}} \right)^{\sigma(t)}$$

Where:
- $Fit$ is the array of the population's current fitness values.
- $f_c$ is the base frequency factor (default $2.0$).
- $T_{max}$ is the maximum number of iterations.

### Modification 2: Time-Varying Inertia Weight ($w$)
**The Problem:** Standard SOA relies solely on pure differential equations for movement, lacking a momentum mechanism to glide over rough, noisy objective surfaces.
**The Solution:** Inspired by Particle Swarm Optimization (PSO), MoSOA incorporates a time-varying inertia weight that linearly decays from $0.95$ to $0.35$. This ensures aggressive early-stage global exploration and precise fine-tuning in the final stages.

$$w(t) = w_{max} - \frac{t}{T_{max}} \cdot (w_{max} - w_{min})$$
Where $w_{max} = 0.95$ and $w_{min} = 0.35$.

### Modification 3: Exponential Refractive Perturbation ($\beta$)
**The Problem:** In massively multimodal landscapes (like F8-F23 or Neural Network hyperparameter tuning), base SOA frequently gets permanently trapped in deep local minima.
**The Solution:** MoSOA introduces a "Refractive Jump" (perturbation) vector. It calculates the distance between the current particle and the global best, and applies an exponentially decaying random noise burst. This acts as a localized search explosion that prevents stagnation.

$$\beta(t) = \exp \left( -p_{\beta} \cdot \frac{t}{T_{max}} \right)$$
$$Noise_{vector} = U(-1, 1)^{D} \cdot \beta(t) \cdot (X_{best} - X_i)$$

*(Our ablation studies specifically compare this exponential decay against linear, cosine, and quadratic alternatives, proving its superiority).*

### Modification 4: Reflective Boundary Handling
**The Problem:** Traditional optimization algorithms "clamp" or "clip" particles that fly outside the search space bounds. This causes particles to pile up at the absolute edges of the hypercube, destroying spatial diversity.
**The Solution:** MoSOA employs a **Reflective Boundary** strategy (Bounce-Back). If a parameter exceeds the maximum bound ($H$), it reflects back inward by the exact amount it overshot, preserving the gradient momentum of the particle.

$$
X_{i, d} = 
\begin{cases} 
2 \cdot L_d - X_{i, d} & \text{if } X_{i, d} < L_d \\
2 \cdot H_d - X_{i, d} & \text{if } X_{i, d} > H_d \\
X_{i, d} & \text{otherwise}
\end{cases}
$$

---

## 2. Complete MoSOA Position Update Cycle

During the **Prey Attack Phase (Local Exploitation)**, MoSOA utilizes a helical spiral movement defined by a shrinking radius:
$$radius = \tanh\left(1 - \frac{t}{T_{max}}\right) \cdot \exp(0.1 \cdot k)$$
Where $k$ is a random angle $[0, 2\pi]$. The spatial displacement in 3D is:
$$x = radius \cdot \cos(k), \quad y = radius \cdot \sin(k), \quad z = radius \cdot k$$

The final position update, combining the inertia weight, personal best memory, exponential perturbation, and the helical attack, is executed as follows:

$$P_{learned} = w(t) X_i + c_1 r_1 (X_{best} - X_i) + c_2 r_2 (P_{best, i} - X_i)$$
$$X_i(t+1) = \text{Reflect}\left[ P_{learned} + Noise_{vector} + (P_{attack} - P_{learned}) \cdot \frac{t}{T_{max}} \right]$$

---

## 3. Experimental Benchmarking Stages

The repository validates MoSOA against 7 competitor metaheuristics across four rigorous stages:

### Stage 1: Mathematical Benchmark (F1-F23)
Comparing MoSOA against SOA, PSO, GWO, GA, TSA, and HGSO across 23 standardized mathematical functions (Unimodal, Multimodal, and Fixed-Dimension).
```bash
make math
```

### Stage 2: Perturbation Ablation Study
Proving why the **Exponential Decay ($\beta$)** strategy in Modification 3 is statistically superior to Linear, Cosine, or Quadratic alternatives.
```bash
make pert
```

### Stage 3: HPO Tuning Benchmark
Generating hyperparameter fitness landscapes by mapping standard mathematical bounds against proxy Neural Network hyperparameter combinations.
```bash
make tune-math
```

### Stage 4: Applied PISTGNN Tuning
Using the finalized MoSOA to optimize the actual neural network architectures (StandardGCN, PIGCLSTM, etc.) for Power Flow prediction.
```bash
make tune-33
```

---

## 4. Hardware and Computational Budget
For the **Applied HPO (Stage 4)**, the configurations in `configs/mosoa.yaml` default to Research Grade:
- **Total Trials**: 30 per algorithm.
- **Max Epochs**: 100 per trial.
- **Population Size**: 10.

These settings are optimized to ensure a minimum of **3 generations** of swarm evolution, providing sufficient temporal depth for the social learning and perturbation mechanisms to take effect without exhausting computational resources.

---
*Developed for the KNUST Engineering Research Project on Optimal Power Flow Prediction (PISTGNN).*
