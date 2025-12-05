# Model-Free Robot Control: Learning Inverse Differential Kinematics

**Author:** Luca Franzin
**Course:** Machine Learning 2025/2026

## Project Overview

This project implements a **Machine Learning approach** to solve the position control problem for robotic manipulators without explicit knowledge of their kinematic model (e.g., Denavit-Hartenberg parameters or the analytical Jacobian matrix).

Instead of deriving the Jacobian analytically, the system learns the **Inverse Differential Kinematics** function using Feed-Forward Neural Networks (FNN) and Residual Networks (ResNets). The goal is to predict the necessary joint displacement $\Delta q$ required to achieve a desired end-effector displacement $\Delta x$, given the current robot state.

The approach is validated on three simulated robot configurations:

  * **Reacher3:** Planar 3-DOF
  * **Reacher4:** Spatial 4-DOF
  * **Reacher6:** Spatial 6-DOF

## Methodology

The solution involves a complete pipeline ranging from data preprocessing to closed-loop control integration.

### 1\. Problem Formulation

The objective is to approximate the function:
$$\Delta q = f_{\theta}(x_{current}, q_{current}, \Delta x)$$
Where the model predicts joint updates based on the current Cartesian position, joint angles, and desired Cartesian displacement.

### 2\. Data Preprocessing

Raw simulation data was filtered to ensure the network learns the local linear relationship required for differential kinematics:

  * **Filtering:** Discarded samples with large joint displacements ($|\Delta q_{i}| > 10^{\circ}$).
  * **Noise Removal:** Removed stationary samples to prevent learning zero-output biases.
  * **Normalization:** Applied `StandardScaler` to inputs and targets.

### 3\. Model Architectures

Extensive hyperparameter optimization (using **Optuna**) was performed to determine the optimal architecture for each robot's dimensionality.

| Robot | Dimensions | Architecture Style | Details |
| :--- | :--- | :--- | :--- |
| **Reacher3** | In: $\mathbb{R}^7$, Out: $\mathbb{R}^3$ | **"Funnel" MLP** | Wide input layer (2048 units) tapering down. No residual connections. |
| **Reacher4** | In: $\mathbb{R}^{10}$, Out: $\mathbb{R}^4$ | **"Wide-Body" MLP** | High capacity, shallow network ($1024 \rightarrow 2048 \rightarrow 1024$).  |
| **Reacher6** | In: $\mathbb{R}^{12}$, Out: $\mathbb{R}^6$ | **Deep ResNet** | 7-Layer Deep Network with Residual Connections to handle kinematic redundancy.  |

### 4\. Closed-Loop Controller

The trained models are integrated into an iterative controller (ROS 2). To handle long-distance targets, the controller implements **step normalization**:
$$dx = dt \cdot \frac{step\_max}{||dt||}$$
This ensures the input to the NN remains within the "trust region" of the learned local manifold (typically $||dt|| < 0.05m$).

## Repository Structure

Based on the report, the codebase is organized as follows:

```text
.
├── src/
│   ├── config.py         # Hyperparameter search space definitions 
│   ├── data_loader.py    # Preprocessing, filtering, and normalization
│   ├── train.py          # Train functions
│   ├── test.py           # Test functions
│   ├── optimization.py   # Optuna optimization functions
│   ├── utils.py          # Helper functions
│   └── model.py          # DynamicMLP and ResNet implementations 
├── ros_control.py        # Iterative closed-loop controller logic 
├── main.ipynb            # Main notebook
└── README.md
```

## Results

### Regression Performance

The models were evaluated on a held-out test set using $R^2$ scores:

  * **Reacher3:** $R^2 \approx 0.6987$
  * **Reacher4:** $R^2 \approx 0.6702$
  * **Reacher6:** $R^2 \approx 0.4377$

### Visual Analysis

  * **Low-DOF (3/4):** Predictions cluster tightly along the diagonal, indicating a deterministic mapping.
  * **High-DOF (6):** Results show a "cloud" of variance. This is due to **Kinematic Redundancy** (the null-space effect), where multiple joint configurations result in the same end-effector pose.

Despite the lower regression score for the 6-DOF robot, the **closed-loop feedback mechanism** successfully compensated for single-step prediction errors, allowing all robots to reach their targets with negligible error.
