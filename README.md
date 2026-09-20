# 2D GPU Car RL Environment (JAX + PyTorch)

A high-performance 2D dynamic vehicle simulation and reinforcement learning framework built with **JAX** (vectorized GPU physics) and **PyTorch** (PPO policy training) interconnected via **zero-copy DLPack**.

Supports **realistic vehicle dynamics**, **nonlinear tire slip (Pacejka)**, and **power-oversteer drifting**.

---

## Architecture Overview

```
+-----------------------------------------------------------------------------------------------+
|                                          GPU VRAM                                             |
|                                                                                               |
|   +----------------------------------------------------+      +---------------------------+   |
|   |                  JAX Environment                   |      |      PyTorch Trainer      |   |
|   |  - Dynamic 2D Physics (Pacejka Tires, Drift)       |      |  - Vectorized PPO         |   |
|   |  - Goal: Unit Vector [ux, uy] & Desired Speed      |      |  - Actor-Critic Network   |   |
|   |  - 2,048 Parallel Environments (JIT + vmap)        |      |  - 450,000+ Steps/Sec     |   |
|   |                                                    |      |                           |   |
|   |  Observations (Ego-centric + Dynamics):            |DLPack|                           |   |
|   |  [ux, uy, v_des, vx, vy, speed_err, delta, omega]->| Zero-|  Loss & Gradient Updates  |   |
|   |                                                    | Copy |                           |   |
|   |  Actions:                                          |<-----|                           |   |
|   |  [a_steer, a_torque]                               |      +---------------------------+   |
|   +----------------------------------------------------+                                      |
|                            ^                                                                  |
|                            | (Hierarchical Interface Ready)                                   |
|   +----------------------------------------------------+                                      |
|   | 2D JAX Raycast Lidar (Obstacle Sensing Ready)      |                                      |
|   | Outputs -> Sub-goal Unit Vector & Desired Speed    |                                      |
|   +----------------------------------------------------+                                      |
+-----------------------------------------------------------------------------------------------+
```

---

## Key Features

1. **2D Dynamic Bicycle Physics with Drifting**:
   - 6-DOF planar body dynamics ($x, y, \theta, v_x, v_y, \omega, \delta$).
   - Pacejka nonlinear lateral tire friction with friction ellipse coupling.
   - High rear wheel engine torque breaks traction, producing realistic power oversteer and drifting.
2. **Lightning-Fast GPU Vectorization**:
   - **450,000+ environment steps/second** on an NVIDIA RTX 3060 Laptop GPU.
   - Vectorized across 2,048 parallel environments via `@jax.jit` and `jax.vmap`.
3. **Zero-Copy DLPack Bridge**:
   - JAX simulations and PyTorch neural networks exchange GPU tensors without CPU copying.
   - Memory management ensures JAX and PyTorch comfortably share 6GB VRAM (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.4`).
4. **Hierarchical Extensibility**:
   - Includes a modular JAX 2D raycasting sensor (`car_rl/env/raycast.py`) and a defined agent interface (`car_rl/hierarchical/interface.py`) ready for the future higher-level obstacle-avoidance agent.

---

## Quickstart

### 1. Activate Environment
The virtual environment is already configured in `.venv`:
```bash
source .venv/bin/activate
```

### 2. Run Automated Verification Tests
```bash
# 1. Verify physical dynamics, tire slip, and drift mechanics:
python -m tests.test_physics

# 2. Benchmark GPU zero-copy DLPack throughput:
python -m tests.test_gpu_bridge
```

### 3. Train the Car Policy
Train the low-level controller to track commanded unit vectors and speeds:
```bash
python train.py --total-timesteps 5000000 --num-envs 2048
```
*(5 million steps completes in ~13 seconds!)*

Checkpoints are saved to `checkpoints/best_model.pt` and `checkpoints/latest_model.pt`.

### 4. Evaluate & Visualize Telemetry
Run an 8-second evaluation demo that commands sharp 90° turns and high-speed directional changes:
```bash
python demo.py --checkpoint checkpoints/best_model.pt --output car_trajectory.png
```
This generates a multi-panel telemetry plot (`car_trajectory.png`) visualizing:
- 2D trajectory path with vehicle heading arrows (red arrows indicate lateral drift).
- Commanded vs. actual speed tracking.
- Lateral slip speed ($|v_y|$).
- Front wheel steering angle (degrees).
- Engine drive / brake torque commands.

---

## Project Structure

```
Car/
├── README.md                    # Project documentation & usage
├── plan.md                      # Architecture design document
├── requirements.txt             # PyTorch, JAX, CUDA dependencies
├── .venv/                       # Symlinked virtual environment
├── car_rl/
│   ├── utils/
│   │   └── gpu_config.py        # CUDA & JAX memory allocation configuration
│   ├── env/
│   │   ├── physics.py           # 2D dynamic bicycle physics (Pacejka slip & drift)
│   │   ├── car_env.py           # Batched pure-functional JAX environment
│   │   ├── raycast.py           # 2D raycasting module for obstacle sensing
│   │   └── torch_wrapper.py     # Zero-copy DLPack JAX-to-PyTorch wrapper
│   ├── models/
│   │   └── actor_critic.py      # PyTorch continuous Actor-Critic networks
│   ├── training/
│   │   ├── config.py            # Hyperparameters dataclass
│   │   └── ppo.py               # Vectorized GPU PPO trainer
│   └── hierarchical/
│       └── interface.py         # Interfaces for future higher-level raycast agent
├── train.py                     # CLI entry point for training
├── demo.py                      # Interactive / evaluation trajectory visualizer
└── tests/
    ├── test_physics.py          # Physics unit tests
    └── test_gpu_bridge.py       # DLPack throughput benchmark
```
