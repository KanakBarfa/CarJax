# Implementation Plan: 2D JAX GPU RL Car Environment with PyTorch Training

## Goal Description
Build a high-performance 2D Reinforcement Learning system with **authentic drifting and vehicle dynamics**:
1. **Simulation Environment (JAX)**: Fully GPU-accelerated 2D **Dynamic Bicycle Physics with Nonlinear Tire Slip (Pacejka / Friction Ellipse)** running thousands of parallel environments using JAX (`vmap`, `jit`).
2. **Drifting & Vehicle Dynamics**:
   - Simulates lateral tire slip ($\alpha_f, \alpha_r$), yaw inertia ($I_z$), and coupled longitudinal/lateral tire friction limits.
   - Allows power-oversteer drifts (rear wheel slip under high torque), weight-transfer slides, and counter-steering drift stabilization.
3. **Low-Level Car Agent**:
   - **Inputs (Goal Conditioned)**:
     - 2D **Unit Vector** $[u_x, u_y]$ indicating the desired direction of travel.
     - **Desired Velocity** $v_{\text{desired}}$ (scalar target speed).
     - Full car body dynamic state (longitudinal speed $v_x$, lateral slip velocity $v_y$, yaw rate $\omega$, steering angle $\delta$).
   - **Outputs (Actuation)**:
     - **Steering control**: Steers/turns the front wheels ($\delta$).
     - **Engine torque control**: Modulates engine drive/braking torque ($T_{\text{engine}}$ applied to the wheels).
4. **Training Framework (PyTorch)**: Vectorized RL training (GPU-accelerated PPO) in PyTorch, transferring tensors directly between JAX and PyTorch on GPU with zero memory copy via DLPack.
5. **Hierarchical Extensibility**: Modular 2D JAX raycast sensor module designed so a future higher-level agent (sensing obstacles via raycasts) can directly command the unit vector $[u_x, u_y]$ and $v_{\text{desired}}$ to this car agent.
6. **Isolated Environment**: Dedicated Python virtual environment (`.venv`) configured for RTX 3060 (6GB VRAM) with optimized CUDA and memory limits.

```
+------------------------------------------------------------------------------------------------+
|                                           GPU Memory                                           |
|                                                                                                |
|   +----------------------------------------------------+       +---------------------------+   |
|   |                  JAX Environment                   |       |      PyTorch Trainer      |   |
|   |  - Dynamic 2D Physics (Pacejka Tires, Slip, Drift) |       |  - Vectorized PPO         |   |
|   |  - Goal: Unit Vector [ux, uy] & Desired Velocity   |       |  - Continuous Policy:     |   |
|   |  - 4096 Parallel Environments (JIT + vmap)         |       |    * Wheel Steering       |   |
|   |                                                    |       |    * Engine Torque        |   |
|   |  Observations (Ego-centric + Dynamics):            |DLPack |                           |   |
|   |  [ux, uy, v_desired, vx, vy, omega, delta] ------->| Zero- |                           |   |
|   |                                                    | Copy  |  Loss & Gradient Updates  |   |
|   |  Actions:                                          |<----- |                           |   |
|   |  [steering_cmd, torque_cmd]                        |       +---------------------------+   |
|   +----------------------------------------------------+                                       |
|                            ^                                                                   |
|                            | (Future Hierarchical Extension)                                   |
|   +----------------------------------------------------+                                       |
|   | 2D JAX Raycast Lidar (Obstacle Sensing Ready)      |                                       |
|   | Outputs -> Unit Vector & Desired Velocity          |                                       |
|   +----------------------------------------------------+                                       |
+------------------------------------------------------------------------------------------------+
```

---

## Detailed Component Specifications

### 1. Dynamic Physics Engine with Drifting (`car_rl/env/physics.py`)
To enable realistic sliding, drifting, and recovery, we use a **Dynamic Bicycle Model with Coupled Tire Friction**:

- **Dynamic State Vector**: $[x, y, \theta, v_x, v_y, \omega, \delta]$
  - $(x, y)$: 2D global position
  - $\theta$: Vehicle yaw / heading angle
  - $v_x$: Body-frame longitudinal velocity (forward speed)
  - $v_y$: Body-frame lateral velocity (sideways slip speed — non-zero during drifting!)
  - $\omega$: Yaw rate ($\dot{\theta}$)
  - $\delta$: Front wheel steering angle

- **Tire Slip Angles**:
  Front and rear tire slip angles capture the angle between wheel orientation and velocity vector:
  $$\alpha_f = \delta - \arctan2\left(v_y + l_f \omega, \; |v_x| + \epsilon\right)$$
  $$\alpha_r = -\arctan2\left(v_y - l_r \omega, \; |v_x| + \epsilon\right)$$

- **Tire Forces & Friction Ellipse (Enabling Drifting)**:
  - Normal loads on front/rear axles: $F_{zf} = m g \frac{l_r}{l_f + l_r}, \quad F_{zr} = m g \frac{l_f}{l_f + l_r}$
  - Peak friction: $F_{\max, f} = \mu F_{zf}, \quad F_{\max, r} = \mu F_{zr}$
  - Drive torque applied to rear wheels (Rear-Wheel Drive for drifting) creates longitudinal force $F_{xr} = T_{\text{engine}} / R_{\text{wheel}}$.
  - **Friction Ellipse**: As rear wheel torque $F_{xr}$ increases, the available lateral grip drops:
    $$F_{yr, \max} = \sqrt{\max\left(0, F_{\max, r}^2 - F_{xr}^2\right)}$$
    *When high throttle is applied, $F_{xr} \approx F_{\max, r}$, causing rear lateral grip to vanish $\to$ the rear steps out into a power slide / drift!*
  - Lateral tire forces via Pacejka / smooth saturation curve:
    $$F_{yf} = F_{\max, f} \cdot \sin\left(C \arctan(B \alpha_f)\right)$$
    $$F_{yr} = F_{yr, \max} \cdot \sin\left(C \arctan(B \alpha_r)\right)$$

- **Equations of Motion (Planar Dynamic Integration)**:
  $$\dot{v}_x = \frac{1}{m}\left(F_{xr} + F_{xf}\cos\delta - F_{yf}\sin\delta - F_{\text{drag}}\right) + v_y \omega$$
  $$\dot{v}_y = \frac{1}{m}\left(F_{yf}\cos\delta + F_{xf}\sin\delta + F_{yr}\right) - v_x \omega$$
  $$\dot{\omega} = \frac{1}{I_z}\left(l_f (F_{yf}\cos\delta + F_{xf}\sin\delta) - l_r F_{yr}\right)$$
  $$\dot{x} = v_x \cos\theta - v_y \sin\theta$$
  $$\dot{y} = v_x \sin\theta + v_y \cos\theta$$
  $$\dot{\theta} = \omega$$

### 2. Goal Conditioning & Observations (`car_rl/env/car_env.py`)
- **Desired Direction Unit Vector**:
  - In vehicle ego-frame: $[u_x^{\text{ego}}, u_y^{\text{ego}}] = [\cos(\theta_{\text{des}} - \theta), \sin(\theta_{\text{des}} - \theta)]$.
- **Desired Velocity**: $v_{\text{desired}}$ (scalar target speed).
- **Vehicle Dynamic Features**:
  - Forward velocity $v_x / v_{\max}$
  - Lateral slip velocity $v_y / v_{\max}$ (allows agent to sense and control drifts)
  - Speed magnitude error: $(v_{\text{desired}} - \sqrt{v_x^2 + v_y^2}) / v_{\max}$
  - Wheel steering angle $\delta / \delta_{\max}$
  - Yaw rate $\omega / \omega_{\max}$

### 3. Reward Function
- **Direction Alignment Reward**:
  $$R_{\text{dir}} = \frac{\vec{v}_{\text{world}} \cdot \vec{u}}{\max(\|\vec{v}\|, 0.1)}$$
  Rewards the car when its actual travel velocity vector aligns with the commanded unit vector (whether going straight or drifting around a tight corner!).
- **Velocity Tracking Reward**:
  $$R_{\text{vel}} = -\left(\frac{\|\vec{v}\| - v_{\text{desired}}}{v_{\max}}\right)^2$$
- **Stability & Control Penalties**:
  - Small penalties on excessive unrecoverable spin ($\omega^2$) and control chattering.

### 4. Zero-Copy GPU DLPack Bridge (`car_rl/env/torch_wrapper.py`)
- Direct GPU tensor transfer between JAX and PyTorch:
  - `torch.from_dlpack(jax.dlpack.to_dlpack(...))`
  - `jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(...))`
- JAX memory fraction set to `0.4` to allow PyTorch to allocate CUDA memory without conflicts on the RTX 3060.

### 5. Vectorized PPO Policy (`car_rl/training/ppo.py`)
- Continuous Actor-Critic network predicting:
  - Steering command $a_{\text{steer}} \in [-1, 1]$
  - Engine drive/brake torque $a_{\text{torque}} \in [-1, 1]$
- Rollout length: 64 steps across 2048 parallel environments on GPU.

### 6. Raycast Sensor (Future Hierarchical Agent Readiness) (`car_rl/env/raycast.py`)
- Batched 2D raycasting module in JAX.
- The future high-level agent will take raycast readings and output $([u_x, u_y], v_{\text{desired}})$ directly to this dynamic car agent.

---

## File Structure

```
Car/
├── plan.md                      # Detailed project specification and plan
├── requirements.txt             # PyTorch, JAX, CUDA, rendering dependencies
├── setup_env.sh                 # Script to create and configure the .venv
├── car_rl/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── gpu_config.py        # CUDA / JAX memory fraction setup
│   ├── env/
│   │   ├── __init__.py
│   │   ├── physics.py           # 2D dynamic vehicle physics (Pacejka slip, drift, torque)
│   │   ├── car_env.py           # Batched pure-functional JAX environment
│   │   ├── raycast.py           # Batched 2D raycasting module for future high agent
│   │   └── torch_wrapper.py     # Zero-copy DLPack JAX-to-PyTorch wrapper
│   ├── models/
│   │   ├── __init__.py
│   │   └── actor_critic.py      # PyTorch continuous Actor-Critic networks
│   ├── training/
│   │   ├── __init__.py
│   │   ├── config.py            # Hyperparameters dataclass
│   │   └── ppo.py               # Vectorized GPU PPO trainer
│   └── hierarchical/
│       ├── __init__.py
│       └── interface.py         # Interface definitions for the high-level agent
├── train.py                     # CLI entry point to train low-level agent
├── demo.py                      # Interactive 2D visualization with Pygame/Matplotlib
└── tests/
    ├── test_gpu_bridge.py       # Zero-copy DLPack verification test
    └── test_physics.py          # Physics dynamics sanity checks (slip & drift)
```

---

## Verification Plan

### Automated Tests
1. **Physics & Drift Verification**:
   - Test tire slip angle calculation under lateral velocity $v_y \ne 0$.
   - Test rear tire traction loss under high throttle (friction circle saturation $\to$ lateral sliding).
   - Test steady-state circular drift stability with counter-steering.
2. **GPU DLPack Bridge Benchmark**:
   - Verify GPU-only memory transit (no CPU roundtrips, no VRAM leak).
   - Test execution throughput of 2048 parallel environments.
3. **PPO Training Convergence**:
   - Run training and verify the policy tracks the unit vector and requested velocity, dynamically handling tire slip and transitions.

### Manual Verification
- Run `python demo.py`:
  - Visually inspect the car drifting through sharp direction changes, steering the front wheels into the slide (counter-steering), and applying engine torque to maintain target speed.
