"""
2D Dynamic Vehicle Physics Engine in JAX.

Simulates planar vehicle dynamics with nonlinear tire friction (Pacejka/Coupled Ellipse),
enabling realistic acceleration, braking, lateral tire slip, power oversteer, and drifting.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp


class VehicleParams(NamedTuple):
    # Dimensions & Mass
    mass: float = 1200.0          # Total vehicle mass (kg)
    iz: float = 1800.0            # Yaw moment of inertia (kg*m^2)
    lf: float = 1.2               # Distance from CG to front axle (m)
    lr: float = 1.3               # Distance from CG to rear axle (m)
    wheel_radius: float = 0.32    # Tire radius (m)
    
    # Steering limits
    max_steer: float = 0.60       # Max front wheel angle (~34.4 degrees)
    steer_rate: float = 3.0       # Max steering angular speed (rad/s)
    
    # Torque limits: pure torque min -600 Nm, max 2400 Nm
    min_torque: float = -600.0   # Max reverse torque (N*m)
    max_torque: float = 2400.0   # Max forward drive torque (N*m)
    
    # Resistances
    drag_coeff: float = 0.40      # 0.5 * rho * Cd * A (kg/m)
    roll_res_coeff: float = 0.015 # Rolling resistance coefficient
    gravity: float = 9.81         # m/s^2
    
    # Tire Friction (Coupled nonlinear model)
    friction_coeff: float = 1.05  # Tire-road peak friction coefficient (mu)
    tire_cornering_f: float = 40000.0  # Front tire cornering stiffness (N/rad)
    tire_cornering_r: float = 45000.0  # Rear tire cornering stiffness (N/rad)
    pacejka_c: float = 1.3        # Pacejka shape factor C
    pacejka_b: float = 10.0       # Pacejka stiffness factor B
    
    # Simulation
    dt: float = 0.02              # Integration timestep (50 Hz)


class VehicleState(NamedTuple):
    x: jnp.ndarray          # Global X position (m)
    y: jnp.ndarray          # Global Y position (m)
    theta: jnp.ndarray      # Global yaw / heading angle (rad)
    vx: jnp.ndarray         # Longitudinal velocity in body frame (m/s)
    vy: jnp.ndarray         # Lateral velocity in body frame (m/s)
    omega: jnp.ndarray      # Yaw rate (rad/s)
    steer: jnp.ndarray      # Current front wheel steering angle (rad)


def init_vehicle_state(
    x: float = 0.0,
    y: float = 0.0,
    theta: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    omega: float = 0.0,
    steer: float = 0.0,
) -> VehicleState:
    """Initialize a vehicle state PyTree."""
    return VehicleState(
        x=jnp.array(x, dtype=jnp.float32),
        y=jnp.array(y, dtype=jnp.float32),
        theta=jnp.array(theta, dtype=jnp.float32),
        vx=jnp.array(vx, dtype=jnp.float32),
        vy=jnp.array(vy, dtype=jnp.float32),
        omega=jnp.array(omega, dtype=jnp.float32),
        steer=jnp.array(steer, dtype=jnp.float32),
    )


def compute_tire_forces(
    vx: jnp.ndarray,
    vy: jnp.ndarray,
    omega: jnp.ndarray,
    steer: jnp.ndarray,
    fx_cmd: jnp.ndarray,
    params: VehicleParams,
):
    """
    Computes front and rear tire forces with coupled longitudinal & lateral friction limits.
    
    fx_cmd: Commanded longitudinal force at driven wheels (>0 for drive, <0 for braking).
    """
    wheelbase = params.lf + params.lr
    # Normal axle loads
    fz_f = params.mass * params.gravity * (params.lr / wheelbase)
    fz_r = params.mass * params.gravity * (params.lf / wheelbase)
    
    # Peak friction capacity
    f_max_f = params.friction_coeff * fz_f
    f_max_r = params.friction_coeff * fz_r
    
    # Longitudinal force distribution (RWD: all drive torque goes to rear axle)
    fx_f_target = 0.0
    fx_r_target = fx_cmd
    
    # Clamp longitudinal forces to friction limits
    fx_f = jnp.clip(fx_f_target, -f_max_f, f_max_f)
    fx_r = jnp.clip(fx_r_target, -f_max_r, f_max_r)
    
    # Friction ellipse: Available lateral grip decreases when longitudinal force is high!
    # This is what causes power-oversteer drifts under high throttle or e-brake slides.
    fy_max_f = jnp.sqrt(jnp.maximum(10.0, f_max_f**2 - fx_f**2))
    fy_max_r = jnp.sqrt(jnp.maximum(10.0, f_max_r**2 - fx_r**2))
    
    # Regularized forward speed to prevent singularity at zero velocity
    v_norm = jnp.sqrt(vx**2 + vy**2)
    vx_safe = jnp.maximum(jnp.abs(vx), 0.5)
    
    # Tire slip angles (rad)
    alpha_f = steer - jnp.arctan2(vy + params.lf * omega, vx_safe)
    alpha_r = -jnp.arctan2(vy - params.lr * omega, vx_safe)
    
    # Pacejka lateral tire curve
    fy_f_raw = fy_max_f * jnp.sin(params.pacejka_c * jnp.arctan(params.pacejka_b * alpha_f))
    fy_r_raw = fy_max_r * jnp.sin(params.pacejka_c * jnp.arctan(params.pacejka_b * alpha_r))
    
    # Smooth fadeout of tire forces at near-zero velocity to prevent oscillation
    speed_factor = jnp.clip(v_norm / 0.5, 0.0, 1.0)
    fy_f = fy_f_raw * speed_factor
    fy_r = fy_r_raw * speed_factor
    
    return fx_f, fx_r, fy_f, fy_r, alpha_f, alpha_r


def step_physics(
    state: VehicleState,
    steer_cmd: jnp.ndarray,
    torque_cmd: jnp.ndarray,
    params: VehicleParams,
) -> VehicleState:
    """
    Pure functional 2D dynamic vehicle step (semi-implicit integration).
    
    steer_cmd: Desired steering normalized to [-1, 1]
    torque_cmd: Desired torque normalized to [-1, 1] (>0 throttle, <0 brake)
    """
    # 1. Update front wheel steering angle with realistic actuator rate limit
    target_steer = jnp.clip(steer_cmd, -1.0, 1.0) * params.max_steer
    steer_diff = target_steer - state.steer
    steer_delta = jnp.clip(steer_diff, -params.steer_rate * params.dt, params.steer_rate * params.dt)
    new_steer = state.steer + steer_delta
    
    # 2. Drivetrain: Pure torque between min_torque (-600 Nm) and max_torque (+2400 Nm)
    # Action torque_cmd is in [-1, 1]
    is_forward = torque_cmd >= 0.0
    torque = jnp.where(
        is_forward,
        torque_cmd * params.max_torque,
        torque_cmd * jnp.abs(params.min_torque),
    )
    fx_cmd_raw = torque / params.wheel_radius
    
    # 3. Electronic Stability Control (ESC) / Traction Control System (TCS)
    # Automatically cuts positive drive torque if rear tires slip laterally (|vy| > 0.4 m/s)
    # or yaw rate exceeds 1.0 rad/s. Prevents power-oversteer spins and fishtailing!
    slip_excess = jnp.maximum(0.0, jnp.abs(state.vy) - 0.4)
    omega_excess = jnp.maximum(0.0, jnp.abs(state.omega) - 1.0)
    tcs_cut = jnp.clip(1.0 - 2.5 * slip_excess - 1.5 * omega_excess, 0.0, 1.0)
    fx_cmd = jnp.where(is_forward, fx_cmd_raw * tcs_cut, fx_cmd_raw)
    
    # 4. Compute tire forces with friction ellipse & slip angles
    fx_f, fx_r, fy_f, fy_r, _, _ = compute_tire_forces(
        state.vx, state.vy, state.omega, new_steer, fx_cmd, params
    )
    
    # 5. Aerodynamic drag and rolling resistance
    v_norm = jnp.sqrt(state.vx**2 + state.vy**2)
    f_drag = params.drag_coeff * state.vx * v_norm
    f_roll = params.roll_res_coeff * params.mass * params.gravity * jnp.tanh(5.0 * state.vx)
    
    # 6. Planar equations of motion (body frame)
    cos_steer = jnp.cos(new_steer)
    sin_steer = jnp.sin(new_steer)
    
    acc_x = (fx_r + fx_f * cos_steer - fy_f * sin_steer - f_drag - f_roll) / params.mass + state.vy * state.omega
    acc_y = (fy_r + fy_f * cos_steer + fx_f * sin_steer) / params.mass - state.vx * state.omega
    alpha_yaw = (params.lf * (fy_f * cos_steer + fx_f * sin_steer) - params.lr * fy_r) / params.iz
    
    # 7. Integrate body velocities and yaw rate
    new_vx = state.vx + acc_x * params.dt
    new_vy = state.vy + acc_y * params.dt
    new_omega = state.omega + alpha_yaw * params.dt
    
    # Standstill locking: prevent crawl when torque command is zero at low speed
    is_neutral = jnp.abs(torque_cmd) < 0.04
    is_stationary = (jnp.abs(new_vx) < 0.08) & (jnp.abs(new_vy) < 0.08)
    new_vx = jnp.where(is_stationary & is_neutral, 0.0, new_vx)
    new_vy = jnp.where(is_stationary & is_neutral, 0.0, new_vy)
    new_omega = jnp.where(is_stationary & is_neutral, 0.0, new_omega)
    
    # 7. Integrate global position and heading
    cos_theta = jnp.cos(state.theta)
    sin_theta = jnp.sin(state.theta)
    
    # World velocities: v_world = R(theta) * v_body
    vx_world = new_vx * cos_theta - new_vy * sin_theta
    vy_world = new_vx * sin_theta + new_vy * cos_theta
    
    new_x = state.x + vx_world * params.dt
    new_y = state.y + vy_world * params.dt
    new_theta = (state.theta + new_omega * params.dt + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    
    return VehicleState(
        x=new_x,
        y=new_y,
        theta=new_theta,
        vx=new_vx,
        vy=new_vy,
        omega=new_omega,
        steer=new_steer,
    )
