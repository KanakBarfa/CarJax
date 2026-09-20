"""
Batched 2D JAX Car Environment for Reinforcement Learning.

Key features for accurate velocity & heading tracking:
- Target velocity range [0, 22] m/s (covers complete stop up to 80 km/h).
- Initial vehicle speed randomized across [0, 22] m/s so the agent learns BOTH
  acceleration (ex > 0) and aggressive braking/deceleration (ex < 0) equally.
- Straight-line alignment penalty eliminates directional drift/offset when aligned.
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
from car_rl.env.physics import (
    VehicleParams,
    VehicleState,
    init_vehicle_state,
    step_physics,
)


class EnvParams(NamedTuple):
    # Physics parameters
    vehicle: VehicleParams = VehicleParams()
    
    # Goal sampling limits (from 0 to 80 km/h)
    min_desired_vel: float = 0.0    # m/s (allows learning full stops)
    max_desired_vel: float = 22.0   # m/s (~80 km/h)
    v_norm_scale: float = 25.0      # Velocity normalization scale
    omega_norm_scale: float = 4.0   # Yaw rate normalization scale
    
    # Episode parameters (random target duration from 3 to 15 seconds)
    max_episode_steps: int = 750        # Up to 15.0 seconds at 50 Hz
    min_goal_duration_steps: int = 150  # 3.0 seconds at 50 Hz
    max_goal_duration_steps: int = 750  # 15.0 seconds at 50 Hz


class EnvState(NamedTuple):
    vehicle: VehicleState
    target_vx: jnp.ndarray      # World-frame target velocity X (m/s)
    target_vy: jnp.ndarray      # World-frame target velocity Y (m/s)
    step_count: jnp.ndarray     # Current episode step
    goal_timer: jnp.ndarray     # Steps elapsed under current target
    goal_duration: jnp.ndarray  # Total duration to hold target (150 to 750 steps / 3s to 15s)
    total_reward: jnp.ndarray   # Cumulative reward
    rng: jax.random.PRNGKey     # Random key for resets/resampling


def sample_goal(rng: jax.random.PRNGKey, params: EnvParams) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
    """Sample a random 2D target velocity vector across all angles [-pi, pi] and duration [3s, 15s]."""
    rng_angle, rng_speed, rng_dur, rng_out = jax.random.split(rng, 4)
    angle = jax.random.uniform(rng_angle, shape=(), minval=-jnp.pi, maxval=jnp.pi)
    speed = jax.random.uniform(
        rng_speed, shape=(), minval=params.min_desired_vel, maxval=params.max_desired_vel
    )
    duration = jax.random.randint(
        rng_dur, shape=(), minval=params.min_goal_duration_steps, maxval=params.max_goal_duration_steps
    )
    target_vx = speed * jnp.cos(angle)
    target_vy = speed * jnp.sin(angle)
    return target_vx, target_vy, duration, rng_out


def compute_observation(state: EnvState, params: EnvParams) -> jnp.ndarray:
    """
    Constructs the 9-dimensional ego-centric observation vector:
    1. ex: Longitudinal velocity error in car frame (v_target_x_ego - vx) / v_scale
    2. ey: Lateral velocity error in car frame (v_target_y_ego - vy) / v_scale
    3. Target speed: ||v_target|| / v_scale
    4. cos_target_ego: cos of relative target angle in car body frame
    5. sin_target_ego: sin of relative target angle in car body frame
    6. Current longitudinal speed: vx / v_scale
    7. Current lateral slip speed: vy / v_scale
    8. Steering angle: steer / max_steer
    9. Yaw rate: omega / omega_scale
    """
    veh = state.vehicle
    
    cos_th = jnp.cos(veh.theta)
    sin_th = jnp.sin(veh.theta)
    
    # Target velocity vector rotated into vehicle body frame
    ego_target_vx = state.target_vx * cos_th + state.target_vy * sin_th
    ego_target_vy = -state.target_vx * sin_th + state.target_vy * cos_th
    
    # Ego velocity errors
    ex = ego_target_vx - veh.vx
    ey = ego_target_vy - veh.vy
    
    target_speed = jnp.sqrt(state.target_vx**2 + state.target_vy**2 + 1e-4)
    has_target = target_speed > 0.1
    cos_target_ego = jnp.where(has_target, ego_target_vx / target_speed, 1.0)
    sin_target_ego = jnp.where(has_target, ego_target_vy / target_speed, 0.0)
    
    obs = jnp.array([
        ex / params.v_norm_scale,
        ey / params.v_norm_scale,
        target_speed / params.v_norm_scale,
        cos_target_ego,
        sin_target_ego,
        veh.vx / params.v_norm_scale,
        veh.vy / params.v_norm_scale,
        veh.steer / params.vehicle.max_steer,
        veh.omega / params.omega_norm_scale,
    ], dtype=jnp.float32)
    
    return obs


def compute_reward(
    veh: VehicleState,
    target_vx: jnp.ndarray,
    target_vy: jnp.ndarray,
    action: jnp.ndarray,
    params: EnvParams,
) -> jnp.ndarray:
    """
    Symmetric, calibrated velocity tracking reward with dynamic stability and spin prevention.
    """
    cos_th = jnp.cos(veh.theta)
    sin_th = jnp.sin(veh.theta)
    
    ego_target_vx = target_vx * cos_th + target_vy * sin_th
    ego_target_vy = -target_vx * sin_th + target_vy * cos_th
    
    # Longitudinal and lateral errors in vehicle frame
    ex = ego_target_vx - veh.vx
    ey = ego_target_vy - veh.vy
    
    speed_err = jnp.abs(ex)
    norm_ey = jnp.abs(ey) / params.v_norm_scale
    
    # Direct speed penalty + lateral error tracking:
    # Strongly penalizes longitudinal speed discrepancy (|ex|) to eliminate settling offset
    r_track = 3.0 - 0.6 * speed_err - 2.0 * norm_ey
    
    # Tight exponential velocity tracking bonus (Laplace kernel):
    # Scale sigma = 0.8 m/s (~2.8 km/h): Provides sharp, continuous gradient to zero error
    total_err = jnp.sqrt(ex**2 + ey**2 + 1e-4)
    r_bonus = 3.0 * jnp.exp(-total_err / 0.8)
    
    # Target heading alignment
    target_speed = jnp.sqrt(target_vx**2 + target_vy**2 + 1e-4)
    cos_target_ego = ego_target_vx / target_speed
    
    # Straight-line stability penalty: when lateral error is small (|ey| < 0.8) and pointing forward
    is_aligned = (jnp.abs(ey) < 0.8) & (cos_target_ego > 0.85)
    steer_norm = jnp.abs(veh.steer / params.vehicle.max_steer)
    straight_steer_penalty = jnp.where(is_aligned, 1.2 * steer_norm, 0.0)
    
    # Lateral slip penalty (loss of tire grip)
    # |vy| > 1.0 m/s indicates significant tire sliding
    slip_pen = 1.5 * jnp.maximum(0.0, jnp.abs(veh.vy) - 1.0)
    
    # High-speed yaw stability penalty:
    # Steady-state maximum yaw rate before loss of grip is mu * g / max(vx, 2.0)
    mu_g = params.vehicle.friction_coeff * params.vehicle.gravity
    max_safe_omega = mu_g / jnp.maximum(jnp.abs(veh.vx), 2.0) + 0.3
    excess_omega = jnp.maximum(0.0, jnp.abs(veh.omega) - max_safe_omega)
    stability_pen = 2.0 * excess_omega
    
    # Power-oversteer traction control penalty:
    # If the vehicle is experiencing lateral slip (|vy| > 1.2), hitting positive throttle induces spinning
    is_sliding = jnp.abs(veh.vy) > 1.2
    throttle_cmd = jnp.maximum(0.0, action[1])
    power_oversteer_pen = jnp.where(is_sliding, 1.5 * throttle_cmd * (jnp.abs(veh.vy) - 1.2), 0.0)
    
    # Control effort regularization
    ctrl_pen = 0.002 * jnp.sum(action**2)
    
    reward = (
        r_track 
        + r_bonus 
        - straight_steer_penalty 
        - slip_pen 
        - stability_pen 
        - power_oversteer_pen 
        - ctrl_pen
    )
    return reward


def env_reset(rng: jax.random.PRNGKey, params: EnvParams = EnvParams()) -> Tuple[EnvState, jnp.ndarray]:
    """
    Pure functional reset.
    50% of environments start in aligned steady-state cruising:
    theta = target_angle, vx = target_speed, vy = 0, omega = 0.
    50% of environments start in randomized speeds and orientations for dynamic transitions.
    Target speed and heading are held for a random duration between 3.0s and 15.0s (150 to 750 steps).
    """
    rng_mode, rng_goal, rng_init, rng_spd, rng_out = jax.random.split(rng, 5)
    
    target_vx, target_vy, duration, rng_goal = sample_goal(rng_goal, params)
    target_speed = jnp.sqrt(target_vx**2 + target_vy**2 + 1e-4)
    target_angle = jnp.arctan2(target_vy, target_vx)
    
    is_steady = jax.random.bernoulli(rng_mode, p=0.5)
    
    th0_rand = jax.random.uniform(rng_init, shape=(), minval=-jnp.pi, maxval=jnp.pi)
    v0_rand = jax.random.uniform(rng_spd, shape=(), minval=0.0, maxval=params.max_desired_vel)
    
    th0 = jnp.where(is_steady, target_angle, th0_rand)
    v0 = jnp.where(is_steady, target_speed, v0_rand)
    veh = init_vehicle_state(theta=th0, vx=v0)
    
    state = EnvState(
        vehicle=veh,
        target_vx=target_vx,
        target_vy=target_vy,
        step_count=jnp.array(0, dtype=jnp.int32),
        goal_timer=jnp.array(0, dtype=jnp.int32),
        goal_duration=duration,
        total_reward=jnp.array(0.0, dtype=jnp.float32),
        rng=rng_out,
    )
    obs = compute_observation(state, params)
    return state, obs


def env_step(
    state: EnvState,
    action: jnp.ndarray,
    params: EnvParams = EnvParams(),
) -> Tuple[EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
    """
    Pure functional step of a single environment.
    action: [steer_cmd, torque_cmd] in [-1, 1]^2
    """
    steer_cmd = jnp.clip(action[0], -1.0, 1.0)
    torque_cmd = jnp.clip(action[1], -1.0, 1.0)
    
    # 1. Physics integration
    new_veh = step_physics(state.vehicle, steer_cmd, torque_cmd, params.vehicle)
    
    # 2. Reward computation
    base_reward = compute_reward(
        new_veh, state.target_vx, state.target_vy, action, params
    )
    
    # 3. Dynamic goal resampling (triggers when goal_timer >= goal_duration, i.e. 3s to 15s)
    new_step = state.step_count + 1
    new_timer = state.goal_timer + 1
    should_resample = (new_timer >= state.goal_duration) & (new_step < params.max_episode_steps)
    
    resampled_vx, resampled_vy, new_dur, new_rng = sample_goal(state.rng, params)
    
    next_vx = jnp.where(should_resample, resampled_vx, state.target_vx)
    next_vy = jnp.where(should_resample, resampled_vy, state.target_vy)
    next_dur = jnp.where(should_resample, new_dur, state.goal_duration)
    next_timer = jnp.where(should_resample, 0, new_timer)
    
    # 4. Spinout detection and early termination
    is_spinout = (jnp.abs(new_veh.vy) > 4.5) & (jnp.abs(new_veh.omega) > 2.2)
    reward = jnp.where(is_spinout, base_reward - 5.0, base_reward)
    
    done = (new_step >= params.max_episode_steps) | is_spinout
    reset_state, reset_obs = env_reset(new_rng, params)
    
    next_state = EnvState(
        vehicle=new_veh,
        target_vx=next_vx,
        target_vy=next_vy,
        step_count=new_step,
        goal_timer=next_timer,
        goal_duration=next_dur,
        total_reward=state.total_reward + reward,
        rng=new_rng,
    )
    
    obs = compute_observation(next_state, params)
    
    final_state = jax.tree.map(
        lambda s_next, s_rst: jnp.where(done, s_rst, s_next),
        next_state,
        reset_state,
    )
    final_obs = jnp.where(done, reset_obs, obs)
    
    target_speed = jnp.sqrt(state.target_vx**2 + state.target_vy**2)
    info = {
        "speed": jnp.sqrt(new_veh.vx**2 + new_veh.vy**2),
        "desired_vel": target_speed,
        "slip_speed": jnp.abs(new_veh.vy),
        "total_reward": next_state.total_reward,
    }
    
    return final_state, final_obs, reward, done, info
