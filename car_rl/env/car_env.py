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
    1. ex: Forward speed error (target_speed - vx) / v_scale
    2. ey: Lateral slip error (0 - vy) / v_scale
    3. Target speed: target_speed / v_scale
    4. cos_target_ego: cos of heading error relative to target direction
    5. sin_target_ego: sin of heading error relative to target direction (>0 turn left, <0 turn right)
    6. Current longitudinal speed: vx / v_scale
    7. Current lateral slip speed: vy / v_scale
    8. Steering angle: steer / max_steer
    9. Yaw rate: omega / omega_scale
    """
    veh = state.vehicle
    
    target_speed = jnp.sqrt(state.target_vx**2 + state.target_vy**2 + 1e-4)
    has_target = target_speed > 0.1
    
    # Target direction and relative heading error
    target_angle = jnp.arctan2(state.target_vy, state.target_vx)
    heading_err = jnp.arctan2(jnp.sin(target_angle - veh.theta), jnp.cos(target_angle - veh.theta))
    cos_target_ego = jnp.where(has_target, jnp.cos(heading_err), 1.0)
    sin_target_ego = jnp.where(has_target, jnp.sin(heading_err), 0.0)
    
    # Forward velocity error: target speed forward - current vx
    ex = target_speed - veh.vx
    # Lateral slip error: target lateral velocity is 0 (cars do not slide sideways)
    ey = -veh.vy
    
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
    Direction-aligning, anti-spin velocity tracking reward:
    - Strong incentive to turn and align forward heading with target direction
    - Forward speed tracking with zero steady-state error
    - Strict lateral slip and power-oversteer spinout prevention
    """
    target_speed = jnp.sqrt(target_vx**2 + target_vy**2 + 1e-4)
    target_angle = jnp.arctan2(target_vy, target_vx)
    heading_err = jnp.arctan2(jnp.sin(target_angle - veh.theta), jnp.cos(target_angle - veh.theta))
    cos_target_ego = jnp.cos(heading_err)
    
    # Forward speed error and lateral slip
    speed_err = jnp.abs(target_speed - veh.vx)
    norm_ey = jnp.abs(veh.vy) / params.v_norm_scale
    
    # Strong alignment incentive: +2.5 when facing target, -2.5 when facing away
    align_incentive = 2.5 * cos_target_ego
    
    # Reverse velocity penalty: strongly discourage driving in reverse (incentive to align forward!)
    reverse_pen = 4.0 * jnp.maximum(0.0, -veh.vx)
    
    # Decisive launch incentive: when below target speed, pointing forward (cos > 0.85),
    # and NOT sliding laterally, reward full throttle for rapid launch
    throttle_cmd = jnp.maximum(0.0, action[1])
    is_aligned = cos_target_ego > 0.85
    not_sliding = jnp.abs(veh.vy) < 0.4
    accel_incentive = jnp.where((target_speed - veh.vx > 1.0) & is_aligned & not_sliding, 2.0 * throttle_cmd, 0.0)
    
    # Overshoot penalty: only penalize over-speeding beyond 0.8 m/s (~3 km/h)
    overshoot_pen = jnp.where(veh.vx - target_speed > 0.8, 1.5 * (veh.vx - target_speed - 0.8), 0.0)
    
    # Direct speed tracking + lateral error tracking + alignment incentive
    r_track = 4.0 - 0.5 * speed_err + align_incentive + accel_incentive - overshoot_pen - reverse_pen - 2.0 * norm_ey
    
    # Dual-scale exponential velocity tracking bonus:
    # Conditioned on being aligned forward (cos > 0.7) to lock in exact speed without spinning mid-corner
    total_err = jnp.sqrt((target_speed - veh.vx)**2 + veh.vy**2 + 1e-4)
    r_bonus = jnp.where(
        cos_target_ego > 0.7,
        2.0 * jnp.exp(-total_err / 2.5) + 4.0 * jnp.exp(-total_err / 0.7),
        0.0,
    )
    
    # Straight-line stability penalty: when aligned, penalize unnecessary steering wiggle
    steer_norm = jnp.abs(veh.steer / params.vehicle.max_steer)
    straight_steer_penalty = jnp.where(is_aligned & not_sliding, 1.0 * steer_norm, 0.0)
    
    # Lateral slip penalty (loss of tire grip)
    slip_pen = 3.0 * jnp.maximum(0.0, jnp.abs(veh.vy) - 0.4)
    
    # High-speed yaw stability penalty:
    mu_g = params.vehicle.friction_coeff * params.vehicle.gravity
    max_safe_omega = mu_g / jnp.maximum(jnp.abs(veh.vx), 2.0) + 0.3
    excess_omega = jnp.maximum(0.0, jnp.abs(veh.omega) - max_safe_omega)
    stability_pen = 3.0 * excess_omega
    
    # Power-oversteer penalty: cut throttle if vehicle begins sliding
    power_oversteer_pen = jnp.where(jnp.abs(veh.vy) > 0.4, 2.5 * throttle_cmd * (jnp.abs(veh.vy) - 0.4), 0.0)
    
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
    3-way balanced reset distribution across all angles [-pi, pi]:
    - Mode 0 (33%): Standstill launch with random target direction (v0 = 0, theta0 in [-pi, pi])
      -> trains turning and launching forward from any angle
    - Mode 1 (33%): Aligned steady-state cruising (v0 = target_speed, theta = target_angle)
      -> trains zero-error cruising
    - Mode 2 (34%): Dynamic random transitions (v0 in [0, max], theta in [-pi, pi])
      -> trains high-speed cornering, braking before turns, and turning without spinning
    """
    rng_mode, rng_goal, rng_init, rng_spd, rng_out = jax.random.split(rng, 5)
    
    target_vx, target_vy, duration, rng_goal = sample_goal(rng_goal, params)
    target_speed = jnp.sqrt(target_vx**2 + target_vy**2 + 1e-4)
    target_angle = jnp.arctan2(target_vy, target_vx)
    
    mode = jax.random.randint(rng_mode, shape=(), minval=0, maxval=3)
    
    th0_rand = jax.random.uniform(rng_init, shape=(), minval=-jnp.pi, maxval=jnp.pi)
    v0_rand = jax.random.uniform(rng_spd, shape=(), minval=0.0, maxval=params.max_desired_vel)
    
    th0 = jnp.where(mode == 1, target_angle, th0_rand)
    v0 = jnp.where(mode == 0, 0.0, jnp.where(mode == 1, target_speed, v0_rand))
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
    
    # Normalize reward by the duration for which it had to follow this particular vector
    # duration_factor in [1.0, 5.0] (corresponding to 3.0s to 15.0s)
    # Equalizes return across short & long durations, forcing immediate full-throttle launch
    duration_factor = state.goal_duration.astype(jnp.float32) / params.min_goal_duration_steps
    scaled_reward = base_reward / duration_factor
    
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
    reward = jnp.where(is_spinout, scaled_reward - 5.0, scaled_reward)
    
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
