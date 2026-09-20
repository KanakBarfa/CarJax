"""
High-level track environment (pure JAX).

Loop per step:
  high_action [a_head, a_speed] (ego)
    -> world target velocity [tvx, tvy]
    -> frozen low-level torch policy -> [steer, torque]
    -> step_physics + railing collision
    -> 11-dim high obs [8 rays, vx, vy, omega] + progress reward

Reward = fastest laps: delta arc-length along centerline (m) scaled,
minus wall-hit penalty and small slip penalty. No map/position given
to the policy, so random tracks force lidar driving.
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp

from car_rl.env.physics import VehicleParams, VehicleState, init_vehicle_state, step_physics
from car_rl.tracks.generator import generate_track, NUM_CENTER
from car_rl.env.track_raycast import (
    RAY_ANGLES,
    NUM_RAYS,
    MAX_RANGE,
    cast_rays_against_segments,
    closest_wall,
)

# High obs dim: 9 rays + vx + vy + omega = 12
HIGH_OBS_DIM = NUM_RAYS + 3
HIGH_ACT_DIM = 2


class TrackParams(NamedTuple):
    vehicle: VehicleParams = VehicleParams()
    half_width: float = 6.0
    max_range: float = MAX_RANGE
    v_scale: float = 25.0
    omega_scale: float = 4.0
    max_desired_vel: float = 22.0
    max_reverse_vel: float = 8.0  # tail-first target speed; driver turns toward it
    car_radius: float = 1.1
    max_steps: int = 3000  # 60 s at 50 Hz: laps (~450 m) must be completable
    progress_scale: float = 2.0   # reward per meter advanced
    wall_penalty: float = 50.0    # crash must wipe ~100 steps of progress, not 10
    lap_bonus: float = 100.0      # finishing is the goal; makes survival worthwhile
    slip_penalty: float = 0.2


class TrackState(NamedTuple):
    vehicle: VehicleState
    center: jnp.ndarray    # (256, 2)
    walls: jnp.ndarray     # (512, 4)
    s_cum: jnp.ndarray     # (256,)
    track_length: jnp.ndarray
    closest_idx: jnp.ndarray
    prev_s: jnp.ndarray
    lap: jnp.ndarray
    step_count: jnp.ndarray
    total_reward: jnp.ndarray
    rng: jax.random.PRNGKey


def high_to_target(
    high_action: jnp.ndarray,
    theta: jnp.ndarray,
    max_vel: float = 22.0,
    max_rev: float = 8.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Ego heading+speed -> world target velocity.

    Speed stick is bidirectional: [0, +1] -> forward target 0..max_vel,
    [-1, 0] -> backward target 0..max_rev (points behind the car).
    Returns tvx, tvy, des_speed (signed), offset.
    """
    a_head = jnp.clip(high_action[0], -1.0, 1.0)
    a_speed = jnp.clip(high_action[1], -1.0, 1.0)
    offset = a_head * jnp.pi
    des_speed = jnp.where(a_speed >= 0.0, a_speed * max_vel, a_speed * max_rev)
    ang = theta + offset
    return des_speed * jnp.cos(ang), des_speed * jnp.sin(ang), des_speed, offset


def compute_low_obs(
    vehicle: VehicleState,
    tvx: jnp.ndarray,
    tvy: jnp.ndarray,
    v_scale: float = 25.0,
    omega_scale: float = 4.0,
    max_steer: float = 0.60,
) -> jnp.ndarray:
    """Same 8-dim ego observation the low-level driver was trained on."""
    target_speed = jnp.sqrt(tvx**2 + tvy**2 + 1e-4)
    has_target = target_speed > 0.1
    target_angle = jnp.arctan2(tvy, tvx)
    herr = jnp.arctan2(
        jnp.sin(target_angle - vehicle.theta), jnp.cos(target_angle - vehicle.theta)
    )
    ex = target_speed - vehicle.vx
    return jnp.array(
        [
            ex / v_scale,
            target_speed / v_scale,
            jnp.where(has_target, jnp.cos(herr), 1.0),
            jnp.where(has_target, jnp.sin(herr), 0.0),
            vehicle.vx / v_scale,
            vehicle.vy / v_scale,
            vehicle.steer / max_steer,
            vehicle.omega / omega_scale,
        ],
        dtype=jnp.float32,
    )


def compute_high_obs(state: TrackState, params: TrackParams) -> jnp.ndarray:
    """12-dim: [9 normalized rays, vx/s, vy/s, omega/s]. No position, no map."""
    rays = cast_rays_against_segments(
        state.vehicle.x,
        state.vehicle.y,
        state.vehicle.theta,
        state.walls,
        RAY_ANGLES,
        params.max_range,
    )
    v = state.vehicle
    tail = jnp.array(
        [v.vx / params.v_scale, v.vy / params.v_scale, v.omega / params.omega_scale],
        dtype=jnp.float32,
    )
    return jnp.concatenate([rays.astype(jnp.float32), tail])


def _closest_idx_s(
    x: jnp.ndarray, y: jnp.ndarray, center: jnp.ndarray, s_cum: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    d2 = (center[:, 0] - x) ** 2 + (center[:, 1] - y) ** 2
    idx = jnp.argmin(d2)
    return idx, s_cum[idx]


def track_reset(
    rng: jax.random.PRNGKey, params: TrackParams = TrackParams()
) -> Tuple[TrackState, jnp.ndarray]:
    rng_track, rng_out = jax.random.split(rng, 2)
    tr = generate_track(rng_track, half_width=params.half_width)
    veh = init_vehicle_state(
        x=tr.start_x, y=tr.start_y, theta=tr.start_theta, vx=0.0
    )
    idx, s = _closest_idx_s(tr.start_x, tr.start_y, tr.center, tr.s_cum)
    state = TrackState(
        vehicle=veh,
        center=tr.center,
        walls=tr.walls,
        s_cum=tr.s_cum,
        track_length=tr.length,
        closest_idx=idx,
        prev_s=s,
        lap=jnp.array(0, dtype=jnp.int32),
        step_count=jnp.array(0, dtype=jnp.int32),
        total_reward=jnp.array(0.0, dtype=jnp.float32),
        rng=rng_out,
    )
    return state, compute_high_obs(state, params)


def track_step(
    state: TrackState,
    low_action: jnp.ndarray,
    params: TrackParams = TrackParams(),
) -> Tuple[TrackState, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
    """Advance physics + railings + progress. No auto-reset (wrapper handles)."""
    steer_cmd = jnp.clip(low_action[0], -1.0, 1.0)
    torque_cmd = jnp.clip(low_action[1], -1.0, 1.0)
    veh = step_physics(state.vehicle, steer_cmd, torque_cmd, params.vehicle)

    # --- Railing collision: any touch ends the episode, so just push the
    # car back inside for a clean terminal state and scrub velocity ---
    dist, cx, cy = closest_wall(veh.x, veh.y, state.walls)
    hit = dist < params.car_radius
    push = params.car_radius - dist
    nx = jnp.where(dist > 1e-6, (veh.x - cx) / (dist + 1e-6), 1.0)
    ny = jnp.where(dist > 1e-6, (veh.y - cy) / (dist + 1e-6), 0.0)
    veh = veh._replace(
        x=jnp.where(hit, veh.x + nx * push, veh.x),
        y=jnp.where(hit, veh.y + ny * push, veh.y),
        vx=jnp.where(hit, veh.vx * 0.3, veh.vx),
        vy=jnp.where(hit, veh.vy * 0.3, veh.vy),
        omega=jnp.where(hit, veh.omega * 0.5, veh.omega),
    )

    # --- Progress along centerline ---
    idx, s = _closest_idx_s(veh.x, veh.y, state.center, state.s_cum)
    raw_ds = s - state.prev_s
    half = state.track_length * 0.5
    ds = jnp.where(raw_ds < -half, raw_ds + state.track_length, raw_ds)
    ds = jnp.where(ds > half, ds - state.track_length, ds)
    ds = jnp.clip(ds, -5.0, 5.0)
    crossed = (state.prev_s > state.track_length * 0.75) & (s < state.track_length * 0.25)
    new_lap = state.lap + jnp.where(crossed, 1, 0)

    reward = (
        ds * params.progress_scale
        - jnp.where(hit, params.wall_penalty, 0.0)
        + jnp.where(crossed, params.lap_bonus, 0.0)
        - params.slip_penalty * jnp.abs(veh.vy) / params.v_scale
    )

    new_step = state.step_count + 1
    # Terminal on wall touch (crash) or timeout. No grinding through walls.
    done = (new_step >= params.max_steps) | hit
    next_state = TrackState(
        vehicle=veh,
        center=state.center,
        walls=state.walls,
        s_cum=state.s_cum,
        track_length=state.track_length,
        closest_idx=idx,
        prev_s=s,
        lap=new_lap,
        step_count=new_step,
        total_reward=state.total_reward + reward,
        rng=state.rng,
    )
    obs = compute_high_obs(next_state, params)
    info = {
        "speed": jnp.sqrt(veh.vx**2 + veh.vy**2),
        "ds": ds,
        "progress_s": s,
        "lap": new_lap,
        "wall_hit": hit.astype(jnp.float32),
        "lapped": crossed.astype(jnp.float32),
        "track_length": state.track_length,
        "total_reward": next_state.total_reward,
    }
    return next_state, obs, reward, done, info
