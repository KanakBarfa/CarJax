"""
Procedural closed-loop track generator (pure JAX, jit + vmap friendly).

Design (locked from user spec):
- Closed-loop F1-style circuits, regenerated randomly every episode so the
  high-level agent cannot memorize a track and must drive from lidar.
- Fixed shapes for vectorization:
    NUM_CONTROL = 10 control points
    NUM_CENTER  = 256 centerline points (Catmull-Rom spline, closed)
    Walls: left + right railing polylines -> (2*NUM_CENTER, 4) segments
- Track width: total 12 m (half_width = 6 m). Railings are hard walls.
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp

NUM_CONTROL = 10
NUM_CENTER = 256
HALF_WIDTH = 6.0
BASE_RADIUS = 65.0
RADIUS_NOISE = 30.0   # wide range: fast sweepers to sharp hairpins
ANGLE_JITTER = 0.27   # kept < half spacing (pi/10 ~ 0.31) to avoid loops


class TrackData(NamedTuple):
    center: jnp.ndarray      # (NUM_CENTER, 2) centerline points
    left: jnp.ndarray        # (NUM_CENTER, 2) left railing polyline
    right: jnp.ndarray       # (NUM_CENTER, 2) right railing polyline
    walls: jnp.ndarray       # (2*NUM_CENTER, 4) segments [x1,y1,x2,y2]
    s_cum: jnp.ndarray       # (NUM_CENTER,) cumulative arc length (m)
    length: jnp.ndarray      # scalar total length (m)
    start_x: jnp.ndarray
    start_y: jnp.ndarray
    start_theta: jnp.ndarray


def _catmull_rom(p0, p1, p2, p3, f):
    """Single Catmull-Rom sample. Points shaped (..., 2), f scalar in [0,1)."""
    f2 = f * f
    f3 = f2 * f
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * f
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * f2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * f3
    )


def generate_track(
    rng: jax.random.PRNGKey,
    half_width: float = HALF_WIDTH,
) -> TrackData:
    """Pure-functional random closed-loop track from an RNG key.

    Every call randomly mirrors the layout (kills left/right bias) and
    randomly flips the race direction (clockwise vs anticlockwise), so the
    high-level agent must drive from lidar, not memorize turn sequences.
    """
    rng_r, rng_a, rng_f = jax.random.split(rng, 3)
    rng_m, rng_d = jax.random.split(rng_f, 2)

    base_angles = jnp.linspace(0.0, 2.0 * jnp.pi, NUM_CONTROL, endpoint=False)
    jitter = jax.random.uniform(
        rng_a, shape=(NUM_CONTROL,), minval=-ANGLE_JITTER, maxval=ANGLE_JITTER
    )
    angles = base_angles + jitter
    radii = BASE_RADIUS + jax.random.uniform(
        rng_r, shape=(NUM_CONTROL,), minval=-RADIUS_NOISE, maxval=RADIUS_NOISE
    )
    control = jnp.stack([radii * jnp.cos(angles), radii * jnp.sin(angles)], axis=1)

    # Random mirror (negate y) -> left/right corner bias eliminated
    mirror = jax.random.bernoulli(rng_m, 0.5)
    control = jnp.where(mirror, control * jnp.array([1.0, -1.0]), control)

    # Uniform parameter samples over [0, NUM_CONTROL)
    t = jnp.linspace(0.0, float(NUM_CONTROL), NUM_CENTER, endpoint=False)
    seg = jnp.floor(t).astype(jnp.int32) % NUM_CONTROL
    frac = t - jnp.floor(t)

    def sample_one(s, f):
        p0 = control[(s - 1) % NUM_CONTROL]
        p1 = control[s % NUM_CONTROL]
        p2 = control[(s + 1) % NUM_CONTROL]
        p3 = control[(s + 2) % NUM_CONTROL]
        return _catmull_rom(p0, p1, p2, p3, f)

    center = jax.vmap(sample_one)(seg, frac)  # (256, 2)

    # Random race direction: traverse the loop clockwise or anticlockwise.
    # Start line stays at index 0, tangent follows automatically.
    reverse = jax.random.bernoulli(rng_d, 0.5)
    center = jnp.where(reverse, center[::-1], center)

    # Tangents (central differences on closed loop) -> normals
    nxt = jnp.roll(center, -1, axis=0)
    prv = jnp.roll(center, 1, axis=0)
    tangent = nxt - prv
    t_norm = jnp.sqrt(jnp.sum(tangent**2, axis=1, keepdims=True) + 1e-6)
    tangent = tangent / t_norm
    normal = jnp.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

    left = center + normal * half_width
    right = center - normal * half_width

    left_next = jnp.roll(left, -1, axis=0)
    right_next = jnp.roll(right, -1, axis=0)
    left_segs = jnp.concatenate([left, left_next], axis=1)    # (256, 4)
    right_segs = jnp.concatenate([right, right_next], axis=1)  # (256, 4)
    walls = jnp.concatenate([left_segs, right_segs], axis=0)   # (512, 4)

    # Arc length table for progress rewards
    seg_vec = nxt - center
    seg_len = jnp.sqrt(jnp.sum(seg_vec**2, axis=1) + 1e-9)
    s_cum = jnp.cumsum(seg_len) - seg_len  # s_cum[0] = 0
    length = jnp.sum(seg_len)

    t0 = tangent[0]
    start_theta = jnp.arctan2(t0[1], t0[0])

    return TrackData(
        center=center,
        left=left,
        right=right,
        walls=walls,
        s_cum=s_cum,
        length=length,
        start_x=center[0, 0],
        start_y=center[0, 1],
        start_theta=start_theta,
    )


def empty_track() -> TrackData:
    """Zero track with correct shapes (for tree_map init)."""
    z2 = jnp.zeros((NUM_CENTER, 2), dtype=jnp.float32)
    return TrackData(
        center=z2,
        left=z2,
        right=z2,
        walls=jnp.zeros((2 * NUM_CENTER, 4), dtype=jnp.float32),
        s_cum=jnp.zeros((NUM_CENTER,), dtype=jnp.float32),
        length=jnp.array(0.0, dtype=jnp.float32),
        start_x=jnp.array(0.0, dtype=jnp.float32),
        start_y=jnp.array(0.0, dtype=jnp.float32),
        start_theta=jnp.array(0.0, dtype=jnp.float32),
    )
