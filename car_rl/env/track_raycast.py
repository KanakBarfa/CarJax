"""
JAX segment raycaster for track railings + wall collision helpers.

High-level lidar spec (user locked):
- 8 rays total: 6 uniform across the front [-90..+90 deg ego],
  2 in the back at [+150, -150 deg ego].
- Normalized distances in [0, 1] (divided by max_range).
- Observations also include vx, vy, omega only (no map/position).
"""

import jax
import jax.numpy as jnp
from typing import Tuple

# 6 front uniform over [-90, +90], 2 back at +/-150 deg
RAY_ANGLES = jnp.array(
    [
        -jnp.pi / 2.0,          # -90 front-left
        -jnp.pi * 0.3,          # -54
        -jnp.pi * 0.1,          # -18
        jnp.pi * 0.1,           # +18
        jnp.pi * 0.3,           # +54
        jnp.pi / 2.0,           # +90 front-right
        jnp.pi * 5.0 / 6.0,     # +150 back-right
        -jnp.pi * 5.0 / 6.0,    # -150 back-left
    ],
    dtype=jnp.float32,
)
NUM_RAYS = 8
MAX_RANGE = 35.0


def ray_segment_t(
    px: jnp.ndarray,
    py: jnp.ndarray,
    dx: jnp.ndarray,
    dy: jnp.ndarray,
    x1: jnp.ndarray,
    y1: jnp.ndarray,
    x2: jnp.ndarray,
    y2: jnp.ndarray,
    max_range: float = MAX_RANGE,
) -> jnp.ndarray:
    """Ray distance to one segment, or max_range on miss. All scalars."""
    rx, ry = dx, dy
    sx, sy = x2 - x1, y2 - y1
    denom = rx * sy - ry * sx  # r x s
    qpx, qpy = x1 - px, y1 - py
    # Parallel -> miss
    safe = jnp.abs(denom) > 1e-9
    t = (qpx * sy - qpy * sx) / jnp.where(safe, denom, 1.0)
    u = (qpx * ry - qpy * rx) / jnp.where(safe, denom, 1.0)
    hit = safe & (t >= 0.0) & (t <= max_range) & (u >= 0.0) & (u <= 1.0)
    return jnp.where(hit, t, max_range)


def cast_rays_against_segments(
    px: jnp.ndarray,
    py: jnp.ndarray,
    theta: jnp.ndarray,
    walls: jnp.ndarray,  # (M, 4)
    ray_angles: jnp.ndarray = RAY_ANGLES,
    max_range: float = MAX_RANGE,
) -> jnp.ndarray:
    """Cast R rays from (px,py,theta) against M wall segments.

    Returns normalized distances (R,) in [0, 1].
    """
    dirs = theta + ray_angles  # (R,)
    rdx = jnp.cos(dirs)
    rdy = jnp.sin(dirs)

    def one_ray(dx, dy):
        ts = jax.vmap(
            lambda x1, y1, x2, y2: ray_segment_t(
                px, py, dx, dy, x1, y1, x2, y2, max_range
            )
        )(walls[:, 0], walls[:, 1], walls[:, 2], walls[:, 3])
        return jnp.min(ts)

    dists = jax.vmap(one_ray)(rdx, rdy)
    return dists / max_range


def point_segment_dist(
    px: jnp.ndarray,
    py: jnp.ndarray,
    x1: jnp.ndarray,
    y1: jnp.ndarray,
    x2: jnp.ndarray,
    y2: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Distance from point to segment + closest point. Scalars in/out."""
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    c = vx * vx + vy * vy + 1e-9
    t = jnp.clip((wx * vx + wy * vy) / c, 0.0, 1.0)
    cx, cy = x1 + t * vx, y1 + t * vy
    d = jnp.sqrt((px - cx) ** 2 + (py - cy) ** 2 + 1e-9)
    return d, cx, cy


def closest_wall(
    px: jnp.ndarray, py: jnp.ndarray, walls: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Min distance to any wall segment + closest point coords."""
    def one_seg(x1, y1, x2, y2):
        d, cx, cy = point_segment_dist(px, py, x1, y1, x2, y2)
        return d, cx, cy

    ds, cxs, cys = jax.vmap(one_seg)(
        walls[:, 0], walls[:, 1], walls[:, 2], walls[:, 3]
    )
    i = jnp.argmin(ds)
    return ds[i], cxs[i], cys[i]
