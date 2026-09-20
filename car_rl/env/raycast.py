"""
2D JAX Raycast Sensor Module (Hierarchical Agent Readiness).

Computes batched 2D lidar / raycast intersections against circles and line segments,
allowing a future higher-level agent to sense obstacles and emit target directions & speeds.
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp


class RaycastConfig(NamedTuple):
    num_rays: int = 16
    fov: float = jnp.pi          # 180 degrees field of view
    max_range: float = 30.0      # Maximum raycast distance (meters)


def compute_ray_directions(heading: jnp.ndarray, config: RaycastConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute (dx, dy) unit vectors for all rays relative to vehicle heading."""
    angles = jnp.linspace(-config.fov / 2.0, config.fov / 2.0, config.num_rays) + heading
    ray_dx = jnp.cos(angles)
    ray_dy = jnp.sin(angles)
    return ray_dx, ray_dy


def ray_circle_intersect(
    ray_ox: jnp.ndarray,
    ray_oy: jnp.ndarray,
    ray_dx: jnp.ndarray,
    ray_dy: jnp.ndarray,
    circle_x: jnp.ndarray,
    circle_y: jnp.ndarray,
    circle_r: jnp.ndarray,
    max_range: float,
) -> jnp.ndarray:
    """Computes ray-circle intersection distance in JAX."""
    # Vector from ray origin to circle center
    oc_x = circle_x - ray_ox
    oc_y = circle_y - ray_oy
    
    # Projection along ray
    t_proj = oc_x * ray_dx + oc_y * ray_dy
    
    # Perpendicular distance squared
    dist_perp_sq = (oc_x**2 + oc_y**2) - t_proj**2
    r_sq = circle_r**2
    
    # Discriminant
    has_hit = (dist_perp_sq <= r_sq) & (t_proj + circle_r > 0.0)
    d_offset = jnp.sqrt(jnp.maximum(0.0, r_sq - dist_perp_sq))
    t_hit = t_proj - d_offset
    
    # Valid hit within [0, max_range]
    valid = has_hit & (t_hit > 0.0) & (t_hit < max_range)
    return jnp.where(valid, t_hit, max_range)


def cast_rays_against_circles(
    veh_x: jnp.ndarray,
    veh_y: jnp.ndarray,
    veh_theta: jnp.ndarray,
    circles_x: jnp.ndarray,  # Shape: (N_circles,)
    circles_y: jnp.ndarray,  # Shape: (N_circles,)
    circles_r: jnp.ndarray,  # Shape: (N_circles,)
    config: RaycastConfig = RaycastConfig(),
) -> jnp.ndarray:
    """
    Casts rays from the vehicle against an array of circular obstacles.
    Returns: Array of shape (num_rays,) with distances in [0, 1] normalized.
    """
    ray_dx, ray_dy = compute_ray_directions(veh_theta, config)
    
    # Vmap over all rays and all circles
    def single_ray(dx, dy):
        # Distances to all circles for this one ray
        dists = jax.vmap(
            lambda cx, cy, cr: ray_circle_intersect(
                veh_x, veh_y, dx, dy, cx, cy, cr, config.max_range
            )
        )(circles_x, circles_y, circles_r)
        return jnp.min(dists)
        
    min_dists = jax.vmap(single_ray)(ray_dx, ray_dy)
    return min_dists / config.max_range
