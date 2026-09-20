"""Sanity tests for random track game: generator, raycast, collision, wrapper."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")

import jax
import jax.numpy as jnp


def test_track_shapes():
    from car_rl.tracks.generator import generate_track, NUM_CENTER

    tr = generate_track(jax.random.PRNGKey(0))
    assert tr.center.shape == (NUM_CENTER, 2), tr.center.shape
    assert tr.walls.shape == (2 * NUM_CENTER, 4), tr.walls.shape
    assert float(tr.length) > 200.0, float(tr.length)
    # Start sits on centerline -> ~half width from each wall
    from car_rl.env.track_raycast import closest_wall

    d, _, _ = closest_wall(tr.start_x, tr.start_y, tr.walls)
    assert 5.0 < float(d) < 7.0, float(d)
    # Different seeds -> different tracks (no memorization possible)
    tr2 = generate_track(jax.random.PRNGKey(1))
    assert float(jnp.abs(tr.center - tr2.center).mean()) > 1.0
    print("test_track_shapes OK, length:", round(float(tr.length), 1))


def test_raycast_hits_wall():
    from car_rl.tracks.generator import generate_track
    from car_rl.env.track_raycast import cast_rays_against_segments, RAY_ANGLES

    tr = generate_track(jax.random.PRNGKey(0))
    rays = cast_rays_against_segments(
        tr.start_x, tr.start_y, tr.start_theta, tr.walls, RAY_ANGLES, 35.0
    )
    assert rays.shape == (8,), rays.shape
    assert bool(jnp.all(rays >= 0.0) & jnp.all(rays <= 1.0))
    # Facing along the straight: side rays (~6 m) much shorter than forward ray
    assert float(rays[0]) < 0.3, rays  # -90 deg side wall ~6/35
    print("test_raycast_hits_wall OK:", [round(float(r), 2) for r in rays])


def test_railing_blocks_exit():
    from car_rl.env.track_env import track_reset, track_step, TrackParams

    p = TrackParams()
    st, _ = track_reset(jax.random.PRNGKey(0), p)
    # Ram full throttle into the left wall for 10 s
    a = jnp.array([0.0, 1.0], dtype=jnp.float32)
    # steer hard left via low-level action directly
    a_left = jnp.array([-1.0, 1.0], dtype=jnp.float32)
    hit_seen = False
    for _ in range(500):
        st, _, _, _, info = track_step(st, a_left, p)
        hit_seen = hit_seen or bool(info["wall_hit"] > 0)
    from car_rl.env.track_raycast import closest_wall

    d, _, _ = closest_wall(st.vehicle.x, st.vehicle.y, st.walls)
    assert float(d) >= p.car_radius - 1e-3, float(d)  # never outside railing
    assert hit_seen, "expected at least one wall contact when ramming sideways"
    print("test_railing_blocks_exit OK, min wall dist:", round(float(d), 2))


def test_high_obs_and_wrapper():
    import torch
    from car_rl.env.track_wrapper import TorchBatchedTrackEnv

    env = TorchBatchedTrackEnv(num_envs=8, seed=0, device="cuda:0")
    obs = env.reset()
    assert obs.shape == (8, 11), obs.shape
    assert torch.isfinite(obs).all()
    act = torch.zeros(8, 2, device="cuda:0")
    act[:, 1] = 1.0
    for _ in range(30):
        obs, rew, done, info = env.step(act)
    assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    assert info["speed"].mean().item() > 2.0, info["speed"].mean().item()
    print("test_high_obs_and_wrapper OK, speed:", round(info["speed"].mean().item(), 2))


if __name__ == "__main__":
    test_track_shapes()
    test_raycast_hits_wall()
    test_railing_blocks_exit()
    test_high_obs_and_wrapper()
    print("ALL TRACK TESTS PASSED")
