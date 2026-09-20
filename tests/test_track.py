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
    assert rays.shape == (9,), rays.shape
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


def test_mirror_and_direction_variants():
    """Mirroring + race direction produce valid, two-handed track variety."""
    from car_rl.tracks.generator import generate_track

    signs = set()
    for seed in range(12):
        tr = generate_track(jax.random.PRNGKey(seed))
        assert float(tr.length) > 200.0
        # Shoelace signed area: mirror and direction flips change its sign
        x, y = tr.center[:, 0], tr.center[:, 1]
        area = float(0.5 * jnp.sum(x * jnp.roll(y, -1) - jnp.roll(x, -1) * y))
        signs.add(area > 0)
        # Start pose must sit on the centerline, heading along travel tangent
        # (env uses central differences: tangent[0] = center[1] - center[-1])
        d0 = float(jnp.sqrt((tr.center[0, 0] - tr.start_x) ** 2 + (tr.center[0, 1] - tr.start_y) ** 2))
        assert d0 < 1e-3, d0
        t = tr.center[1] - tr.center[-1]
        want = float(jnp.arctan2(t[1], t[0]))
        got = float((tr.start_theta + jnp.pi) % (2 * jnp.pi) - jnp.pi)
        err = abs((want - got + jnp.pi) % (2 * jnp.pi) - jnp.pi)
        assert err < 0.05, (seed, want, got)
    assert signs == {True, False}, "expected both loop orientations across seeds"
    print("test_mirror_and_direction_variants OK")


def test_wall_touch_ends_episode():
    """Any railing contact is terminal (crash) with a one-off penalty."""
    from car_rl.env.track_env import track_reset, track_step, TrackParams
    from car_rl.env.track_raycast import closest_wall as cw

    p = TrackParams()
    st, _ = track_reset(jax.random.PRNGKey(0), p)
    d0, cx0, cy0 = cw(st.vehicle.x, st.vehicle.y, st.walls)
    nx = (st.vehicle.x - cx0) / (d0 + 1e-6)
    ny = (st.vehicle.y - cy0) / (d0 + 1e-6)
    veh = st.vehicle._replace(
        x=cx0 + nx * (p.car_radius - 0.2),
        y=cy0 + ny * (p.car_radius - 0.2),
        theta=jnp.array(0.0),
        vx=-5.0 * nx,  # straight into the wall
        vy=-5.0 * ny,
    )
    st = st._replace(vehicle=veh)
    st2, _, rew, done, info = track_step(st, jnp.zeros(2, dtype=jnp.float32), p)
    assert bool(info["wall_hit"] > 0), "setup must contact the wall"
    assert bool(done), "wall touch must end the episode"
    assert float(rew) < 0.0, "crash must be penalized"
    d2, _, _ = cw(st2.vehicle.x, st2.vehicle.y, st2.walls)
    assert float(d2) >= p.car_radius - 0.05, float(d2)  # clean terminal state
    print("test_wall_touch_ends_episode OK")


def test_reverse_target_mapping():
    """Full back-stick commands a backward (negative-speed) target."""
    from car_rl.env.track_env import high_to_target

    tvx, tvy, des, off = high_to_target(jnp.array([0.0, -1.0]), jnp.array(0.0))
    assert float(des) < -7.9, float(des)          # ~-8 m/s reverse target
    assert float(tvx) < 0.0 and abs(float(tvy)) < 1e-5  # points behind
    tvx2, _, des2, _ = high_to_target(jnp.array([0.0, 1.0]), jnp.array(0.0))
    assert abs(float(des2) - 22.0) < 1e-5 and float(tvx2) > 0.0
    print("test_reverse_target_mapping OK")


def test_high_obs_and_wrapper():
    import torch
    from car_rl.env.track_wrapper import TorchBatchedTrackEnv

    env = TorchBatchedTrackEnv(num_envs=8, seed=0, device="cuda:0")
    obs = env.reset()
    assert obs.shape == (8, 12), obs.shape
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
    test_mirror_and_direction_variants()
    test_wall_touch_ends_episode()
    test_reverse_target_mapping()
    test_high_obs_and_wrapper()
    print("ALL TRACK TESTS PASSED")
