#!/usr/bin/env python3
"""Headless eval: high-level lidar policy + frozen driver on one random track.

Saves a track map + telemetry plot and prints lap summary.
Works with an untrained high-level net (random weights) too.
"""

import os
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import jax
import jax.numpy as jnp

from car_rl.env.track_env import (
    TrackParams,
    track_reset,
    track_step,
    compute_high_obs,
    compute_low_obs,
    high_to_target,
)
from car_rl.models.actor_critic import ActorCritic


def run_eval(high_ckpt="checkpoints/high_best.pt", low_ckpt="checkpoints/best_model.pt",
             out="track_eval.png", steps=1500, seed=0):
    params = TrackParams()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    high = ActorCritic(obs_dim=12, act_dim=2).to(device)
    if os.path.exists(high_ckpt):
        high.load_state_dict(torch.load(high_ckpt, map_location=device))
        print(f"Loaded high-level policy from {high_ckpt}")
    else:
        print(f"No {high_ckpt}, evaluating with random high-level weights.")
    high.eval()

    low = ActorCritic(obs_dim=8, act_dim=2).to(device)
    if os.path.exists(low_ckpt):
        low.load_state_dict(torch.load(low_ckpt, map_location=device))
    low.eval()

    jitted_step = jax.jit(lambda s, a: track_step(s, a, params))
    rng = jax.random.PRNGKey(seed)
    rng, k = jax.random.split(rng)
    state, _ = track_reset(k, params)
    center = np.asarray(state.center)
    walls = np.asarray(state.walls)
    track_len = float(state.track_length)

    xs, ys, speeds, des_speeds, min_rays, laps, wall = [], [], [], [], [], [], []
    crashes = 0
    with torch.no_grad():
        for _ in range(steps):
            ho = compute_high_obs(state, params)
            ho_t = torch.from_dlpack(ho).unsqueeze(0).to(device)
            ha_t = torch.clamp(high.act_deterministic(ho_t).squeeze(0), -1, 1)
            ha = jax.dlpack.from_dlpack(ha_t)
            tvx, tvy, des, _ = high_to_target(ha, state.vehicle.theta, params.max_desired_vel)
            lo = compute_low_obs(state.vehicle, tvx, tvy, params.v_scale,
                                 params.omega_scale, params.vehicle.max_steer)
            lo_t = torch.from_dlpack(lo).unsqueeze(0).to(device)
            la_t = torch.clamp(low.act_deterministic(lo_t).squeeze(0), -1, 1)
            la = jax.dlpack.from_dlpack(la_t)
            state, _, rew, done, info = jitted_step(state, la)
            xs.append(float(state.vehicle.x))
            ys.append(float(state.vehicle.y))
            speeds.append(float(jnp.sqrt(state.vehicle.vx**2 + state.vehicle.vy**2)))
            des_speeds.append(float(des))
            min_rays.append(float(jnp.min(ho[:9])) * params.max_range)
            laps.append(int(info["lap"]))
            wall.append(float(info["wall_hit"]))
            if bool(done):
                crashes += int(float(info["wall_hit"]))
                rng, k = jax.random.split(rng)
                state, _ = track_reset(k, params)

    xs, ys = np.array(xs), np.array(ys)
    print(f"Track length: {track_len:.0f} m | laps: {laps[-1]} | "
          f"crashes: {crashes} | top speed: {max(speeds)*3.6:.0f} km/h")

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(center[:, 0], center[:, 1], "k--", lw=1, alpha=0.4, label="centerline")
    ax.plot(walls[:256, 0], walls[:256, 1], "r-", lw=1.5, label="railings")
    ax.plot(walls[:256, 2], walls[:256, 3], "r-", lw=1.5)
    ax.plot(walls[256:, 0], walls[256:, 1], "r-", lw=1.5)
    ax.plot(walls[256:, 2], walls[256:, 3], "r-", lw=1.5)
    ax.plot(xs, ys, "b-", lw=1.2, label="car")
    ax.plot(xs[0], ys[0], "go", ms=8, label="start")
    ax.set_aspect("equal")
    ax.set_title(f"Random track ({track_len:.0f} m) — laps: {laps[-1]}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    t = np.arange(steps) * 0.02
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, np.array(speeds) * 3.6, "b-", label="speed")
    ax2.plot(t, np.array(des_speeds) * 3.6, "r--", label="commanded")
    ax2.set_title("Speed vs commanded (km/h)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t, min_rays, "g-", label="min lidar (m)")
    ax3.set_title("Closest railing + wall contacts")
    ax3.set_xlabel("time (s)")
    ax3b = ax3.twinx()
    ax3b.plot(t, np.cumsum(wall), "r-", alpha=0.6, label="wall hits (cum)")
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--high", type=str, default="checkpoints/high_best.pt")
    ap.add_argument("--low", type=str, default="checkpoints/best_model.pt")
    ap.add_argument("--output", type=str, default="track_eval.png")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run_eval(a.high, a.low, a.output, a.steps, a.seed)
