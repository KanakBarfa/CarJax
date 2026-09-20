#!/usr/bin/env python3
"""Grade high-level checkpoints + rank input dims by mean-ablation.

For each obs dim: replace it with its reference mean (neutral value, no
distribution shock) and measure the drop in track progress over fixed seeds.
Groups (all rays / all velocity) ablated too.
Metric: meters advanced along centerline (laps * length + s).
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")

import argparse
import numpy as np
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

DIM_NAMES = [
    "ray-90", "ray-54", "ray-18", "ray+18", "ray+54", "ray+90",
    "ray+150", "ray-150", "vx", "vy", "omega",
]
RAY_GROUP = list(range(8))
VEL_GROUP = [8, 9, 10]


def make_policies(high_ckpt, low_ckpt, device):
    high = ActorCritic(obs_dim=11, act_dim=2).to(device)
    if os.path.exists(high_ckpt):
        high.load_state_dict(torch.load(high_ckpt, map_location=device))
    high.eval()
    low = ActorCritic(obs_dim=8, act_dim=2).to(device)
    if os.path.exists(low_ckpt):
        low.load_state_dict(torch.load(low_ckpt, map_location=device))
    low.eval()
    return high, low


def rollout(high, low, params, jitted_step, seed, steps, ablate=None, ref_mean=None, device=None):
    """ablate: list of dim indices replaced by ref_mean. Returns (progress, walls, top_speed, obs_mean)."""
    state, _ = track_reset(jax.random.PRNGKey(seed), params)
    walls = 0
    top = 0.0
    obs_sum = np.zeros(11, dtype=np.float64)
    with torch.no_grad():
        for t in range(steps):
            ho = np.asarray(compute_high_obs(state, params), dtype=np.float32)
            obs_sum += ho
            if ablate is not None:
                ho = ho.copy()
                ho[ablate] = ref_mean[ablate]
            ha = torch.clamp(
                high.act_deterministic(torch.tensor(ho, device=device).unsqueeze(0)).squeeze(0), -1, 1
            ).cpu().numpy()
            tvx, tvy, _, _ = high_to_target(
                jnp.asarray(ha), state.vehicle.theta, params.max_desired_vel
            )
            lo = np.asarray(
                compute_low_obs(state.vehicle, tvx, tvy, params.v_scale,
                                params.omega_scale, params.vehicle.max_steer),
                dtype=np.float32,
            )
            la = torch.clamp(
                low.act_deterministic(torch.tensor(lo, device=device).unsqueeze(0)).squeeze(0), -1, 1
            ).cpu().numpy()
            state, _, _, _, info = jitted_step(state, jnp.asarray(la, dtype=jnp.float32))
            walls += int(float(info["wall_hit"]))
            top = max(top, float(info["speed"]))
    progress = float(state.lap) * float(state.track_length) + float(state.prev_s)
    return progress, walls, top, obs_sum / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--high", default="checkpoints/high_latest.pt")
    ap.add_argument("--other", default="checkpoints/high_best.pt")
    ap.add_argument("--low", default="checkpoints/best_model.pt")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    a = ap.parse_args()

    params = TrackParams()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    jitted_step = jax.jit(lambda s, la: track_step(s, la, params))

    for tag, ckpt in [("latest", a.high), ("best", a.other)]:
        high, low = make_policies(ckpt, a.low, device)
        print(f"--- {tag} ({ckpt}) baseline ---")
        for s in a.seeds:
            p, w, top, _ = rollout(high, low, params, jitted_step, s, a.steps, device=device)
            print(f"  seed {s}: progress {p:7.1f} m  walls {w:4d}  top {top*3.6:5.1f} km/h")

    # Ablation on --high only
    high, low = make_policies(a.high, a.low, device)
    base, ref = {}, {}
    for s in a.seeds:
        p, w, top, m = rollout(high, low, params, jitted_step, s, a.steps, device=device)
        base[s] = (p, w)
        ref[s] = m

    configs = [(f"no-{DIM_NAMES[i]}", [i]) for i in range(11)]
    configs += [("no-rays(all8)", RAY_GROUP), ("no-velocity(all3)", VEL_GROUP)]

    print("\n--- ablation (drop = baseline_progress - ablated_progress, avg over seeds) ---")
    rows = []
    for name, dims in configs:
        drops, wsum = [], 0
        for s in a.seeds:
            p, w, _, _ = rollout(high, low, params, jitted_step, s, a.steps,
                                 ablate=dims, ref_mean=ref[s], device=device)
            drops.append(base[s][0] - p)
            wsum += w
        rows.append((name, float(np.mean(drops)), wsum // len(a.seeds)))
    rows.sort(key=lambda r: -r[1])
    for rank, (name, drop, w) in enumerate(rows, 1):
        print(f"  {rank:2d}. {name:18s} progress-drop {drop:+8.1f} m   walls/run {w:4d}")


if __name__ == "__main__":
    main()
