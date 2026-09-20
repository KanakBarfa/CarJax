#!/usr/bin/env python3
"""Playable track game: random railed circuits + lidar AI vs manual drive.

Controls:
  [A]      : toggle AI driver / manual mouse-target
  AI mode  : high-level lidar net commands frozen low-level driver (hands off)
  Manual   : left-click sets target (car -> mouse = heading + speed), SPACE stops
  [R]      : new random track
  [ESC]    : exit
"""

import math
import os
import sys
import argparse
import collections

import pygame
import torch
import jax
import jax.numpy as jnp
import numpy as np

from car_rl.env.track_env import (
    TrackParams,
    track_reset,
    track_step,
    compute_high_obs,
    compute_low_obs,
    high_to_target,
)
from car_rl.env.track_raycast import RAY_ANGLES, MAX_RANGE
from car_rl.models.actor_critic import ActorCritic

W, H = 1280, 720
PPM = 6.0  # zoomed out: tracks are ~150 m across

BG = (24, 26, 32)
ROAD = (52, 56, 66)
RAIL = (220, 60, 60)
CENTER_LINE = (120, 130, 150)
CAR_BODY = (235, 75, 50)
RAY_OK = (0, 230, 130)
RAY_CLOSE = (255, 60, 60)
TEXT = (230, 235, 245)
MUTED = (160, 170, 190)


def world_to_screen(wx, wy, cx, cy):
    return int((wx - cx) * PPM + W / 2), int(H / 2 - (wy - cy) * PPM)


def draw_poly(surface, pts_world, cx, cy, color, width=0):
    pts = [world_to_screen(x, y, cx, cy) for x, y in pts_world]
    if width == 0:
        if len(pts) > 2:
            pygame.draw.polygon(surface, color, pts)
    else:
        pygame.draw.lines(surface, color, True, pts, width)


def main():
    ap = argparse.ArgumentParser(description="Track Game - Lidar AI vs You")
    ap.add_argument("--seed", type=int, default=None,
                    help="Track RNG seed (default: random every launch)")
    args = ap.parse_args()
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    print(f"Track seed: {seed} ([R] reshuffles in-game)")
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((W, H), pygame.DOUBLEBUF)
    pygame.display.set_caption("Track Game - Lidar AI vs You")
    clock = pygame.time.Clock()
    f_med = pygame.font.SysFont("DejaVu Sans", 18, bold=True)
    f_small = pygame.font.SysFont("DejaVu Sans", 14)

    params = TrackParams()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    high = ActorCritic(obs_dim=12, act_dim=2).to(device)
    if os.path.exists("checkpoints/high_best.pt"):
        high.load_state_dict(torch.load("checkpoints/high_best.pt", map_location=device))
        print("Loaded trained high-level policy.")
    else:
        print("No high_best.pt — AI drives with random weights (train with train_high.py).")
    high.eval()
    low = ActorCritic(obs_dim=8, act_dim=2).to(device)
    if os.path.exists("checkpoints/best_model.pt"):
        low.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    low.eval()

    jitted_step = jax.jit(lambda s, a: track_step(s, a, params))
    rng = jax.random.PRNGKey(seed)
    rng, k = jax.random.split(rng)
    state, _ = track_reset(k, params)

    ai_mode = True
    man_tvx = man_tvy = 0.0
    man_speed = 0.0
    cam_x, cam_y = float(state.vehicle.x), float(state.vehicle.y)
    wall_flash = 0

    print("Controls: [A] AI/manual | click = target | SPACE = stop | [R] new track | [ESC] exit")
    running = True
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_a:
                    ai_mode = not ai_mode
                    print("Mode:", "AI" if ai_mode else "MANUAL")
                elif event.key == pygame.K_r:
                    rng, k = jax.random.split(rng)
                    state, _ = track_reset(k, params)
                elif event.key == pygame.K_SPACE:
                    man_tvx = man_tvy = man_speed = 0.0
            elif event.type == pygame.MOUSEBUTTONDOWN and not ai_mode:
                if event.button == 1:
                    mx, my = event.pos
                    dx = (mx - W / 2) / PPM
                    dy = -(my - H / 2) / PPM
                    dist = math.hypot(dx, dy)
                    if dist < 1.0:
                        man_tvx = man_tvy = man_speed = 0.0
                    else:
                        man_speed = min(params.max_desired_vel, dist * 2.0)
                        man_tvx = dx / dist * man_speed
                        man_tvy = dy / dist * man_speed

        # --- policy evaluation (numpy bridge, 1 env, UI rate) ---
        ho = np.asarray(compute_high_obs(state, params), dtype=np.float32)
        with torch.no_grad():
            if ai_mode:
                ha = torch.clamp(high.act_deterministic(
                    torch.tensor(ho, device=device).unsqueeze(0)).squeeze(0), -1, 1)
                ha_np = ha.cpu().numpy()
                tvx, tvy, des, off = high_to_target(
                    jnp.asarray(ha_np), state.vehicle.theta, params.max_desired_vel)
                tvx, tvy, des, off = float(tvx), float(tvy), float(des), float(off)
            else:
                tvx, tvy, des = man_tvx, man_tvy, man_speed
                off = 0.0
            lo = np.asarray(compute_low_obs(
                state.vehicle, jnp.asarray(tvx), jnp.asarray(tvy),
                params.v_scale, params.omega_scale, params.vehicle.max_steer), dtype=np.float32)
            la = torch.clamp(low.act_deterministic(
                torch.tensor(lo, device=device).unsqueeze(0)).squeeze(0), -1, 1)
            la_np = la.cpu().numpy()

        state, _, _, done, info = jitted_step(state, jnp.asarray(la_np, dtype=jnp.float32))
        if bool(done):
            # Crash or timeout: fresh random track, keep playing
            rng, k = jax.random.split(rng)
            state, _ = track_reset(k, params)
            man_tvx = man_tvy = man_speed = 0.0
        wall_flash = 12 if float(info["wall_hit"]) > 0 else max(0, wall_flash - 1)

        cam_x += (float(state.vehicle.x) - cam_x) * 0.25
        cam_y += (float(state.vehicle.y) - cam_y) * 0.25

        # --- render ---
        screen.fill(BG)
        walls_np = np.asarray(state.walls)
        # walls rows [0:256] = left railing segs, [256:512] = right segs
        left_pts = walls_np[:256:2, 0:2]
        right_pts = walls_np[256::2, 0:2]
        center = np.asarray(state.center)[::2]
        road_poly = np.concatenate([left_pts, right_pts[::-1]], axis=0)
        draw_poly(screen, road_poly, cam_x, cam_y, ROAD)
        draw_poly(screen, left_pts, cam_x, cam_y, RAIL, 2)
        draw_poly(screen, right_pts, cam_x, cam_y, RAIL, 2)
        # start line
        l0, r0 = walls_np[0, 0:2], walls_np[256, 0:2]
        pygame.draw.line(screen, (0, 230, 130),
                         world_to_screen(*l0, cam_x, cam_y),
                         world_to_screen(*r0, cam_x, cam_y), 3)

        cx, cy = float(state.vehicle.x), float(state.vehicle.y)
        th = float(state.vehicle.theta)
        car_sx, car_sy = world_to_screen(cx, cy, cam_x, cam_y)

        # rays (9: 7 front incl. dead-ahead, 2 back)
        ray_dists = ho[:9] * MAX_RANGE
        for ang, d in zip(np.asarray(RAY_ANGLES), ray_dists):
            ex = cx + math.cos(th + float(ang)) * float(d)
            ey = cy + math.sin(th + float(ang)) * float(d)
            col = RAY_CLOSE if float(d) < 5.0 else RAY_OK
            pygame.draw.line(screen, col, (car_sx, car_sy),
                             world_to_screen(ex, ey, cam_x, cam_y), 1)

        # commanded heading arrow
        if des > 0.5:
            cmd_ang = th + off if ai_mode else math.atan2(man_tvy, man_tvx)
            tip = world_to_screen(cx + math.cos(cmd_ang) * des * 0.8,
                                  cy + math.sin(cmd_ang) * des * 0.8, cam_x, cam_y)
            pygame.draw.line(screen, (70, 160, 255), (car_sx, car_sy), tip, 2)

        # car body (rotated rect)
        L, WID = 4.2, 1.8
        cs, sn = math.cos(th), math.sin(th)
        corners = []
        for lx, ly in [(L/2, WID/2), (L/2, -WID/2), (-L/2, -WID/2), (-L/2, WID/2)]:
            corners.append(world_to_screen(cx + lx*cs - ly*sn, cy + lx*sn + ly*cs, cam_x, cam_y))
        pygame.draw.polygon(screen, CAR_BODY, corners)

        # HUD
        speed = float(info["speed"]) * 3.6
        prog = 100.0 * float(info["progress_s"]) / max(float(info["track_length"]), 1e-3)
        lines = [
            f"MODE: {'AI LIDAR' if ai_mode else 'MANUAL (click)'}   [A] toggle",
            f"SPEED {speed:5.1f} km/h   CMD {des*3.6:4.0f}  OFF {math.degrees(off):+5.0f} deg",
            f"LAP {int(info['lap'])}   PROGRESS {prog:5.1f}%   LEN {float(info['track_length']):.0f} m",
            f"STEER {math.degrees(float(state.vehicle.steer)):5.1f} deg   TORQUE {float(la_np[1]):+.2f}"
            + ("   WALL HIT!" if wall_flash else ""),
            "[R] new track | SPACE stop (manual) | [ESC] exit",
        ]
        y = 16
        for i, s in enumerate(lines):
            col = (255, 90, 90) if (i == 3 and wall_flash) else (TEXT if i < 4 else MUTED)
            screen.blit((f_med if i == 0 else f_small).render(s, True, col), (16, y))
            y += 26 if i == 0 else 22

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
