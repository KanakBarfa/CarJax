#!/usr/bin/env python3
"""
Interactive Real-Time UI for 2D Car Simulation & RL Agent.

Features:
- Simple Click-to-Move Target Vector:
  * Left Click anywhere on screen to place the target marker!
  * The vector from car to marker defines the target velocity vector [vx_tgt, vy_tgt].
  * Direction = Heading, Distance = Speed.
  * The car continuously tracks this target velocity vector.
  * Click again anytime to move the marker.
  * [Right Click] or [SPACE] : Stop (set target to 0).
  * [R]                      : Reset vehicle to origin.
  * Scroll Wheel             : Adjust speed sensitivity scale.
  * [ESC]                    : Exit.
- Zero Lag: Bounded skid marks history (solid 60 FPS).
- Clean Telemetry HUD: Speedometer, Target Speed, Drift Meter, Steer, and Torque.
"""

import os
import sys
import math
import collections
import pygame
import torch
import jax
import jax.numpy as jnp

from car_rl.env.physics import VehicleParams, VehicleState, init_vehicle_state, step_physics
from car_rl.env.car_env import EnvParams, EnvState, compute_observation
from car_rl.models.actor_critic import ActorCritic


# UI Configuration
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
PIXELS_PER_METER = 22.0  # Zoom level
CAR_LENGTH_M = 4.2       # Visual body length (along longitudinal X)
CAR_WIDTH_M = 1.8        # Visual body width (along lateral Y)
WHEEL_LENGTH_M = 0.85    # Wheel rolling length (along wheel direction)
WHEEL_WIDTH_M = 0.35     # Wheel tire thickness

# Colors (Dark Motorsport Theme)
BG_COLOR = (24, 26, 32)
GRID_COLOR = (36, 40, 50)
CAR_BODY_COLOR = (235, 75, 50)     # Sporty Red
CAR_ROOF_COLOR = (40, 44, 54)
CAR_GLASS_COLOR = (100, 160, 220)
WHEEL_COLOR = (15, 15, 18)
TARGET_COLOR = (0, 230, 130)       # Bright Neon Green
TARGET_LINE = (0, 180, 110)
ACTUAL_COLOR = (70, 160, 255)      # Cyan Blue
DRIFT_COLOR = (255, 180, 0)        # Amber / Orange
TEXT_COLOR = (230, 235, 245)
TEXT_MUTED = (160, 170, 190)
HUD_BG = (18, 20, 26, 215)


def world_to_screen(wx: float, wy: float, cam_x: float, cam_y: float) -> tuple:
    """Converts 2D physics world coordinates (meters) to Pygame screen coordinates (pixels)."""
    sx = (wx - cam_x) * PIXELS_PER_METER + WINDOW_WIDTH / 2.0
    sy = WINDOW_HEIGHT / 2.0 - (wy - cam_y) * PIXELS_PER_METER
    return int(sx), int(sy)


def rotate_point(x: float, y: float, angle_rad: float) -> tuple:
    """Rotates a 2D point around (0,0) by angle in radians."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


def draw_oriented_box(surface, color, center_wx, center_wy, length_m, width_m, angle_rad, cam_x, cam_y):
    """
    Draws a rectangle oriented at angle_rad.
    length_m is along the forward heading axis (angle_rad).
    width_m is along the lateral axis (angle_rad + 90 deg).
    """
    hl = length_m / 2.0
    hw = width_m / 2.0
    corners = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    screen_pts = []
    for lx, ly in corners:
        rx, ry = rotate_point(lx, ly, angle_rad)
        sx, sy = world_to_screen(center_wx + rx, center_wy + ry, cam_x, cam_y)
        screen_pts.append((sx, sy))
    pygame.draw.polygon(surface, color, screen_pts)


def draw_vector_arrow(surface, start_pt, end_pt, color, width=3, head_size=12):
    """Draws a directional arrow with arrowhead from start_pt to end_pt."""
    pygame.draw.line(surface, color, start_pt, end_pt, width)
    dx = end_pt[0] - start_pt[0]
    dy = end_pt[1] - start_pt[1]
    length = math.hypot(dx, dy)
    if length < 5:
        return
    angle = math.atan2(dy, dx)
    p1 = (
        int(end_pt[0] - head_size * math.cos(angle - math.pi / 6)),
        int(end_pt[1] - head_size * math.sin(angle - math.pi / 6)),
    )
    p2 = (
        int(end_pt[0] - head_size * math.cos(angle + math.pi / 6)),
        int(end_pt[1] - head_size * math.sin(angle + math.pi / 6)),
    )
    pygame.draw.polygon(surface, color, [end_pt, p1, p2])


def main():
    pygame.init()
    pygame.font.init()
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.DOUBLEBUF)
    pygame.display.set_caption("2D GPU Car Simulation - Click Target Velocity")
    clock = pygame.time.Clock()
    
    font_large = pygame.font.SysFont("DejaVu Sans", 24, bold=True)
    font_med = pygame.font.SysFont("DejaVu Sans", 18, bold=True)
    font_small = pygame.font.SysFont("DejaVu Sans", 14)
    
    # Physics & Environment Parameters
    env_params = EnvParams()
    params = env_params.vehicle
    
    # Initialize PyTorch Agent with 9D observation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = ActorCritic(obs_dim=9, act_dim=2).to(device)
    checkpoint_path = "checkpoints/best_model.pt"
    if os.path.exists(checkpoint_path):
        agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded trained policy from {checkpoint_path}")
    agent.eval()
    
    # Simulation State
    state = init_vehicle_state(x=0.0, y=0.0, theta=0.0, vx=0.0)
    cam_x, cam_y = 0.0, 0.0
    
    # Clicked Target Velocity Vector (Offset from car in screen pixels)
    target_dx = 0.0
    target_dy = 0.0
    desired_speed = 0.0
    target_vx = 0.0
    target_vy = 0.0
    speed_sensitivity = 2.0  # Speed scaling factor
    
    # Skid marks history: capped to 150 items to eliminate render latency
    skid_marks = collections.deque(maxlen=150)
    prev_left_tire = None
    prev_right_tire = None
    
    # JIT-compile physics step
    physics_step_jit = jax.jit(lambda s, st, t: step_physics(s, st, t, params))
    
    print("\n=======================================================")
    print("   2D GPU Car Simulation - Click to Move Target Marker ")
    print("=======================================================")
    print(" CONTROLS:")
    print("   [Left Click]   : Click anywhere to place the target marker!")
    print("                    (Vector from car to marker = Target Velocity)")
    print("   [Right Click]  : Stop (set target velocity = 0)")
    print("   [SPACE]        : Stop (set target velocity = 0)")
    print("   Scroll Wheel   : Adjust speed sensitivity scale")
    print("   [R]            : Reset car position")
    print("   [ESC]          : Exit")
    print("=======================================================\n")
    
    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # 60 FPS
        
        car_sx, car_sy = world_to_screen(float(state.x), float(state.y), cam_x, cam_y)
        
        # 1. Event Handling (Mouse Clicks & Keys)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Left Click: Place target marker relative to car
                    mx, my = event.pos
                    dx = mx - car_sx
                    dy = my - car_sy
                    dist_px = math.hypot(dx, dy)
                    
                    if dist_px < 15:
                        # Clicked directly on the car: stop
                        target_dx = 0.0
                        target_dy = 0.0
                        desired_speed = 0.0
                        target_vx = 0.0
                        target_vy = 0.0
                    else:
                        target_dx = dx
                        target_dy = dy
                        dist_m = dist_px / PIXELS_PER_METER
                        desired_speed = min(env_params.max_desired_vel, dist_m * speed_sensitivity)
                        ux = target_dx / dist_px
                        uy = -target_dy / dist_px  # Inverted because screen Y is down
                        target_vx = ux * desired_speed
                        target_vy = uy * desired_speed
                elif event.button == 3:
                    # Right Click: Stop
                    target_dx = 0.0
                    target_dy = 0.0
                    desired_speed = 0.0
                    target_vx = 0.0
                    target_vy = 0.0
            elif event.type == pygame.MOUSEWHEEL:
                speed_sensitivity = max(0.5, min(4.0, speed_sensitivity + event.y * 0.2))
                # Re-scale current target speed if active
                dist_px = math.hypot(target_dx, target_dy)
                if dist_px > 10:
                    dist_m = dist_px / PIXELS_PER_METER
                    desired_speed = min(env_params.max_desired_vel, dist_m * speed_sensitivity)
                    ux = target_dx / dist_px
                    uy = -target_dy / dist_px
                    target_vx = ux * desired_speed
                    target_vy = uy * desired_speed
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    target_dx = 0.0
                    target_dy = 0.0
                    desired_speed = 0.0
                    target_vx = 0.0
                    target_vy = 0.0
                elif event.key == pygame.K_r:
                    state = init_vehicle_state()
                    target_dx = 0.0
                    target_dy = 0.0
                    desired_speed = 0.0
                    target_vx = 0.0
                    target_vy = 0.0
                    skid_marks.clear()
                    
        # 2. Ego-Centric Observation for RL Agent
        car_th = float(state.theta)
        target_speed = math.hypot(target_vx, target_vy)
        has_target = target_speed > 0.1
        
        target_angle = math.atan2(target_vy, target_vx)
        heading_err = (target_angle - car_th + math.pi) % (2.0 * math.pi) - math.pi
        cos_target_ego = math.cos(heading_err) if has_target else 1.0
        sin_target_ego = math.sin(heading_err) if has_target else 0.0
        
        ex = target_speed - float(state.vx)
        ey = -float(state.vy)
        
        obs = torch.tensor([[
            ex / env_params.v_norm_scale,
            ey / env_params.v_norm_scale,
            desired_speed / env_params.v_norm_scale,
            cos_target_ego,
            sin_target_ego,
            float(state.vx) / env_params.v_norm_scale,
            float(state.vy) / env_params.v_norm_scale,
            float(state.steer) / params.max_steer,
            float(state.omega) / env_params.omega_norm_scale,
        ]], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            act = agent.act_deterministic(obs).cpu().squeeze(0)
            steer_act = float(act[0])
            torque_act = float(act[1])
            
        # 3. Physics Step
        state = physics_step_jit(
            state,
            jnp.array(steer_act, dtype=jnp.float32),
            jnp.array(torque_act, dtype=jnp.float32),
        )
        
        # Smooth camera tracking
        cam_x += (float(state.x) - cam_x) * 0.35
        cam_y += (float(state.y) - cam_y) * 0.35
        
        # 4. Skid Marks
        car_th = float(state.theta)
        cos_th = math.cos(car_th)
        sin_th = math.sin(car_th)
        lr = params.lr
        track_w = 0.8
        
        rl_x = float(state.x) - lr * cos_th + track_w * (-sin_th)
        rl_y = float(state.y) - lr * sin_th + track_w * cos_th
        rr_x = float(state.x) - lr * cos_th - track_w * (-sin_th)
        rr_y = float(state.y) - lr * sin_th - track_w * cos_th
        
        lateral_slip = abs(float(state.vy))
        is_sliding = lateral_slip > 0.4
        
        if is_sliding and prev_left_tire is not None:
            skid_marks.append((prev_left_tire[0], prev_left_tire[1], rl_x, rl_y))
            skid_marks.append((prev_right_tire[0], prev_right_tire[1], rr_x, rr_y))
            
        prev_left_tire = (rl_x, rl_y)
        prev_right_tire = (rr_x, rr_y)
        
        # ================= RENDERING =================
        screen.fill(BG_COLOR)
        
        # A. Background Grid (10m spacing)
        grid_spacing_m = 10.0
        start_gx = math.floor((cam_x - WINDOW_WIDTH / (2 * PIXELS_PER_METER)) / grid_spacing_m) * grid_spacing_m
        end_gx = math.ceil((cam_x + WINDOW_WIDTH / (2 * PIXELS_PER_METER)) / grid_spacing_m) * grid_spacing_m
        start_gy = math.floor((cam_y - WINDOW_HEIGHT / (2 * PIXELS_PER_METER)) / grid_spacing_m) * grid_spacing_m
        end_gy = math.ceil((cam_y + WINDOW_HEIGHT / (2 * PIXELS_PER_METER)) / grid_spacing_m) * grid_spacing_m
        
        gx = start_gx
        while gx <= end_gx:
            sx, _ = world_to_screen(gx, 0, cam_x, cam_y)
            if 0 <= sx <= WINDOW_WIDTH:
                pygame.draw.line(screen, GRID_COLOR, (sx, 0), (sx, WINDOW_HEIGHT), 1)
            gx += grid_spacing_m
            
        gy = start_gy
        while gy <= end_gy:
            _, sy = world_to_screen(0, gy, cam_x, cam_y)
            if 0 <= sy <= WINDOW_HEIGHT:
                pygame.draw.line(screen, GRID_COLOR, (0, sy), (WINDOW_WIDTH, sy), 1)
            gy += grid_spacing_m
            
        # B. Skid Marks
        for x1, y1, x2, y2 in skid_marks:
            p1 = world_to_screen(x1, y1, cam_x, cam_y)
            p2 = world_to_screen(x2, y2, cam_x, cam_y)
            pygame.draw.line(screen, (34, 36, 44), p1, p2, 4)
            
        car_sx, car_sy = world_to_screen(float(state.x), float(state.y), cam_x, cam_y)
        
        # C. Target Velocity Vector & Placed Marker
        if has_target:
            marker_px = int(car_sx + target_dx)
            marker_py = int(car_sy + target_dy)
            
            # Line from car to placed marker
            pygame.draw.line(screen, TARGET_LINE, (car_sx, car_sy), (marker_px, marker_py), 2)
            
            # Marker circle
            pygame.draw.circle(screen, TARGET_COLOR, (marker_px, marker_py), 8, 2)
            pygame.draw.circle(screen, TARGET_COLOR, (marker_px, marker_py), 3)
            
            # Floating target speed badge
            target_badge = font_small.render(f"🎯 {desired_speed * 3.6:3.0f} km/h", True, TARGET_COLOR)
            screen.blit(target_badge, (marker_px + 14, marker_py - 10))
            
        # Car Actual Velocity Vector (Cyan Blue arrow)
        current_speed = math.hypot(float(state.vx), float(state.vy))
        current_speed_kmh = current_speed * 3.6
        if current_speed > 0.3:
            vel_len_px = current_speed * PIXELS_PER_METER * 0.8
            world_vx = float(state.vx) * cos_th - float(state.vy) * sin_th
            world_vy = float(state.vx) * sin_th + float(state.vy) * cos_th
            vel_tip_sx = int(car_sx + vel_len_px * (world_vx / current_speed))
            vel_tip_sy = int(car_sy - vel_len_px * (world_vy / current_speed))
            draw_vector_arrow(screen, (car_sx, car_sy), (vel_tip_sx, vel_tip_sy), ACTUAL_COLOR, width=2, head_size=10)
            
        # D. Wheels (Front steering & Rear fixed)
        lf = params.lf
        steer_angle = float(state.steer)
        
        for side in [track_w, -track_w]:
            wx = float(state.x) + lf * cos_th + side * (-sin_th)
            wy = float(state.y) + lf * sin_th + side * cos_th
            draw_oriented_box(
                screen, WHEEL_COLOR, wx, wy,
                WHEEL_LENGTH_M, WHEEL_WIDTH_M, car_th + steer_angle, cam_x, cam_y
            )
            
        for side in [track_w, -track_w]:
            wx = float(state.x) - lr * cos_th + side * (-sin_th)
            wy = float(state.y) - lr * sin_th + side * cos_th
            draw_oriented_box(
                screen, WHEEL_COLOR, wx, wy,
                WHEEL_LENGTH_M, WHEEL_WIDTH_M, car_th, cam_x, cam_y
            )
            
        # E. Car Body
        draw_oriented_box(
            screen, CAR_BODY_COLOR, float(state.x), float(state.y),
            CAR_LENGTH_M, CAR_WIDTH_M, car_th, cam_x, cam_y
        )
        
        # Cabin / Roof
        roof_offset = -0.3
        draw_oriented_box(
            screen, CAR_ROOF_COLOR,
            float(state.x) + roof_offset * cos_th, float(state.y) + roof_offset * sin_th,
            CAR_LENGTH_M * 0.45, CAR_WIDTH_M * 0.75, car_th, cam_x, cam_y
        )
        
        # Windshield
        windshield_offset = 0.6
        draw_oriented_box(
            screen, CAR_GLASS_COLOR,
            float(state.x) + windshield_offset * cos_th, float(state.y) + windshield_offset * sin_th,
            0.5, CAR_WIDTH_M * 0.68, car_th, cam_x, cam_y
        )
        
        # Headlights & Taillights
        hl_dist = CAR_LENGTH_M * 0.46
        hl_w = CAR_WIDTH_M * 0.35
        for side in [hl_w, -hl_w]:
            hx = float(state.x) + hl_dist * cos_th + side * (-sin_th)
            hy = float(state.y) + hl_dist * sin_th + side * cos_th
            hsx, hsy = world_to_screen(hx, hy, cam_x, cam_y)
            pygame.draw.circle(screen, (255, 255, 210), (hsx, hsy), 4)
            
        tl_dist = -CAR_LENGTH_M * 0.48
        for side in [hl_w, -hl_w]:
            tx = float(state.x) + tl_dist * cos_th + side * (-sin_th)
            ty = float(state.y) + tl_dist * sin_th + side * cos_th
            tsx, tsy = world_to_screen(tx, ty, cam_x, cam_y)
            pygame.draw.circle(screen, (255, 40, 40), (tsx, tsy), 4)
            
        # F. Modern HUD & Telemetry
        desired_speed_kmh = desired_speed * 3.6
        
        hud_rect = pygame.Rect(20, 20, 360, 240)
        hud_surf = pygame.Surface((hud_rect.width, hud_rect.height), pygame.SRCALPHA)
        hud_surf.fill(HUD_BG)
        pygame.draw.rect(hud_surf, (80, 90, 110), hud_surf.get_rect(), 1, border_radius=8)
        screen.blit(hud_surf, hud_rect.topleft)
        
        # Mode Badge
        screen.blit(font_med.render("MODE: AI DRIVER (Click to Place)", True, TARGET_COLOR), (35, 32))
        
        # Speedometer
        speed_str = f"SPEED: {current_speed_kmh:5.1f} km/h"
        target_str = f" (TARGET: {desired_speed_kmh:4.0f})" if has_target else " (STOPPED)"
        screen.blit(font_med.render(speed_str + target_str, True, TEXT_COLOR), (35, 62))
        
        # Speed bar
        bar_w = 320
        pygame.draw.rect(screen, (40, 44, 54), (35, 90, bar_w, 10), border_radius=4)
        speed_fill = min(1.0, current_speed_kmh / 100.0) * bar_w
        pygame.draw.rect(screen, (70, 160, 240), (35, 90, int(speed_fill), 10), border_radius=4)
        if has_target and desired_speed_kmh > 0:
            target_mark_x = int(35 + min(1.0, desired_speed_kmh / 100.0) * bar_w)
            pygame.draw.line(screen, TARGET_COLOR, (target_mark_x, 87), (target_mark_x, 103), 3)
            
        # Drift Meter
        slip_text = f"LATERAL SLIP: {lateral_slip:4.2f} m/s"
        screen.blit(font_small.render(slip_text, True, TEXT_COLOR), (35, 112))
        if is_sliding:
            drift_badge = font_med.render("★ DRIFTING ★", True, (255, 60, 60))
            screen.blit(drift_badge, (215, 109))
            
        steer_deg = math.degrees(steer_angle)
        screen.blit(font_small.render(f"STEERING: {steer_deg:5.1f}°", True, TEXT_COLOR), (35, 138))
        torque_col = (100, 255, 140) if torque_act > 0 else (255, 110, 110) if torque_act < 0 else TEXT_COLOR
        actual_nm = torque_act * params.max_torque if torque_act >= 0 else torque_act * abs(params.min_torque)
        screen.blit(font_small.render(f"TORQUE: {actual_nm:+5.0f} Nm", True, torque_col), (190, 138))
        
        # Quick Help
        pygame.draw.line(screen, (50, 56, 70), (35, 168), (35 + bar_w, 168), 1)
        screen.blit(font_small.render("Left Click : Place target marker (car -> marker = vel)", True, (140, 220, 180)), (35, 178))
        screen.blit(font_small.render("Right Click / SPACE: Stop | Scroll: Speed scale", True, TEXT_MUTED), (35, 198))
        screen.blit(font_small.render("[R]: Reset Car | [ESC]: Exit", True, TEXT_MUTED), (35, 218))
        
        pygame.display.flip()
        
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
