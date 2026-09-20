#!/usr/bin/env python3
"""
Interactive Real-Time UI for 2D Car Simulation & RL Agent.

Features:
- Dual Modes:
  * AI Driver Mode: Car uses trained PPO policy to track mouse cursor velocity vector [vx_tgt, vy_tgt]!
    (Magnitude = speed, Direction = heading -> only 2 inputs!)
  * Manual Drift Mode: Drive the dynamic vehicle directly with WASD / Arrow Keys + Spacebar Handbrake!
- Corrected 0-degree vehicle & wheel orientation (length along forward velocity vector).
- Realistic front wheel steering visualization.
- Visual tire skid marks during drifts (|vy| > 0.4 m/s).
- Full real-time telemetry HUD: Speedometer, Drift Meter, Steering Angle, and Torque.
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
TARGET_COLOR = (0, 220, 140)       # Neon Green
DRIFT_COLOR = (255, 180, 0)        # Amber / Orange
TEXT_COLOR = (230, 235, 245)
HUD_BG = (18, 20, 26, 210)


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
    # 4 corners in local vehicle frame: Front-Left, Front-Right, Rear-Right, Rear-Left
    corners = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    screen_pts = []
    for lx, ly in corners:
        rx, ry = rotate_point(lx, ly, angle_rad)
        sx, sy = world_to_screen(center_wx + rx, center_wy + ry, cam_x, cam_y)
        screen_pts.append((sx, sy))
    pygame.draw.polygon(surface, color, screen_pts)


def main():
    pygame.init()
    pygame.font.init()
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.DOUBLEBUF)
    pygame.display.set_caption("2D GPU Car Simulation - Interactive AI & Drift UI")
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
    
    # Modes & Commands
    is_ai_mode = True
    speed_sensitivity = 1.0  # (m/s) per meter of distance
    desired_speed = 0.0      # Dynamically calculated from distance to cursor
    target_ux = 1.0
    target_uy = 0.0
    
    # Skid marks history in world coordinates (wx1, wy1, wx2, wy2, alpha)
    skid_marks = collections.deque(maxlen=1500)
    prev_left_tire = None
    prev_right_tire = None
    
    # JIT-compile physics step for ultra-low latency
    physics_step_jit = jax.jit(lambda s, st, t: step_physics(s, st, t, params))
    
    running = True
    print("\n--- UI Controls ---")
    print(" [M]         : Toggle AI Driver / Manual Drive Mode")
    print(" Mouse Move  : Aim direction & distance (further = faster!)")
    print(" Scroll Wheel: Adjust speed sensitivity scale")
    print(" [W] / [S]   : Throttle / Brakes (Manual Mode)")
    print(" [A] / [D]   : Steer Left / Right (Manual Mode)")
    print(" [SPACE]     : Handbrake Drift (Manual Mode)")
    print(" [R]         : Reset Car Position")
    print(" [ESC]       : Exit\n")
    
    while running:
        dt = clock.tick(60) / 1000.0  # 60 FPS
        mouse_px, mouse_py = pygame.mouse.get_pos()
        
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_m:
                    is_ai_mode = not is_ai_mode
                elif event.key == pygame.K_r:
                    state = init_vehicle_state()
                    skid_marks.clear()
            elif event.type == pygame.MOUSEWHEEL:
                speed_sensitivity = max(0.2, min(3.0, speed_sensitivity + event.y * 0.1))
                
        # 2. Control Processing
        keys = pygame.key.get_pressed()
        
        # Compute mouse target in world coordinates
        mouse_wx = cam_x + (mouse_px - WINDOW_WIDTH / 2.0) / PIXELS_PER_METER
        mouse_wy = cam_y - (mouse_py - WINDOW_HEIGHT / 2.0) / PIXELS_PER_METER
        dx = mouse_wx - float(state.x)
        dy = mouse_wy - float(state.y)
        dist_to_mouse = math.hypot(dx, dy)
        
        if dist_to_mouse > 0.4:
            target_ux = dx / dist_to_mouse
            target_uy = dy / dist_to_mouse
            # Distance directly determines desired speed (magnitude of vector)!
            desired_speed = min(env_params.max_desired_vel, dist_to_mouse * speed_sensitivity)
        else:
            target_ux = 0.0
            target_uy = 0.0
            desired_speed = 0.0
            
        target_vx = target_ux * desired_speed
        target_vy = target_uy * desired_speed
        
        car_th = float(state.theta)
        cos_th = math.cos(car_th)
        sin_th = math.sin(car_th)
        
        if is_ai_mode:
            # 2-input target velocity in ego frame:
            ego_target_vx = target_vx * cos_th + target_vy * sin_th
            ego_target_vy = -target_vx * sin_th + target_vy * cos_th
            
            ex = ego_target_vx - float(state.vx)
            ey = ego_target_vy - float(state.vy)
            
            target_speed = math.hypot(target_vx, target_vy)
            has_target = target_speed > 0.1
            cos_target_ego = (ego_target_vx / target_speed) if has_target else 1.0
            sin_target_ego = (ego_target_vy / target_speed) if has_target else 0.0
            
            # 9-dimensional observation vector:
            obs = torch.tensor([
                ex / env_params.v_norm_scale,
                ey / env_params.v_norm_scale,
                desired_speed / env_params.v_norm_scale,
                cos_target_ego,
                sin_target_ego,
                float(state.vx) / env_params.v_norm_scale,
                float(state.vy) / env_params.v_norm_scale,
                float(state.steer) / params.max_steer,
                float(state.omega) / env_params.omega_norm_scale,
            ], dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                act = agent.act_deterministic(obs).cpu().squeeze(0)
                steer_act = float(act[0])
                torque_act = float(act[1])
        else:
            # Manual Driver
            steer_target = 0.0
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                steer_target += 1.0  # Turn left
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                steer_target -= 1.0  # Turn right
                
            torque_act = 0.0
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                torque_act += 1.0   # Gas
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                torque_act -= 1.0   # Brake
            if keys[pygame.K_SPACE]:
                torque_act = -1.0   # Handbrake
                
            steer_act = steer_target
            
        # 3. Physics Step
        state = physics_step_jit(
            state,
            jnp.array(steer_act, dtype=jnp.float32),
            jnp.array(torque_act, dtype=jnp.float32),
        )
        
        # Smooth camera tracking
        cam_x += (float(state.x) - cam_x) * 0.1
        cam_y += (float(state.y) - cam_y) * 0.1
        
        # 4. Skid Marks Calculation (Tire Tracks)
        lr = params.lr
        track_w = 0.8
        
        # Rear tire positions in world coordinates
        # Heading is (cos_th, sin_th); Lateral left is (-sin_th, cos_th)
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
            pygame.draw.line(screen, GRID_COLOR, (sx, 0), (sx, WINDOW_HEIGHT), 1)
            gx += grid_spacing_m
            
        gy = start_gy
        while gy <= end_gy:
            _, sy = world_to_screen(0, gy, cam_x, cam_y)
            pygame.draw.line(screen, GRID_COLOR, (0, sy), (WINDOW_WIDTH, sy), 1)
            gy += grid_spacing_m
            
        # B. Skid Marks
        for x1, y1, x2, y2 in skid_marks:
            p1 = world_to_screen(x1, y1, cam_x, cam_y)
            p2 = world_to_screen(x2, y2, cam_x, cam_y)
            pygame.draw.line(screen, (34, 36, 44), p1, p2, 4)
            
        # C. Target Velocity Vector (AI Mode)
        car_sx, car_sy = world_to_screen(float(state.x), float(state.y), cam_x, cam_y)
        if is_ai_mode:
            # Line from car to cursor representing the target velocity vector
            pygame.draw.line(screen, (0, 180, 110), (car_sx, car_sy), (mouse_px, mouse_py), 2)
            
            # Cursor marker
            pygame.draw.circle(screen, TARGET_COLOR, (mouse_px, mouse_py), 8, 2)
            pygame.draw.circle(screen, TARGET_COLOR, (mouse_px, mouse_py), 3)
            
            # Floating target speed label beside cursor
            target_badge = font_small.render(f"🎯 {desired_speed * 3.6:3.0f} km/h", True, TARGET_COLOR)
            screen.blit(target_badge, (mouse_px + 14, mouse_py - 10))
            
        # D. Wheels (Aligned with car heading + steering!)
        lf = params.lf
        steer_angle = float(state.steer)
        
        # Front Left & Right Wheels (at +lf along forward axis, rotated by theta + steer)
        for side in [track_w, -track_w]:
            wx = float(state.x) + lf * cos_th + side * (-sin_th)
            wy = float(state.y) + lf * sin_th + side * cos_th
            draw_oriented_box(
                screen, WHEEL_COLOR, wx, wy,
                WHEEL_LENGTH_M, WHEEL_WIDTH_M, car_th + steer_angle, cam_x, cam_y
            )
            
        # Rear Left & Right Wheels (at -lr along forward axis, rotated by theta)
        for side in [track_w, -track_w]:
            wx = float(state.x) - lr * cos_th + side * (-sin_th)
            wy = float(state.y) - lr * sin_th + side * cos_th
            draw_oriented_box(
                screen, WHEEL_COLOR, wx, wy,
                WHEEL_LENGTH_M, WHEEL_WIDTH_M, car_th, cam_x, cam_y
            )
            
        # E. Car Body (Length along forward axis, Width along lateral axis)
        draw_oriented_box(
            screen, CAR_BODY_COLOR, float(state.x), float(state.y),
            CAR_LENGTH_M, CAR_WIDTH_M, car_th, cam_x, cam_y
        )
        
        # Cabin / Roof
        roof_offset = -0.3
        roof_x = float(state.x) + roof_offset * cos_th
        roof_y = float(state.y) + roof_offset * sin_th
        draw_oriented_box(
            screen, CAR_ROOF_COLOR, roof_x, roof_y,
            CAR_LENGTH_M * 0.45, CAR_WIDTH_M * 0.75, car_th, cam_x, cam_y
        )
        
        # Windshield
        windshield_offset = 0.6
        ws_x = float(state.x) + windshield_offset * cos_th
        ws_y = float(state.y) + windshield_offset * sin_th
        draw_oriented_box(
            screen, CAR_GLASS_COLOR, ws_x, ws_y,
            0.5, CAR_WIDTH_M * 0.68, car_th, cam_x, cam_y
        )
        
        # Front Headlights (Yellow/White, facing forward)
        hl_dist = CAR_LENGTH_M * 0.46
        hl_w = CAR_WIDTH_M * 0.35
        for side in [hl_w, -hl_w]:
            hx = float(state.x) + hl_dist * cos_th + side * (-sin_th)
            hy = float(state.y) + hl_dist * sin_th + side * cos_th
            hsx, hsy = world_to_screen(hx, hy, cam_x, cam_y)
            pygame.draw.circle(screen, (255, 255, 210), (hsx, hsy), 4)
            
        # Rear Taillights (Red, facing backward)
        tl_dist = -CAR_LENGTH_M * 0.48
        for side in [hl_w, -hl_w]:
            tx = float(state.x) + tl_dist * cos_th + side * (-sin_th)
            ty = float(state.y) + tl_dist * sin_th + side * cos_th
            tsx, tsy = world_to_screen(tx, ty, cam_x, cam_y)
            pygame.draw.circle(screen, (255, 40, 40), (tsx, tsy), 4)
            
        # F. Modern HUD & Telemetry
        current_speed_kmh = math.hypot(float(state.vx), float(state.vy)) * 3.6
        desired_speed_kmh = desired_speed * 3.6
        
        hud_rect = pygame.Rect(20, 20, 360, 260)
        hud_surf = pygame.Surface((hud_rect.width, hud_rect.height), pygame.SRCALPHA)
        hud_surf.fill(HUD_BG)
        pygame.draw.rect(hud_surf, (80, 90, 110), hud_surf.get_rect(), 1, border_radius=8)
        screen.blit(hud_surf, hud_rect.topleft)
        
        # Mode Badge
        mode_text = "MODE: AI DRIVER (2D Target Vector)" if is_ai_mode else "MODE: MANUAL DRIFT"
        mode_color = TARGET_COLOR if is_ai_mode else DRIFT_COLOR
        screen.blit(font_med.render(mode_text, True, mode_color), (35, 32))
        
        # Speedometer
        speed_str = f"SPEED: {current_speed_kmh:5.1f} km/h"
        target_str = f" (TARGET: {desired_speed_kmh:4.0f})" if is_ai_mode else ""
        screen.blit(font_med.render(speed_str + target_str, True, TEXT_COLOR), (35, 65))
        
        # Speed bar
        bar_w = 320
        pygame.draw.rect(screen, (40, 44, 54), (35, 95, bar_w, 10), border_radius=4)
        speed_fill = min(1.0, current_speed_kmh / 100.0) * bar_w
        pygame.draw.rect(screen, (70, 160, 240), (35, 95, int(speed_fill), 10), border_radius=4)
        
        # Drift Meter
        slip_text = f"LATERAL SLIP: {lateral_slip:4.2f} m/s"
        screen.blit(font_small.render(slip_text, True, TEXT_COLOR), (35, 118))
        if is_sliding:
            drift_badge = font_med.render("★ DRIFTING ★", True, (255, 60, 60))
            screen.blit(drift_badge, (215, 115))
            
        # Drift bar
        pygame.draw.rect(screen, (40, 44, 54), (35, 140, bar_w, 8), border_radius=4)
        slip_fill = min(1.0, lateral_slip / 4.0) * bar_w
        pygame.draw.rect(screen, DRIFT_COLOR if is_sliding else (100, 120, 140), (35, 140, int(slip_fill), 8), border_radius=4)
        
        # Steering & Torque meters
        steer_deg = math.degrees(steer_angle)
        screen.blit(font_small.render(f"STEERING: {steer_deg:5.1f}°", True, TEXT_COLOR), (35, 160))
        screen.blit(font_small.render(f"TORQUE:   {torque_act:+4.2f}", True, TEXT_COLOR), (200, 160))
        
        # Quick Help
        screen.blit(font_small.render("[M] Toggle Mode | [R] Reset | [ESC] Exit", True, (160, 170, 190)), (35, 195))
        if is_ai_mode:
            screen.blit(font_small.render("Mouse Distance = Speed | Scroll: Sensitivity", True, (130, 200, 160)), (35, 218))
        else:
            screen.blit(font_small.render("WASD: Drive/Steer | SPACE: Handbrake", True, (240, 190, 120)), (35, 218))
            
        pygame.display.flip()
        
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
