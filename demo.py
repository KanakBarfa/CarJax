#!/usr/bin/env python3
"""
Evaluation and Visualization Demo for Trained Car RL Agent.

Executes a single-vehicle rollout with dynamically changing commanded unit vectors
and velocities, plotting the 2D path, drift angles, speed tracking, and controls.
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless-safe rendering
import matplotlib.pyplot as plt
import torch
import jax
import jax.numpy as jnp

from car_rl.env.car_env import (
    EnvParams,
    EnvState,
    env_reset,
    env_step,
    compute_observation,
)
from car_rl.env.physics import VehicleParams, init_vehicle_state
from car_rl.models.actor_critic import ActorCritic


def run_demo(checkpoint_path: str = "checkpoints/best_model.pt", output_plot: str = "car_trajectory.png", steps: int = 400):
    params = EnvParams()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    agent = ActorCritic(obs_dim=9, act_dim=2).to(device)
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
        agent.eval()
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Running with initial weights for visualization.")

    # Initialize environment in JAX
    rng = jax.random.PRNGKey(123)
    rng, rng_init = jax.random.split(rng)
    veh = init_vehicle_state(vx=5.0)
    
    def get_command(step_idx: int):
        if step_idx < 100:
            angle = 0.0          # East
            vel = 12.0
        elif step_idx < 200:
            angle = np.pi / 2.0  # North
            vel = 16.0
        elif step_idx < 300:
            angle = -3.0 * np.pi / 4.0 # South-West
            vel = 20.0
        else:
            angle = -np.pi / 4.0 # South-East
            vel = 8.0
        return np.cos(angle) * vel, np.sin(angle) * vel, vel

    tvx0, tvy0, v0 = get_command(0)
    state = EnvState(
        vehicle=veh,
        target_vx=jnp.array(tvx0, dtype=jnp.float32),
        target_vy=jnp.array(tvy0, dtype=jnp.float32),
        step_count=jnp.array(0, dtype=jnp.int32),
        total_reward=jnp.array(0.0, dtype=jnp.float32),
        rng=rng,
    )
    
    # Trajectory logging
    traj_x, traj_y, traj_theta = [], [], []
    traj_vx, traj_vy = [], []
    traj_steer, traj_torque = [], []
    traj_target_vel = []
    traj_target_angles = []
    
    print(f"Simulating {steps} dynamic vehicle steps ({steps * params.vehicle.dt:.1f} seconds)...")
    
    for t in range(steps):
        tvx_cmd, tvy_cmd, vel_cmd = get_command(t)
        state = state._replace(
            target_vx=jnp.array(tvx_cmd, dtype=jnp.float32),
            target_vy=jnp.array(tvy_cmd, dtype=jnp.float32),
        )
        
        obs_jax = compute_observation(state, params)
        obs_torch = torch.from_dlpack(obs_jax).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_torch = agent.act_deterministic(obs_torch)
            action_jax = jax.dlpack.from_dlpack(action_torch.squeeze(0))
            
        next_state, _, _, _, _ = env_step(state, action_jax, params)
        
        veh_curr = state.vehicle
        traj_x.append(float(veh_curr.x))
        traj_y.append(float(veh_curr.y))
        traj_theta.append(float(veh_curr.theta))
        traj_vx.append(float(veh_curr.vx))
        traj_vy.append(float(veh_curr.vy))
        traj_steer.append(float(veh_curr.steer))
        traj_torque.append(float(action_torch[0, 1].item()))
        traj_target_vel.append(vel_cmd)
        traj_target_angles.append(np.arctan2(tvy_cmd, tvx_cmd))
        
        state = next_state

    # Plot results
    time_arr = np.arange(steps) * params.vehicle.dt
    speed_arr = np.sqrt(np.array(traj_vx)**2 + np.array(traj_vy)**2)
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.8, 1, 1])
    
    # 1. 2D World Trajectory
    ax_traj = fig.add_subplot(gs[0, :])
    ax_traj.plot(traj_x, traj_y, "b-", linewidth=2.5, label="Car Trajectory")
    ax_traj.plot(traj_x[0], traj_y[0], "go", markersize=10, label="Start")
    ax_traj.plot(traj_x[-1], traj_y[-1], "ro", markersize=10, label="Finish")
    
    # Draw vehicle orientation arrows periodically
    arrow_interval = max(1, steps // 20)
    for i in range(0, steps, arrow_interval):
        dx = np.cos(traj_theta[i]) * 2.5
        dy = np.sin(traj_theta[i]) * 2.5
        # Red arrows if drifting (|vy| > 0.5), cyan otherwise
        is_drifting = abs(traj_vy[i]) > 0.5
        color = "red" if is_drifting else "darkcyan"
        ax_traj.arrow(
            traj_x[i], traj_y[i], dx, dy,
            head_width=1.0, head_length=1.2, fc=color, ec=color, alpha=0.8
        )
    ax_traj.set_title("2D Vehicle Trajectory & Orientation (Red = Lateral Drift/Slide)", fontsize=13)
    ax_traj.set_xlabel("X (meters)")
    ax_traj.set_ylabel("Y (meters)")
    ax_traj.axis("equal")
    ax_traj.grid(True, linestyle="--", alpha=0.6)
    ax_traj.legend()

    # 2. Speed Tracking
    ax_speed = fig.add_subplot(gs[1, 0])
    ax_speed.plot(time_arr, speed_arr, "b-", linewidth=2, label="Actual Speed")
    ax_speed.plot(time_arr, traj_target_vel, "r--", linewidth=2, label="Commanded Speed")
    ax_speed.set_title("Velocity Tracking (m/s)")
    ax_speed.set_xlabel("Time (s)")
    ax_speed.set_ylabel("Speed (m/s)")
    ax_speed.grid(True, linestyle="--", alpha=0.6)
    ax_speed.legend()

    # 3. Lateral Slip Speed (Drift Magnitude)
    ax_slip = fig.add_subplot(gs[1, 1])
    ax_slip.plot(time_arr, np.abs(traj_vy), "r-", linewidth=2, label="|vy| (Slip Speed)")
    ax_slip.axhline(0.5, color="orange", linestyle=":", label="Drift Threshold")
    ax_slip.set_title("Lateral Slip Velocity (Drifting Indicator)")
    ax_slip.set_xlabel("Time (s)")
    ax_slip.set_ylabel("Lateral Speed (m/s)")
    ax_slip.grid(True, linestyle="--", alpha=0.6)
    ax_slip.legend()

    # 4. Steering Angle
    ax_steer = fig.add_subplot(gs[2, 0])
    ax_steer.plot(time_arr, np.degrees(traj_steer), "g-", linewidth=2, label="Wheel Angle (deg)")
    ax_steer.set_title("Front Wheel Steering Angle (degrees)")
    ax_steer.set_xlabel("Time (s)")
    ax_steer.set_ylabel("Steering (deg)")
    ax_steer.grid(True, linestyle="--", alpha=0.6)
    ax_steer.legend()

    # 5. Engine Torque Command
    ax_torque = fig.add_subplot(gs[2, 1])
    ax_torque.plot(time_arr, traj_torque, "m-", linewidth=2, label="Torque Actuation [-1, 1]")
    ax_torque.set_title("Engine Drive / Brake Torque Command")
    ax_torque.set_xlabel("Time (s)")
    ax_torque.set_ylabel("Normalized Torque")
    ax_torque.grid(True, linestyle="--", alpha=0.6)
    ax_torque.legend()

    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    plt.close()
    print(f"\n✓ Trajectory and telemetry plot saved to {output_plot}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--output", type=str, default="car_trajectory.png")
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    
    run_demo(args.checkpoint, args.output, args.steps)
