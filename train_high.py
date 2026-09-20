#!/usr/bin/env python3
"""Train the high-level lidar racing agent (frozen low-level driver)."""

import argparse
from car_rl.training.config import PPOConfig
from car_rl.training.ppo_high import PPOHighTrainer
from car_rl.env.track_env import TrackParams


def main():
    p = argparse.ArgumentParser(description="Train high-level track racing policy")
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--total-timesteps", type=int, default=30_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exp-name", type=str, default="car_high_ppo")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--gamma", type=float, default=0.995)
    args = p.parse_args()

    config = PPOConfig(
        exp_name=args.exp_name,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        learning_rate=args.lr,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        gamma=args.gamma,
    )
    PPOHighTrainer(config, TrackParams()).train()


if __name__ == "__main__":
    main()
