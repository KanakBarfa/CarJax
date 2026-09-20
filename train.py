#!/usr/bin/env python3
"""
Training CLI entry point for low-level car agent.
"""

import argparse
from car_rl.training.config import PPOConfig
from car_rl.training.ppo import PPOTrainer


def main():
    parser = argparse.ArgumentParser(description="Train 2D GPU Car RL Policy using JAX + PyTorch")
    parser.add_argument("--num-envs", type=int, default=2048, help="Number of parallel JAX environments on GPU")
    parser.add_argument("--num-steps", type=int, default=64, help="Rollout steps per environment")
    parser.add_argument("--total-timesteps", type=int, default=3_000_000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--exp-name", type=str, default="car_drift_ppo", help="Experiment name")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoints directory")
    
    args = parser.parse_args()
    
    config = PPOConfig(
        exp_name=args.exp_name,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        learning_rate=args.lr,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    trainer = PPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
