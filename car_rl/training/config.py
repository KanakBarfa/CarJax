"""Training and algorithm hyperparameters configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PPOConfig:
    # Experiment
    exp_name: str = "car_low_level_ppo"
    seed: int = 42
    device: str = "cuda:0"
    
    # Environment & Rollout
    num_envs: int = 2048
    num_steps: int = 64            # Steps per rollout: 2048 * 64 = 131,072 steps / iter
    total_timesteps: int = 15_000_000
    
    # PPO Hyperparameters
    learning_rate: float = 8e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8       # Mini-batch size = 16,384 transitions
    update_epochs: int = 4
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    
    # Checkpointing & Logging
    save_interval: int = 10        # Save checkpoint every N iterations
    log_interval: int = 1          # Log metrics every N iterations
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "runs"
