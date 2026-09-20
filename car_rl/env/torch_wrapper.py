"""
Zero-Copy DLPack Wrapper for Batched JAX Car Environment.

Bridges JAX GPU simulations directly with PyTorch training loops without CPU copies.
"""

from typing import Tuple, Dict, Any
import torch
import jax
import jax.numpy as jnp
from car_rl.env.car_env import (
    EnvParams,
    EnvState,
    env_reset,
    env_step,
)


class TorchBatchedCarEnv:
    """
    Vectorized JAX environment with direct PyTorch CUDA tensor interface.
    """
    def __init__(
        self,
        num_envs: int = 2048,
        params: EnvParams = EnvParams(),
        seed: int = 42,
        device: str = "cuda:0",
    ):
        self.num_envs = num_envs
        self.params = params
        self.device = torch.device(device)
        self.obs_dim = 8
        self.act_dim = 2
        
        # JIT-compiled vectorized functions
        self._v_reset = jax.jit(jax.vmap(env_reset, in_axes=(0, None)))
        self._v_step = jax.jit(jax.vmap(env_step, in_axes=(0, 0, None)))
        
        # Initialize PRNG keys
        root_key = jax.random.PRNGKey(seed)
        self.keys = jax.random.split(root_key, num_envs)
        
        # Internal JAX state
        self._state: EnvState = None
        
    def reset(self) -> torch.Tensor:
        """
        Resets all parallel environments.
        Returns: obs as PyTorch CUDA tensor [num_envs, obs_dim].
        """
        states, obs = self._v_reset(self.keys, self.params)
        self._state = states
        # Zero-copy conversion from JAX to PyTorch
        return torch.from_dlpack(obs)
        
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Steps all parallel environments with PyTorch actions.
        actions: PyTorch CUDA tensor [num_envs, 2] in [-1, 1].
        
        Returns:
            next_obs: PyTorch CUDA tensor [num_envs, 8]
            rewards: PyTorch CUDA tensor [num_envs]
            dones: PyTorch CUDA tensor [num_envs]
            infos: Dictionary of PyTorch metrics
        """
        assert actions.is_cuda, "Actions tensor must be on CUDA GPU"
        
        # Clamp actions to physical limits [-1, 1] and zero-copy transfer
        actions_clamped = torch.clamp(actions, -1.0, 1.0)
        actions_jax = jax.dlpack.from_dlpack(actions_clamped)
        
        # Vectorized environment step in JAX
        next_states, next_obs, rewards, dones, infos = self._v_step(
            self._state, actions_jax, self.params
        )
        self._state = next_states
        
        # Zero-copy conversion back to PyTorch
        obs_torch = torch.from_dlpack(next_obs)
        rewards_torch = torch.from_dlpack(rewards)
        dones_torch = torch.from_dlpack(dones)
        
        infos_torch = {
            "speed": torch.from_dlpack(infos["speed"]),
            "desired_vel": torch.from_dlpack(infos["desired_vel"]),
            "slip_speed": torch.from_dlpack(infos["slip_speed"]),
            "total_reward": torch.from_dlpack(infos["total_reward"]),
        }
        
        return obs_torch, rewards_torch, dones_torch, infos_torch
