"""
Torch wrapper for batched track env with frozen low-level driver in the loop.

Per env step (all on GPU):
  high_action torch [E,2] (ego heading + speed)
    -> JAX world target [tvx, tvy]
    -> JAX low-level obs [E,8]
    -> torch frozen ActorCritic -> low_action [E,2]
    -> JAX track_step (physics + railings + reward)
Done envs get a fresh random track (new RNG), selected via tree where.
"""

from typing import Tuple, Dict, Any
import os
import torch
import jax
import jax.numpy as jnp

from car_rl.env.track_env import (
    TrackParams,
    TrackState,
    track_reset,
    track_step,
    compute_high_obs,
    compute_low_obs,
    high_to_target,
    HIGH_OBS_DIM,
    HIGH_ACT_DIM,
)
from car_rl.models.actor_critic import ActorCritic


class TorchBatchedTrackEnv:
    def __init__(
        self,
        num_envs: int = 1024,
        params: TrackParams = TrackParams(),
        seed: int = 42,
        device: str = "cuda:0",
        low_checkpoint: str = "checkpoints/best_model.pt",
        low_hidden: int = 256,
    ):
        self.num_envs = num_envs
        self.params = params
        self.device = torch.device(device)
        self.obs_dim = HIGH_OBS_DIM
        self.act_dim = HIGH_ACT_DIM

        # Frozen low-level driver
        self.low = ActorCritic(obs_dim=8, act_dim=2, hidden_dim=low_hidden).to(self.device)
        if os.path.exists(low_checkpoint):
            self.low.load_state_dict(torch.load(low_checkpoint, map_location=self.device))
            print(f"[track] loaded frozen low-level driver from {low_checkpoint}")
        else:
            print(f"[track] WARNING: {low_checkpoint} missing, using random low-level weights")
        self.low.eval()
        for p in self.low.parameters():
            p.requires_grad_(False)

        self._v_reset = jax.jit(jax.vmap(track_reset, in_axes=(0, None)))
        self._v_step = jax.jit(jax.vmap(track_step, in_axes=(0, 0, None)))
        self._v_low_obs = jax.jit(
            jax.vmap(compute_low_obs, in_axes=(0, 0, 0, None, None, None))
        )
        self._v_target = jax.jit(
            jax.vmap(high_to_target, in_axes=(0, 0, None))
        )
        self._v_high_obs = jax.jit(jax.vmap(compute_high_obs, in_axes=(0, None)))

        root = jax.random.PRNGKey(seed)
        self.keys = jax.random.split(root, num_envs)
        self._state: TrackState = None

    def reset(self) -> torch.Tensor:
        states, obs = self._v_reset(self.keys, self.params)
        self._state = states
        return torch.from_dlpack(obs).to(self.device)

    @torch.no_grad()
    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        assert actions.is_cuda, "high actions must be CUDA tensor"
        ha = torch.clamp(actions, -1.0, 1.0)
        ha_jax = jax.dlpack.from_dlpack(ha)

        thetas = self._state.vehicle.theta  # (E,)
        tvx, tvy, _, _ = self._v_target(ha_jax, thetas, self.params.max_desired_vel)
        low_obs_jax = self._v_low_obs(
            self._state.vehicle,
            tvx,
            tvy,
            self.params.v_scale,
            self.params.omega_scale,
            self.params.vehicle.max_steer,
        )
        low_obs_t = torch.from_dlpack(low_obs_jax).to(self.device)
        low_act_t = torch.clamp(self.low.act_deterministic(low_obs_t), -1.0, 1.0)
        low_act_jax = jax.dlpack.from_dlpack(low_act_t)

        nxt, obs, rew, done, info = self._v_step(self._state, low_act_jax, self.params)

        # Fresh random tracks for done envs (split state rng per env)
        split0, split1 = jax.vmap(lambda k: tuple(jax.random.split(k, 2)))(nxt.rng)
        rst, rst_obs = self._v_reset(split0, self.params)
        # select reset where done
        self._state = jax.tree.map(
            lambda a, b: jnp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), b, a)
            if isinstance(a, jnp.ndarray) or hasattr(a, "ndim")
            else a,
            nxt,
            rst,
        )
        # rng must advance even for non-done envs
        self._state = self._state._replace(rng=split1)

        obs_t = torch.from_dlpack(jnp.where(done[:, None], rst_obs, obs)).to(self.device)
        rew_t = torch.from_dlpack(rew).to(self.device)
        done_t = torch.from_dlpack(done).to(self.device)
        info_t = {
            "speed": torch.from_dlpack(info["speed"]).to(self.device),
            "ds": torch.from_dlpack(info["ds"]).to(self.device),
            "lap": torch.from_dlpack(info["lap"]).to(self.device),
            "wall_hit": torch.from_dlpack(info["wall_hit"]).to(self.device),
            "lapped": torch.from_dlpack(info["lapped"]).to(self.device),
            "track_length": torch.from_dlpack(info["track_length"]).to(self.device),
        }
        # expose low-level commands for logging/demo (steer, torque)
        info_t["low_act"] = low_act_t
        return obs_t, rew_t, done_t, info_t
