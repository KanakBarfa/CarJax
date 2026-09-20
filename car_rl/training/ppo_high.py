"""High-level track PPO: lidar racing policy over a frozen low-level driver."""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from car_rl.env.track_wrapper import TorchBatchedTrackEnv
from car_rl.env.track_env import TrackParams
from car_rl.models.actor_critic import ActorCritic
from car_rl.training.config import PPOConfig


class PPOHighTrainer:
    """PPO for the 11-dim lidar obs -> 2-dim ego command policy.

    Reuses PPOConfig fields (num_envs, num_steps, lr, gamma, ...).
    Only the high-level net is optimized; the low-level driver stays frozen.
    """

    def __init__(self, config: PPOConfig = PPOConfig(), track_params: TrackParams = TrackParams()):
        self.cfg = config
        self.device = torch.device(config.device)
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(self.cfg.tensorboard_dir, exist_ok=True)

        self.env = TorchBatchedTrackEnv(
            num_envs=self.cfg.num_envs,
            params=track_params,
            seed=self.cfg.seed,
            device=self.cfg.device,
        )
        self.agent = ActorCritic(
            obs_dim=self.env.obs_dim, act_dim=self.env.act_dim, hidden_dim=256
        ).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.cfg.learning_rate, eps=1e-5)
        self.writer = SummaryWriter(os.path.join(self.cfg.tensorboard_dir, self.cfg.exp_name))

        self.obs = torch.zeros((self.cfg.num_steps, self.cfg.num_envs, self.env.obs_dim), device=self.device)
        self.actions = torch.zeros((self.cfg.num_steps, self.cfg.num_envs, self.env.act_dim), device=self.device)
        self.logprobs = torch.zeros((self.cfg.num_steps, self.cfg.num_envs), device=self.device)
        self.rewards = torch.zeros((self.cfg.num_steps, self.cfg.num_envs), device=self.device)
        self.dones = torch.zeros((self.cfg.num_steps, self.cfg.num_envs), device=self.device)
        self.values = torch.zeros((self.cfg.num_steps, self.cfg.num_envs), device=self.device)

        self.batch_size = self.cfg.num_envs * self.cfg.num_steps
        self.minibatch_size = self.batch_size // self.cfg.num_minibatches
        self.num_iterations = max(1, self.cfg.total_timesteps // self.batch_size)
        self.best_reward = -float("inf")

    def train(self):
        print("=== Starting High-Level Track PPO ===")
        print(f"Envs: {self.cfg.num_envs} | Steps: {self.cfg.num_steps} | Batch: {self.batch_size:,}")
        print(f"Total: {self.cfg.total_timesteps:,} | Iters: {self.num_iterations} | {self.device}\n")

        global_step = 0
        start_time = time.time()
        next_obs = self.env.reset()
        next_done = torch.zeros(self.cfg.num_envs, device=self.device)

        for iteration in range(1, self.num_iterations + 1):
            iter_start = time.time()
            if self.cfg.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                self.optimizer.param_groups[0]["lr"] = frac * self.cfg.learning_rate

            sum_speed = sum_ds = sum_wall = sum_lap = 0.0
            for step in range(self.cfg.num_steps):
                global_step += self.cfg.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob
                next_obs, reward, next_done, info = self.env.step(action)
                self.rewards[step] = reward
                sum_speed += info["speed"].mean().item()
                sum_ds += info["ds"].mean().item()
                sum_wall += info["wall_hit"].mean().item()
                sum_lap += info["lap"].float().mean().item()

            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(-1)
                advantages = torch.zeros_like(self.rewards)
                lastgaelam = 0
                for t in reversed(range(self.cfg.num_steps)):
                    if t == self.cfg.num_steps - 1:
                        nextnonterminal = 1.0 - next_done.float()
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1].float()
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.cfg.gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = (
                        delta + self.cfg.gamma * self.cfg.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + self.values

            b_obs = self.obs.reshape((-1, self.env.obs_dim))
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1, self.env.act_dim))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            b_inds = torch.randperm(self.batch_size, device=self.device)
            for epoch in range(self.cfg.update_epochs):
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    newvalue = newvalue.view(-1)
                    if self.cfg.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds], -self.cfg.clip_coef, self.cfg.clip_coef
                        )
                        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    loss = pg_loss - self.cfg.ent_coef * entropy.mean() + self.cfg.vf_coef * v_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()

            iter_time = time.time() - iter_start
            sps = int(self.batch_size / max(iter_time, 1e-6))
            mean_reward = self.rewards.mean().item()
            n = self.cfg.num_steps
            print(
                f"Iter {iteration:03d}/{self.num_iterations:03d} | Step {global_step:,} | "
                f"R: {mean_reward:+6.3f} | v: {sum_speed/n:4.1f} m/s | "
                f"ds: {sum_ds/n:+.3f} m | wall: {sum_wall/n:.3f} | lap: {sum_lap/n:.2f} | SPS: {sps:,}"
            )
            self.writer.add_scalar("high/mean_reward", mean_reward, global_step)
            self.writer.add_scalar("high/mean_speed", sum_speed / n, global_step)
            self.writer.add_scalar("high/mean_ds", sum_ds / n, global_step)
            self.writer.add_scalar("high/wall_rate", sum_wall / n, global_step)

            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                torch.save(self.agent.state_dict(), os.path.join(self.cfg.checkpoint_dir, "high_best.pt"))
            if iteration % self.cfg.save_interval == 0 or iteration == self.num_iterations:
                torch.save(self.agent.state_dict(), os.path.join(self.cfg.checkpoint_dir, "high_latest.pt"))

        print(f"\nHigh-level training complete in {time.time()-start_time:.1f}s. Best R: {self.best_reward:.3f}")
        self.writer.close()


if __name__ == "__main__":
    PPOHighTrainer().train()
