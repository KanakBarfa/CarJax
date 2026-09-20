"""
Vectorized PyTorch PPO Trainer for Batched JAX Car Simulation.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from car_rl.env.torch_wrapper import TorchBatchedCarEnv
from car_rl.models.actor_critic import ActorCritic
from car_rl.training.config import PPOConfig


class PPOTrainer:
    def __init__(self, config: PPOConfig = PPOConfig()):
        self.cfg = config
        self.device = torch.device(config.device)
        
        # Create output directories
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(self.cfg.tensorboard_dir, exist_ok=True)
        
        # Initialize vectorized JAX environment
        self.env = TorchBatchedCarEnv(
            num_envs=self.cfg.num_envs,
            seed=self.cfg.seed,
            device=self.cfg.device,
        )
        
        # Initialize Actor-Critic network
        self.agent = ActorCritic(
            obs_dim=self.env.obs_dim,
            act_dim=self.env.act_dim,
            hidden_dim=256,
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.cfg.learning_rate, eps=1e-5
        )
        
        self.writer = SummaryWriter(os.path.join(self.cfg.tensorboard_dir, self.cfg.exp_name))
        
        # Preallocate rollout buffers directly on GPU
        self.obs = torch.zeros((self.cfg.num_steps, self.cfg.num_envs, self.env.obs_dim), device=self.device)
        self.actions = torch.zeros((self.cfg.num_steps, self.cfg.num_envs, self.env.act_dim), device=self.device)
        self.logprobs = torch.zeros((self.cfg.num_steps, self.cfg.num_envs), device=self.device)
        self.rewards = torch.zeros((self.cfg.num_steps, self.cfg.num_envs), device=self.device)
        self.dones = torch.zeros((self.cfg.num_steps, self.cfg.num_envs), device=self.device)
        self.values = torch.zeros((self.cfg.num_steps, self.cfg.num_envs), device=self.device)
        
        self.batch_size = self.cfg.num_envs * self.cfg.num_steps
        self.minibatch_size = self.batch_size // self.cfg.num_minibatches
        self.num_iterations = self.cfg.total_timesteps // self.batch_size
        
        self.best_reward = -float("inf")

    def train(self):
        print(f"=== Starting PPO Training ===")
        print(f"Parallel Envs: {self.cfg.num_envs} | Rollout Steps: {self.cfg.num_steps}")
        print(f"Batch Size: {self.batch_size:,} | Total Timesteps: {self.cfg.total_timesteps:,}")
        print(f"Iterations: {self.num_iterations} | Device: {self.device}\n")
        
        global_step = 0
        start_time = time.time()
        
        # Initial reset
        next_obs = self.env.reset()
        next_done = torch.zeros(self.cfg.num_envs, device=self.device)
        
        for iteration in range(1, self.num_iterations + 1):
            iter_start = time.time()
            
            # Anneal learning rate
            if self.cfg.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lr_now = frac * self.cfg.learning_rate
                self.optimizer.param_groups[0]["lr"] = lr_now
            
            # 1. Collect Rollout
            tracked_speeds = []
            tracked_desired = []
            tracked_slips = []
            
            for step in range(self.cfg.num_steps):
                global_step += self.cfg.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done
                
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    self.values[step] = value.flatten()
                    
                self.actions[step] = action
                self.logprobs[step] = logprob
                
                # Step environment
                next_obs, reward, next_done, info = self.env.step(action)
                self.rewards[step] = reward
                
                tracked_speeds.append(info["speed"].mean().item())
                tracked_desired.append(info["desired_vel"].mean().item())
                tracked_slips.append(info["slip_speed"].mean().item())
            
            # 2. Compute Generalized Advantage Estimation (GAE)
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
                    advantages[t] = lastgaelam = delta + self.cfg.gamma * self.cfg.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values
                
            # 3. Flatten the batch
            b_obs = self.obs.reshape((-1, self.env.obs_dim))
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1, self.env.act_dim))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)
            
            # 4. PPO Mini-batch Optimization
            b_inds = torch.randperm(self.batch_size, device=self.device)
            clipfracs = []
            
            for epoch in range(self.cfg.update_epochs):
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]
                    
                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean().item()]
                        
                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    
                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.cfg.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.cfg.clip_coef,
                            self.cfg.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                        
                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.cfg.ent_coef * entropy_loss + self.cfg.vf_coef * v_loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()
                    
            iter_time = time.time() - iter_start
            sps = int(self.batch_size / iter_time)
            mean_reward = self.rewards.mean().item()
            mean_speed = sum(tracked_speeds) / len(tracked_speeds)
            mean_des_speed = sum(tracked_desired) / len(tracked_desired)
            mean_slip = sum(tracked_slips) / len(tracked_slips)
            
            # Logging
            if iteration % self.cfg.log_interval == 0:
                print(
                    f"Iter {iteration:03d}/{self.num_iterations:03d} | "
                    f"Step: {global_step:,} | "
                    f"Reward: {mean_reward:6.2f} | "
                    f"Speed: {mean_speed:4.1f}/{mean_des_speed:4.1f} m/s | "
                    f"Slip: {mean_slip:4.2f} m/s | "
                    f"SPS: {sps:,}"
                )
                self.writer.add_scalar("charts/mean_reward", mean_reward, global_step)
                self.writer.add_scalar("charts/mean_speed", mean_speed, global_step)
                self.writer.add_scalar("charts/mean_slip", mean_slip, global_step)
                self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                
            # Save checkpoints
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                torch.save(
                    self.agent.state_dict(),
                    os.path.join(self.cfg.checkpoint_dir, "best_model.pt"),
                )
                
            if iteration % self.cfg.save_interval == 0 or iteration == self.num_iterations:
                torch.save(
                    self.agent.state_dict(),
                    os.path.join(self.cfg.checkpoint_dir, "latest_model.pt"),
                )
                
        total_time = time.time() - start_time
        print(f"\nTraining Complete in {total_time:.1f}s! Best Reward: {self.best_reward:.2f}")
        self.writer.close()


if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train()
