"""
PyTorch Continuous Actor-Critic Network for Vehicle Control.

Standard Gaussian Policy (CleanRL continuous PPO formulation):
- Linear output layer (no terminal tanh saturation).
- Action mean directly models continuous control.
- Mathematically consistent Normal log-probabilities.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer: nn.Linear, std: float = 1.414, bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal layer initialization for RL networks."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 9, act_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        
        # Critic network (Value estimator V(s))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        
        # Actor network (Policy mean mu(s))
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.3),
        )
        
        # Learnable log standard deviation for continuous Gaussian policy (sigma ~ 0.45)
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), -0.8))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate state value V(s)."""
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples an action from the policy distribution or computes log-prob of an existing action.
        """
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(x)
        
        return action, log_prob, entropy, value

    def act_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic policy action for evaluation / deployment."""
        return torch.clamp(self.actor_mean(x), -1.0, 1.0)
