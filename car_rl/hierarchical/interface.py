"""
Hierarchical Agent Interface Specification.

Defines the contract between:
- A Higher-Level Agent: senses obstacles (via 2D raycasts) and emits [unit_vector, desired_velocity].
- The Lower-Level Car Agent: translates [unit_vector, desired_velocity] into wheel steering and engine torque.
"""

from typing import NamedTuple, Protocol
import torch
import jax.numpy as jnp


class HighLevelAction(NamedTuple):
    """Command emitted by the high-level agent to drive the low-level car."""
    ux: float            # Desired direction unit vector X
    uy: float            # Desired direction unit vector Y
    desired_velocity: float  # Commanded scalar speed (m/s)


class HighLevelObservation(NamedTuple):
    """Perception received by the high-level agent."""
    raycast_distances: torch.Tensor  # Normalized obstacle distances [num_rays] in [0, 1]
    car_position: torch.Tensor       # [x, y] global position
    car_heading: float               # Theta yaw angle
    car_speed: float                 # Scalar speed magnitude
    target_destination: torch.Tensor # Final goal destination [gx, gy]


class HighLevelPolicy(Protocol):
    """Interface for future obstacle-avoidance high-level agent."""
    def act(self, obs: HighLevelObservation) -> HighLevelAction:
        """
        Evaluate raycasts and destination to output sub-goal direction and speed.
        """
        ...
