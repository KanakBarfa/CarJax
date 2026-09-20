"""Car RL: GPU-accelerated 2D vehicle dynamic simulation and RL training."""

from car_rl.utils.gpu_config import setup_gpu_environment

# Ensure GPU memory settings are configured before any JAX/PyTorch imports
setup_gpu_environment()

__version__ = "0.1.0"
