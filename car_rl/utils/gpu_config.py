import os
import sys

def setup_gpu_environment(jax_mem_fraction: float = 0.4):
    """
    Configures environment variables before JAX and PyTorch are initialized.
    Prevents JAX from grabbing 75-90% of GPU memory by default, ensuring
    PyTorch and JAX can comfortably share the GPU without Out-Of-Memory errors.
    """
    if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(jax_mem_fraction)
        
    # Prevent TF/XLA spam
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
