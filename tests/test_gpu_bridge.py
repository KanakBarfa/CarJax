"""Benchmark and verification test for zero-copy GPU DLPack bridge."""

import time
import torch
from car_rl.env.torch_wrapper import TorchBatchedCarEnv


def test_gpu_bridge_and_perf():
    num_envs = 2048
    num_steps = 100
    
    print(f"Initializing {num_envs} parallel JAX environments on GPU...")
    env = TorchBatchedCarEnv(num_envs=num_envs)
    
    obs = env.reset()
    assert obs.is_cuda, "Observation must be a PyTorch CUDA tensor"
    assert obs.shape == (num_envs, 8), f"Expected shape ({num_envs}, 8), got {obs.shape}"
    
    # Warmup step (triggers JIT compilation)
    print("Running warmup step (triggering JAX JIT)...")
    actions = torch.zeros((num_envs, 2), device="cuda:0", dtype=torch.float32)
    env.step(actions)
    torch.cuda.synchronize()
    
    print(f"Running benchmark: {num_steps} steps across {num_envs} environments...")
    start_time = time.perf_counter()
    
    for _ in range(num_steps):
        # Generate random actions on GPU
        actions = 2.0 * torch.rand((num_envs, 2), device="cuda:0", dtype=torch.float32) - 1.0
        next_obs, rewards, dones, infos = env.step(actions)
        
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    total_samples = num_envs * num_steps
    fps = total_samples / total_time
    
    print(f"\nCompleted {total_samples:,} environment steps in {total_time:.3f} seconds!")
    print(f"Throughput: {fps:,.0f} steps/second on RTX 3060!")
    print(f"Observation shape: {next_obs.shape}, device: {next_obs.device}")
    print(f"Reward shape: {rewards.shape}, device: {rewards.device}")
    
    assert fps > 30000, f"Expected >30k SPS, got {fps}"
    print("✓ GPU DLPack Bridge & Throughput benchmark PASSED!")


if __name__ == "__main__":
    test_gpu_bridge_and_perf()
