"""Test suite for 2D JAX vehicle physics."""

import jax
import jax.numpy as jnp
from car_rl.env.physics import (
    VehicleParams,
    init_vehicle_state,
    step_physics,
    compute_tire_forces,
)


def test_straight_line_acceleration():
    params = VehicleParams()
    state = init_vehicle_state()
    
    # Apply full throttle (torque_cmd = 1.0) for 50 steps (1 second)
    step_jit = jax.jit(lambda s: step_physics(s, jnp.array(0.0), jnp.array(1.0), params))
    
    for _ in range(50):
        state = step_jit(state)
        
    print(f"After 1s acceleration: vx={float(state.vx):.2f} m/s, x={float(state.x):.2f} m")
    assert state.vx > 2.0, "Car should accelerate under full throttle"
    assert abs(state.vy) < 1e-4, "No lateral velocity on straight line"
    assert abs(state.omega) < 1e-4, "No yaw rate on straight line"
    print("✓ Straight line acceleration test passed.")


def test_steering_and_turning():
    params = VehicleParams()
    # Car moving forward at 10 m/s
    state = init_vehicle_state(vx=10.0)
    
    # Steer right (steer_cmd = 0.5) with moderate throttle
    step_jit = jax.jit(lambda s: step_physics(s, jnp.array(0.5), jnp.array(0.3), params))
    
    for _ in range(25):
        state = step_jit(state)
        
    print(f"After turn: steer={float(state.steer):.3f}, omega={float(state.omega):.3f}, theta={float(state.theta):.3f}")
    assert state.steer > 0.1, "Front wheels should be turned"
    assert abs(state.omega) > 0.1, "Car should have yaw rotation"
    print("✓ Steering and turning test passed.")


def test_drift_power_oversteer():
    params = VehicleParams()
    # High speed and sharp steering with maximum throttle
    state = init_vehicle_state(vx=15.0)
    
    step_jit = jax.jit(lambda s: step_physics(s, jnp.array(0.8), jnp.array(1.0), params))
    
    max_vy = 0.0
    for _ in range(40):
        state = step_jit(state)
        max_vy = max(max_vy, abs(float(state.vy)))
        
    print(f"Drift test: max lateral slip velocity |vy| = {max_vy:.2f} m/s")
    assert max_vy > 0.5, "Car should slide laterally (drift) during aggressive power turn"
    print("✓ Power oversteer / drift test passed.")


if __name__ == "__main__":
    test_straight_line_acceleration()
    test_steering_and_turning()
    test_drift_power_oversteer()
    print("\nALL PHYSICS TESTS PASSED SUCCESSFULLY!")
