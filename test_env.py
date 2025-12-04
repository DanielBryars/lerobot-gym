"""Quick test of SO101 environment."""
import env
import gymnasium as gym

print("Creating environment...")
e = gym.make("base-sO101-env-v0")

print("Resetting...")
obs, _ = e.reset()

print(f"Obs shape: {obs.shape}")
print(f"Action space: {e.action_space}")

print("\nRunning 10 random steps...")
for i in range(10):
    action = e.action_space.sample()
    obs, reward, terminated, truncated, info = e.step(action)
    print(f"  Step {i+1}: reward={reward:.3f}")

e.close()
print("\nSuccess!")
