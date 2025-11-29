from pathlib import Path
import cv2
import numpy as np
import imageio
import gymnasium as gym
from src.base_so101_env import SO101Env

gripper_close = 0.05
env = SO101Env(
    xml_pth=Path("assets/SO-ARM100/Simulation/SO101/scene_with_cube.xml"), 
    obs_w=640, 
    obs_h=480,
    n_sim_steps=10,
    # cam_azi = 170,
)

frames = []   # <--- store frames here

try:
    obs, _ = env.reset()
    action = np.array([0.0, 0.0, 0.0, 1, 1.5, 2])

    # warm-up steps
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(action)

    def run_and_capture(action, steps):
        for _ in range(steps):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            frames.append(obs)

    run_and_capture(np.array([0.1, 0.2, 0.2, 1, 1.5, 2]), 10)
    run_and_capture(np.array([0.0, 0.2, 0.2, 1, 1.5, gripper_close]), 20)
    run_and_capture(np.array([0, -0.6, 0.2, 1, 1.5, gripper_close]), 20)
    run_and_capture(np.array([0, -0.6, 0.1, 1, 1.5, gripper_close]), 20)
    run_and_capture(np.array([0, -0.6, -0.0, 1, 1.5, gripper_close]), 20)
    run_and_capture(np.array([0, -0.6, -0.2, 1, 1.5, gripper_close]), 20)
    run_and_capture(np.array([0, -0.6, -0.4, 1, 1.5, gripper_close]), 20)

finally:
    env.close()

imageio.mimsave("assets/media/output.gif", frames, fps=20)
