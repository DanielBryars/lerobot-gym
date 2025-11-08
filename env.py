from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class SO101Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        xml_pth: Path = Path("assets/SO-ARM100/Simulation/SO101/scene_with_cube.xml"),
        width: int = 640,
        height: int = 480,
        max_episode_steps: int = 1_000,
        camera_distance: float = 1.0,
        camera_azimuth: int = 100,
    ) -> None:
        """Most simple S0101 environment. Reinforcement learning environment where reward is
        defined by the euclidian distance between the gripper and a red block that it needs to pick up.



        Args:
            xml_pth (Path, optional): Path to the scene .xml file that containing the robot and the cube it needs to pickup. Defaults to Path("assets/SO-ARM100/Simulation/SO101/scene_with_cube.xml").
            width (int, optional): Render width. Defaults to 640.
            height (int, optional): _description_. Defaults to 480.
            max_episode_steps (int, optional): Size of on Episode. Defaults to 200.
            camera_distance (float, optional): Distance of the render camera to the robot. Defaults to 1.0.
            camera_azimuth (int, optional): Azimuth of the render camera. Defaults to 100.
        """
        self.mj_model = mujoco.MjModel.from_xml_path(str(xml_pth))
        self.mj_data = mujoco.MjData(self.mj_model)
        self.width = width
        self.height = height
        self.mj_renderer = mujoco.Renderer(
            self.mj_model, height=self.height, width=self.width
        )

        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.action_space = gym.spaces.Box(
            low=np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453]),
            high=np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533]),
            shape=(self.mj_model.nu,),  # Number of actuators
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.gripper_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1"
        )
        self.cube_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"
        )
        self.camera_distance = camera_distance
        self.camera_azimuth = camera_azimuth

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Apply the action and update the scene
        self.mj_data.ctrl = action
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.mj_renderer.update_scene(self.mj_data)

        # Get a rendered observation
        obs = self._get_obs()

        # Compute the reward
        reward = self._compute_reward()

        # Check if the episode is terminated or truncated
        terminated = self.current_step >= self.max_episode_steps
        truncated = False
        info = {}

        self.current_step += 1

        return obs, reward, terminated, truncated, info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to the initial state.
        Returns:
            np.ndarray: Initial observation
            dict: Additional information
        """
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.current_step = 0
        obs = self._get_obs()

        return obs, {}

    def _compute_reward(self) -> float:
        """Compute the reward as the negative Euclidean distance between the gripper and the cube."""
        # Get the positions of the gripper and cube geoms
        gripper_pos = self.mj_data.geom_xpos[self.gripper_geom_id]
        cube_pos = self.mj_data.geom_xpos[self.cube_geom_id]

        # Return the negative distance as the reward
        return -np.linalg.norm(gripper_pos - cube_pos)

    def _get_obs(self) -> np.ndarray:
        """Render observation

        Returns:
            np.ndarray: Obervation, rendered image
        """
        self.mj_renderer.update_scene(self.mj_data)
        camera = mujoco.MjvCamera()
        camera.distance = self.camera_distance  # Decrease this value to zoom in
        camera.azimuth = self.camera_azimuth  # Camera azimuth angle (degrees)
        self.mj_renderer.update_scene(self.mj_data, camera=camera)
        return self.mj_renderer.render().copy()

    def close(self) -> None:
        # del self.mj_renderer
        del self.mj_data
        del self.mj_model


gym.register(
    id="base-sO101-env-v0",
    entry_point=SO101Env,
)


def make_env(n_envs: int = 1, use_async_envs: bool = False):
    """
    Create vectorized environments for your custom task.

    Args:
        n_envs: Number of parallel environments
        use_async_envs: Whether to use AsyncVectorEnv or SyncVectorEnv

    Returns:
        gym.vector.VectorEnv or dict mapping suite names to vectorized envs
    """

    def _make_single_env():
        # Create your custom environment
        return gym.make("base-sO101-env-v0")

    # Choose vector environment type
    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv

    # Create vectorized environment
    vec_env = env_cls([_make_single_env for _ in range(n_envs)])

    return vec_env
