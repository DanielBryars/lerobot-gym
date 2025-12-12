from pathlib import Path
from typing import Any, Optional

import cv2
import gymnasium as gym
import mujoco
import numpy as np


def load_model_with_wrist_camera(xml_path: Path) -> mujoco.MjModel:
    """Load MuJoCo model and add a wrist camera attached to the gripper body.

    Uses MuJoCo's spec API to programmatically add a camera to the gripper body,
    keeping changes in the main repo rather than modifying the submodule's XML.
    """
    # Load the model spec
    spec = mujoco.MjSpec.from_file(str(xml_path))

    # Find the gripper body by iterating through all bodies
    gripper_body = None
    for body in spec.bodies:
        if body.name == "gripper":
            gripper_body = body
            break

    if gripper_body is None:
        print("Warning: Could not find gripper body, wrist camera not added")
        return mujoco.MjModel.from_xml_path(str(xml_path))

    # Add wrist camera to gripper body
    # Position it behind/above the gripper jaws, looking down at them
    # Gripper jaws are in -Z direction (gripperframe at z=-0.098)
    cam = gripper_body.add_camera()
    cam.name = "wrist_cam"
    cam.pos = [0.0, 0.03, -0.02]  # Offset up (Y) and slightly toward jaws (-Z)
    # Tilt down ~60 degrees around X axis to look at gripper jaws
    # quat for 60° rotation around X: [cos(30°), sin(30°), 0, 0] = [0.866, 0.5, 0, 0]
    cam.quat = [0.866, 0.5, 0.0, 0.0]
    cam.fovy = 75  # Wider FOV to see more of the workspace

    # Compile and return the model
    return spec.compile()


class SO101Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    # Get absolute path to assets relative to this file's location
    _DEFAULT_XML = Path(__file__).parent.parent / "assets" / "SO-ARM100" / "Simulation" / "SO101" / "scene_with_cube.xml"

    def __init__(
        self,
        xml_pth: Path = None,
        obs_h: int = 480,
        obs_w: int = 640,
        n_max_epi_steps: int = 1_000,
        cam_dis: float = 1.0,
        cam_azi: int = 100,
        cam_elev: int = -20,
        n_sim_steps: int = 10,
        add_wrist_camera: bool = True,
    ) -> None:
        """Most simple S0101 environment. Reinforcement learning environment where reward is
        defined by the euclidian distance between the gripper and a red block that it needs to pick up.


        Args:
            xml_pth (Path, optional): Path to the scene .xml file that containing the robot and the cube it needs to pickup. Defaults to Path("assets/SO-ARM100/Simulation/SO101/scene_with_cube.xml").
            obs_w (int, optional): Render obs_w. Defaults to 640.
            obs_h (int, optional): _description_. Defaults to 480.
            n_max_epi_steps (int, optional): Size of on Episode. Defaults to 1_000.
            cam_dis (float, optional): Distance of the render camera to the robot. Defaults to 1.0.
            cam_azi (int, optional): Azimuth of the render camera. Defaults to 100.
            cam_elev (int, optional): Elevation of the render camera. Defaults to -20.
            n_sim_steps (int, optional): Number of mujoco simulation steps.
            add_wrist_camera (bool, optional): Add wrist-mounted ego camera. Defaults to True.
        """
        if xml_pth is None:
            xml_pth = self._DEFAULT_XML

        # Load model - optionally with wrist camera added programmatically
        if add_wrist_camera:
            self.mj_model = load_model_with_wrist_camera(xml_pth)
        else:
            self.mj_model = mujoco.MjModel.from_xml_path(str(xml_pth))
        self.mj_data = mujoco.MjData(self.mj_model)

        # Store wrist camera ID if it exists
        self.wrist_cam_id = None
        if add_wrist_camera:
            try:
                self.wrist_cam_id = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam"
                )
            except Exception:
                pass
        self.obs_h = obs_h
        self.obs_w = obs_w
        self.n_sim_steps = n_sim_steps
        self.mj_renderer = mujoco.Renderer(
            self.mj_model, height=self.obs_h, width=self.obs_w
        )

        self.n_max_epi_steps = n_max_epi_steps
        self.current_step = 0
        self.action_space = gym.spaces.Box(
            low=np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453]),
            high=np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533]),
            shape=(self.mj_model.nu,),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.obs_h, self.obs_w, 3), dtype=np.uint8
        )
        self.gripper_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1"
        )
        self.cube_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"
        )
        self.cam_dis = cam_dis
        self.cam_azi = cam_azi
        self.cam_elev = cam_elev

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Apply the action and update the scene
        self.mj_data.ctrl = action
        for _ in range(self.n_sim_steps):
            mujoco.mj_step(self.mj_model, self.mj_data)
        self.mj_renderer.update_scene(self.mj_data)
        
        # Get a rendered observation
        obs = self._get_obs()

        # Compute the reward
        reward = self._compute_reward()

        # Check if the episode is terminated or truncated
        terminated = self.current_step >= self.n_max_epi_steps
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
        mj_cam = mujoco.MjvCamera()
        mj_cam.distance = self.cam_dis
        mj_cam.azimuth = self.cam_azi
        mj_cam.elevation = self.cam_elev
        self.mj_renderer.update_scene(self.mj_data, camera=mj_cam)
        return self.mj_renderer.render().copy()

    def get_wrist_camera_obs(self, width: int = 640, height: int = 480) -> Optional[np.ndarray]:
        """Render observation from wrist-mounted ego camera.

        Args:
            width: Image width (default 640)
            height: Image height (default 480)

        Returns:
            np.ndarray: RGB image from wrist camera, or None if camera not available
        """
        if self.wrist_cam_id is None:
            return None

        # Create a renderer at the requested size if needed
        if not hasattr(self, '_wrist_renderer') or \
           self._wrist_renderer_size != (width, height):
            self._wrist_renderer = mujoco.Renderer(self.mj_model, height=height, width=width)
            self._wrist_renderer_size = (width, height)

        # Update scene with wrist camera
        self._wrist_renderer.update_scene(self.mj_data, camera=self.wrist_cam_id)
        return self._wrist_renderer.render().copy()

    def close(self) -> None:
        # del self.mj_renderer
        del self.mj_data
        del self.mj_model

