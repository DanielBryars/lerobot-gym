#!/usr/bin/env python
"""
Simulated SO100/SO101 Follower Robot for LeRobot.

This module provides a simulated version of the SO100 follower arm that implements
the same Robot interface as the real hardware. It can be used as a drop-in replacement
for recording datasets, running inference, etc.

The simulation uses MuJoCo for physics and can optionally render to VR.

Usage:
    from so100_sim_follower import SO100SimFollower, SO100SimFollowerConfig

    config = SO100SimFollowerConfig(
        id="sim_follower",
        cameras={"wrist_cam": OpenCVCameraConfig(index=0, width=640, height=480, fps=30)},
    )
    robot = SO100SimFollower(config)
    robot.connect()

    # Use like a real robot
    obs = robot.get_observation()
    robot.send_action({"shoulder_pan.pos": 0.0, ...})
"""

import logging
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from lerobot.cameras import CameraConfig
from lerobot.motors import MotorCalibration
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)


# Motor names in order
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Sim action space bounds (radians) - matches the real robot
SIM_ACTION_LOW = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453])
SIM_ACTION_HIGH = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533])

# Default scene XML path
DEFAULT_SCENE_XML = Path(__file__).parent / "scenes" / "so101_with_wrist_cam.xml"


def radians_to_normalized(radians: np.ndarray, use_degrees: bool = False) -> dict[str, float]:
    """Convert sim radians to lerobot normalized values."""
    result = {}
    for i, name in enumerate(MOTOR_NAMES):
        if use_degrees:
            result[name] = float(np.degrees(radians[i]))
        else:
            # Map radians to [-100, 100] for joints, [0, 100] for gripper
            if name == "gripper":
                t = (radians[i] - SIM_ACTION_LOW[i]) / (SIM_ACTION_HIGH[i] - SIM_ACTION_LOW[i])
                result[name] = float(t * 100)
            else:
                t = (radians[i] - SIM_ACTION_LOW[i]) / (SIM_ACTION_HIGH[i] - SIM_ACTION_LOW[i])
                result[name] = float(t * 200 - 100)
    return result


def normalized_to_radians(normalized: dict[str, float], use_degrees: bool = False) -> np.ndarray:
    """Convert lerobot normalized values to sim radians."""
    radians = np.zeros(6, dtype=np.float32)
    for i, name in enumerate(MOTOR_NAMES):
        val = normalized.get(name, 0.0)
        if use_degrees:
            radians[i] = np.radians(val)
        else:
            if name == "gripper":
                t = val / 100.0
            else:
                t = (val + 100) / 200.0
            radians[i] = SIM_ACTION_LOW[i] + t * (SIM_ACTION_HIGH[i] - SIM_ACTION_LOW[i])
    return radians


@RobotConfig.register_subclass("so100_sim_follower")
@dataclass
class SO100SimFollowerConfig(RobotConfig):
    """Configuration for the simulated SO100 follower."""

    # Path to MuJoCo scene XML (uses default if not specified)
    scene_xml: Path | None = None

    # Number of physics steps per action
    n_sim_steps: int = 10

    # Camera configs - simulated cameras render from MuJoCo
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Camera dimensions for MuJoCo rendering
    camera_width: int = 640
    camera_height: int = 480

    # Use degrees instead of normalized [-100, 100] (for backward compat)
    use_degrees: bool = False

    # Enable VR output (requires OpenXR setup)
    enable_vr: bool = False


class SO100SimFollower(Robot):
    """
    Simulated SO100 Follower Arm using MuJoCo.

    This is a drop-in replacement for the real SO100Follower that uses
    MuJoCo for physics simulation instead of real hardware.
    """

    config_class = SO100SimFollowerConfig
    name = "so100_sim_follower"

    def __init__(self, config: SO100SimFollowerConfig):
        super().__init__(config)
        self.config = config

        # MuJoCo state (initialized on connect)
        self.mj_model = None
        self.mj_data = None
        self.mj_renderer = None
        self._connected = False

        # VR renderer (optional)
        self.vr_renderer = None

        # Scene XML path
        self.scene_xml = config.scene_xml if config.scene_xml else DEFAULT_SCENE_XML

        # Track camera IDs in the MuJoCo model
        self._camera_ids = {}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in MOTOR_NAMES}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.camera_height, self.config.camera_width, 3)
            for cam in self.config.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        # Simulation doesn't need calibration
        return True

    def connect(self, calibrate: bool = True) -> None:
        """Initialize the MuJoCo simulation."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info(f"Loading MuJoCo scene: {self.scene_xml}")
        self.mj_model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.mj_data = mujoco.MjData(self.mj_model)

        # Create renderer for camera observations
        self.mj_renderer = mujoco.Renderer(
            self.mj_model,
            height=self.config.camera_height,
            width=self.config.camera_width
        )

        # Find camera IDs in the model
        for cam_name in self.config.cameras:
            try:
                cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                self._camera_ids[cam_name] = cam_id
                logger.info(f"Found camera '{cam_name}' with ID {cam_id}")
            except Exception:
                logger.warning(f"Camera '{cam_name}' not found in MuJoCo model, will use default view")
                self._camera_ids[cam_name] = None

        # Initialize simulation
        for _ in range(100):
            mujoco.mj_step(self.mj_model, self.mj_data)

        # Initialize VR if enabled
        if self.config.enable_vr:
            self._init_vr()

        self._connected = True
        logger.info(f"{self} connected (simulated).")

    def _init_vr(self):
        """Initialize VR renderer if enabled."""
        try:
            from teleop_sim_vr import VRRenderer
            self.vr_renderer = VRRenderer(self.mj_model, self.mj_data)
            self.vr_renderer.init_all()
            logger.info("VR renderer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize VR: {e}")
            self.vr_renderer = None

    def calibrate(self) -> None:
        """No calibration needed for simulation."""
        pass

    def configure(self) -> None:
        """No configuration needed for simulation."""
        pass

    def get_observation(self) -> dict[str, Any]:
        """Get current joint positions and camera images."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        # Read joint positions from simulation
        joint_radians = self.mj_data.qpos[:6].copy()
        obs_dict = radians_to_normalized(joint_radians, self.config.use_degrees)
        obs_dict = {f"{motor}.pos": obs_dict[motor] for motor in MOTOR_NAMES}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Render camera images
        for cam_name in self.config.cameras:
            start = time.perf_counter()
            cam_id = self._camera_ids.get(cam_name)

            if cam_id is not None:
                # Render from named camera
                self.mj_renderer.update_scene(self.mj_data, camera=cam_id)
            else:
                # Render from default view
                self.mj_renderer.update_scene(self.mj_data)

            obs_dict[cam_name] = self.mj_renderer.render().copy()

            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} render {cam_name}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send action to simulation and step physics."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract goal positions
        goal_pos = {}
        for key, val in action.items():
            if key.endswith(".pos"):
                motor = key.removesuffix(".pos")
                goal_pos[motor] = val

        # Convert to radians
        joint_radians = normalized_to_radians(goal_pos, self.config.use_degrees)

        # Clip to valid range
        joint_radians = np.clip(joint_radians, SIM_ACTION_LOW, SIM_ACTION_HIGH)

        # Apply to simulation
        self.mj_data.ctrl[:6] = joint_radians

        # Step physics
        for _ in range(self.config.n_sim_steps):
            mujoco.mj_step(self.mj_model, self.mj_data)

        # Update VR display if enabled
        if self.vr_renderer is not None:
            if not self.vr_renderer.render_frame():
                logger.warning("VR session ended")
                self.vr_renderer = None

        # Return the action that was actually sent
        return {f"{motor}.pos": goal_pos.get(motor, 0.0) for motor in MOTOR_NAMES}

    def disconnect(self) -> None:
        """Clean up simulation resources."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.vr_renderer is not None:
            self.vr_renderer.cleanup()
            self.vr_renderer = None

        if self.mj_renderer is not None:
            del self.mj_renderer
            self.mj_renderer = None

        if self.mj_data is not None:
            del self.mj_data
            self.mj_data = None

        if self.mj_model is not None:
            del self.mj_model
            self.mj_model = None

        self._connected = False
        logger.info(f"{self} disconnected.")


# For convenience, also create an SO101 variant
@RobotConfig.register_subclass("so101_sim_follower")
@dataclass
class SO101SimFollowerConfig(SO100SimFollowerConfig):
    """Configuration for the simulated SO101 follower (same as SO100)."""
    pass


class SO101SimFollower(SO100SimFollower):
    """Simulated SO101 Follower (same as SO100 for now)."""
    config_class = SO101SimFollowerConfig
    name = "so101_sim_follower"


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    config = SO100SimFollowerConfig(
        id="test_sim",
        cameras={"wrist_cam": None},  # Will use default view
    )

    robot = SO100SimFollower(config)
    robot.connect()

    print("\nObservation features:", robot.observation_features)
    print("Action features:", robot.action_features)

    # Get observation
    obs = robot.get_observation()
    print("\nObservation:")
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: shape={val.shape}")
        else:
            print(f"  {key}: {val:.2f}")

    # Send an action
    action = {f"{motor}.pos": 0.0 for motor in MOTOR_NAMES}
    result = robot.send_action(action)
    print("\nAction sent:", result)

    robot.disconnect()
    print("\nTest complete!")
