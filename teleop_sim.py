#!/usr/bin/env python
"""
Teleoperate the SO101 simulation using a physical leader arm.

Look at the sim output and move your leader arm to control the simulated robot.
Great for checking calibration matches between real and sim.

Usage:
    python teleop_sim.py
    python teleop_sim.py --leader-port COM8 --fps 30
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

import env  # registers the environment


# Sim action space bounds (radians)
SIM_ACTION_LOW = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453])
SIM_ACTION_HIGH = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533])


def normalized_to_radians(normalized_values: np.ndarray) -> np.ndarray:
    """
    Convert from lerobot normalized values to sim radians.

    Leader returns:
    - Joints 0-4: normalized to [-100, 100]
    - Gripper (5): normalized to [0, 100]

    Sim expects radians within SIM_ACTION_LOW to SIM_ACTION_HIGH.
    """
    radians = np.zeros(6, dtype=np.float32)

    # Joints 0-4: map [-100, 100] -> [low, high]
    for i in range(5):
        norm_val = normalized_values[i]
        # Map from [-100, 100] to [0, 1]
        t = (norm_val + 100) / 200.0
        # Map to [low, high]
        radians[i] = SIM_ACTION_LOW[i] + t * (SIM_ACTION_HIGH[i] - SIM_ACTION_LOW[i])

    # Gripper: map [0, 100] -> [low, high]
    t = normalized_values[5] / 100.0
    radians[5] = SIM_ACTION_LOW[5] + t * (SIM_ACTION_HIGH[5] - SIM_ACTION_LOW[5])

    return radians


def load_config():
    """Try to load config.json from lerobot-scratch repo."""
    config_paths = [
        Path("E:/git/ai/lerobot-scratch/config.json"),
        Path("../lerobot-scratch/config.json"),
        Path("config.json"),
    ]
    for path in config_paths:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def create_leader_bus(port: str):
    """Create a FeetechMotorsBus for leader arm with STS3250 motors."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    # Use sts3250 instead of sts3215
    bus = FeetechMotorsBus(
        port=port,
        motors={
            "shoulder_pan": Motor(1, "sts3250", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "sts3250", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "sts3250", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "sts3250", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "sts3250", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "sts3250", MotorNormMode.RANGE_0_100),
        },
    )
    return bus


def load_calibration(arm_id: str = "leader_so100"):
    """Load calibration from JSON file."""
    import draccus
    from lerobot.motors import MotorCalibration

    calib_path = Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/so100_leader_sts3250" / f"{arm_id}.json"

    if not calib_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {calib_path}\n"
            "Run 'python calibrate_from_zero.py --leader' first."
        )

    with open(calib_path) as f, draccus.config_type("json"):
        return draccus.load(dict[str, MotorCalibration], f)


def run_teleop(leader_port: str, fps: int = 30):
    """Run teleoperation with physical leader arm controlling sim."""

    # Create leader arm connection with STS3250 motors
    print(f"Connecting to leader arm on {leader_port} (STS3250 motors)...")
    bus = create_leader_bus(leader_port)
    bus.connect()

    # Load calibration from JSON file (created by calibrate_from_zero.py)
    print("Loading calibration from file...")
    bus.calibration = load_calibration()

    # Disable torque so leader arm can be moved freely
    bus.disable_torque()
    print("Leader arm connected!")

    # Create simulation environment
    print("Creating SO101 simulation...")
    sim_env = gym.make("base-sO101-env-v0")
    obs_image, _ = sim_env.reset()

    # Setup display window
    cv2.namedWindow("SO101 Teleop Sim", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SO101 Teleop Sim", 800, 600)

    print(f"\nTeleoperation started at {fps} FPS")
    print("Move your leader arm to control the simulation")
    print("Press 'q' to quit, 'r' to reset")
    print("Camera: arrow keys to rotate, +/- to zoom\n")

    frame_time = 1.0 / fps
    step_count = 0

    # Camera controls (access underlying env)
    cam_azimuth = sim_env.unwrapped.cam_azi
    cam_distance = sim_env.unwrapped.cam_dis
    cam_elevation = sim_env.unwrapped.cam_elev

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    try:
        while True:
            loop_start = time.time()

            # Read leader arm positions
            positions = bus.sync_read("Present_Position")

            # Extract joint positions in order
            normalized = np.array([
                positions["shoulder_pan"],
                positions["shoulder_lift"],
                positions["elbow_flex"],
                positions["wrist_flex"],
                positions["wrist_roll"],
                positions["gripper"],
            ], dtype=np.float32)

            # Convert to radians for sim
            joint_radians = normalized_to_radians(normalized)

            # Clip to valid range (just in case)
            joint_radians = np.clip(joint_radians, SIM_ACTION_LOW, SIM_ACTION_HIGH)

            # Send to simulation
            obs_image, reward, terminated, truncated, info = sim_env.step(joint_radians)
            step_count += 1

            # Display
            frame = cv2.cvtColor(obs_image, cv2.COLOR_RGB2BGR)
            cv2.putText(frame, f"Step: {step_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Reward: {reward:.3f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show joint values (both normalized and radians)
            for i, name in enumerate(joint_names):
                deg = np.degrees(joint_radians[i])
                cv2.putText(frame, f"{name[:10]}: {normalized[i]:6.1f} -> {joint_radians[i]:5.2f}rad ({deg:6.1f}°)",
                           (10, 100 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            # Show camera info
            cv2.putText(frame, f"Cam: azi={cam_azimuth} elev={cam_elevation} dist={cam_distance:.1f} (wasd/+-)",
                       (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            cv2.imshow("SO101 Teleop Sim", frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested")
                break
            elif key == ord('r'):
                print("Resetting simulation...")
                obs_image, _ = sim_env.reset()
                step_count = 0
            # Camera controls: a/d = rotate, w/s = elevation, +/- = zoom
            elif key == ord('a'):  # Rotate left
                cam_azimuth -= 5
                sim_env.unwrapped.cam_azi = cam_azimuth
            elif key == ord('d'):  # Rotate right
                cam_azimuth += 5
                sim_env.unwrapped.cam_azi = cam_azimuth
            elif key == ord('w'):  # Tilt up
                cam_elevation = min(89, cam_elevation + 5)
                sim_env.unwrapped.cam_elev = cam_elevation
            elif key == ord('s'):  # Tilt down
                cam_elevation = max(-89, cam_elevation - 5)
                sim_env.unwrapped.cam_elev = cam_elevation
            elif key == ord('+') or key == ord('='):  # Zoom in
                cam_distance = max(0.3, cam_distance - 0.1)
                sim_env.unwrapped.cam_dis = cam_distance
            elif key == ord('-'):  # Zoom out
                cam_distance += 0.1
                sim_env.unwrapped.cam_dis = cam_distance

            # Maintain frame rate
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    finally:
        cv2.destroyAllWindows()
        sim_env.close()
        bus.disconnect()
        print("Teleop ended")


def main():
    parser = argparse.ArgumentParser(description="Teleoperate SO101 sim with leader arm")
    parser.add_argument("--leader-port", "-p", type=str, default=None,
                        help="Serial port for leader arm (default: from config.json)")
    parser.add_argument("--fps", "-f", type=int, default=30,
                        help="Target frame rate (default: 30)")

    args = parser.parse_args()

    # Get leader port from config if not specified
    leader_port = args.leader_port
    if leader_port is None:
        config = load_config()
        if config and "leader" in config:
            leader_port = config["leader"]["port"]
            print(f"Using leader port from config: {leader_port}")
        else:
            leader_port = "COM8"  # fallback default
            print(f"No config found, using default: {leader_port}")

    run_teleop(leader_port, args.fps)


if __name__ == "__main__":
    main()
