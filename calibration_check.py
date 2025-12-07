#!/usr/bin/env python
"""
Calibration diagnostic tool for SO101 teleop.

Helps identify offsets between real arm and simulation.
Move the real arm to known positions and compare with sim.

Usage:
    python calibration_check.py
    python calibration_check.py --leader-port COM8
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

import env  # registers the environment

# Sim action space bounds (radians) - from so101_new_calib.xml actuator ctrlrange
SIM_ACTION_LOW = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453])
SIM_ACTION_HIGH = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533])

# Calibration offsets: add these to the radian values to align real with sim
# Tune these values until real arm matches sim arm position
JOINT_OFFSETS_RAD = np.array([
    0.0,    # shoulder_pan
    0.0,    # shoulder_lift
    0.0,    # elbow_flex
    0.0,    # wrist_flex
    0.0,    # wrist_roll
    0.0,    # gripper (try np.pi/2 if 90° off)
])

# Set to True to invert specific joints (flip direction)
JOINT_INVERT = np.array([
    False,  # shoulder_pan
    False,  # shoulder_lift
    False,  # elbow_flex
    False,  # wrist_flex
    False,  # wrist_roll
    False,  # gripper (try True if opens when should close)
])


def load_config():
    """Load config.json for COM port."""
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
    """Create motor bus for leader arm."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

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


def normalized_to_radians(normalized_values: np.ndarray) -> np.ndarray:
    """
    Convert from lerobot normalized values to sim radians.

    This is the CURRENT conversion - before any offset corrections.
    """
    radians = np.zeros(6, dtype=np.float32)

    # Joints 0-4: map [-100, 100] -> [low, high]
    for i in range(5):
        t = (normalized_values[i] + 100) / 200.0
        radians[i] = SIM_ACTION_LOW[i] + t * (SIM_ACTION_HIGH[i] - SIM_ACTION_LOW[i])

    # Gripper: map [0, 100] -> [low, high]
    t = normalized_values[5] / 100.0
    radians[5] = SIM_ACTION_LOW[5] + t * (SIM_ACTION_HIGH[5] - SIM_ACTION_LOW[5])

    return radians


def apply_calibration_offsets(radians: np.ndarray) -> np.ndarray:
    """Apply calibration offsets and inversions."""
    result = radians.copy()

    # Apply inversions
    for i in range(6):
        if JOINT_INVERT[i]:
            # Invert around the center of the range
            center = (SIM_ACTION_LOW[i] + SIM_ACTION_HIGH[i]) / 2
            result[i] = 2 * center - result[i]

    # Apply offsets
    result += JOINT_OFFSETS_RAD

    return result


def run_calibration_check(leader_port: str):
    """Run interactive calibration check."""

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"]

    print("="*60)
    print("CALIBRATION CHECK TOOL")
    print("="*60)
    print("\nThis tool helps you find the offset between real arm and sim.")
    print("\nInstructions:")
    print("1. Move your real arm to a known position")
    print("2. Compare with the simulated arm on screen")
    print("3. Note any differences")
    print("4. Edit JOINT_OFFSETS_RAD in this file to correct")
    print("\nKeys:")
    print("  q - quit")
    print("  r - reset sim")
    print("  p - print current values")
    print("  1-6 - adjust offset for joint (hold shift for negative)")
    print("="*60)

    # Connect to leader arm
    print(f"\nConnecting to leader arm on {leader_port}...")
    bus = create_leader_bus(leader_port)
    bus.connect()
    bus.calibration = load_calibration()
    bus.disable_torque()
    print("Leader arm connected!")

    # Create sim
    print("Creating simulation...")
    sim_env = gym.make("base-sO101-env-v0")
    obs, _ = sim_env.reset()

    cv2.namedWindow("Calibration Check", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration Check", 1000, 750)

    # Mutable offsets for interactive adjustment
    offsets = JOINT_OFFSETS_RAD.copy()

    try:
        while True:
            # Read leader positions
            positions = bus.sync_read("Present_Position")
            normalized = np.array([
                positions["shoulder_pan"],
                positions["shoulder_lift"],
                positions["elbow_flex"],
                positions["wrist_flex"],
                positions["wrist_roll"],
                positions["gripper"],
            ], dtype=np.float32)

            # Convert to radians
            radians_raw = normalized_to_radians(normalized)

            # Apply offsets
            radians_corrected = radians_raw + offsets

            # Apply inversions
            for i in range(6):
                if JOINT_INVERT[i]:
                    center = (SIM_ACTION_LOW[i] + SIM_ACTION_HIGH[i]) / 2
                    radians_corrected[i] = 2 * center - radians_corrected[i]

            # Clip to valid range
            radians_clipped = np.clip(radians_corrected, SIM_ACTION_LOW, SIM_ACTION_HIGH)

            # Step sim
            obs, _, _, _, _ = sim_env.step(radians_clipped)

            # Display
            frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

            # Add text overlay
            cv2.putText(frame, "CALIBRATION CHECK", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            y = 70
            for i, name in enumerate(joint_names):
                # Show: normalized -> raw_rad (deg) -> corrected_rad (deg)
                raw_deg = np.degrees(radians_raw[i])
                clip_deg = np.degrees(radians_clipped[i])
                offset_deg = np.degrees(offsets[i])
                text = f"{name[:12]:12s}: {normalized[i]:6.1f} -> {radians_clipped[i]:6.3f} rad ({clip_deg:6.1f}°)"
                offset_text = f"(offset: {offsets[i]:+.3f} = {offset_deg:+.1f}°)"
                cv2.putText(frame, text, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                cv2.putText(frame, offset_text, (500, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                y += 25

            # Conversion reference
            y += 20
            cv2.putText(frame, "Degrees: 1 rad = 57.3 deg, 0.25 rad = 14.3 deg", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y += 20
            cv2.putText(frame, "Keys: 1-6 adjust offset (+0.05), Shift+1-6 (-0.05), p=print, q=quit", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            cv2.imshow("Calibration Check", frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                obs, _ = sim_env.reset()
            elif key == ord('p'):
                print("\n" + "="*40)
                print("Current values:")
                print(f"Normalized: {normalized}")
                print(f"Raw radians: {radians_raw}")
                print(f"Offsets: {offsets}")
                print(f"Corrected: {radians_clipped}")
                print("\nCopy this to JOINT_OFFSETS_RAD:")
                print(f"JOINT_OFFSETS_RAD = np.array({offsets.tolist()})")
                print("="*40)
            # Number keys 1-6 adjust offsets
            elif ord('1') <= key <= ord('6'):
                idx = key - ord('1')
                offsets[idx] += 0.05  # +2.86 degrees
                print(f"Joint {idx} offset: {offsets[idx]:.3f} rad ({np.degrees(offsets[idx]):.1f} deg)")
            elif key == ord('!'):  # Shift+1
                offsets[0] -= 0.05
            elif key == ord('@'):  # Shift+2
                offsets[1] -= 0.05
            elif key == ord('#'):  # Shift+3
                offsets[2] -= 0.05
            elif key == ord('$'):  # Shift+4
                offsets[3] -= 0.05
            elif key == ord('%'):  # Shift+5
                offsets[4] -= 0.05
            elif key == ord('^'):  # Shift+6
                offsets[5] -= 0.05

            time.sleep(0.01)

    finally:
        cv2.destroyAllWindows()
        sim_env.close()
        bus.disconnect()

        print("\n" + "="*60)
        print("FINAL OFFSETS")
        print("="*60)
        print("\nCopy these to teleop_sim.py and calibration_check.py:")
        print(f"\nJOINT_OFFSETS_RAD = np.array({offsets.tolist()})")
        print(f"\n# In degrees: {np.degrees(offsets).tolist()}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Calibration check tool")
    parser.add_argument("--leader-port", "-p", type=str, default=None)
    args = parser.parse_args()

    leader_port = args.leader_port
    if leader_port is None:
        config = load_config()
        if config and "leader" in config:
            leader_port = config["leader"]["port"]
        else:
            leader_port = "COM8"

    run_calibration_check(leader_port)


if __name__ == "__main__":
    main()
