#!/usr/bin/env python
"""
Show the SO101 simulation at the "zero" calibration pose.

This displays what the arm should look like when calibrating.
Match your physical arm to this pose before running calibration.

Usage:
    python show_zero_pose.py
"""
import time
import cv2
import gymnasium as gym
import numpy as np

import env  # registers the environment


def main():
    print("="*60)
    print("SO101 CALIBRATION POSE VIEWER")
    print("="*60)
    print("\nThis shows the arm at the 'zero' position for each joint.")
    print("Match your physical arm to this pose when calibrating.")
    print("\nPress 'q' to quit, 's' to save screenshot")
    print("="*60)

    # Create sim
    sim_env = gym.make("base-sO101-env-v0")
    obs, _ = sim_env.reset()

    cv2.namedWindow("Zero Pose", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Zero Pose", 1000, 750)

    # Joint positions at "zero" (all zeros = middle of range for new_calib)
    # Gripper at closed position (-0.17453 rad = low end of range)
    zero_pose = np.array([
        0.0,      # shoulder_pan - pointing forward
        0.0,      # shoulder_lift - horizontal
        0.0,      # elbow_flex - middle bend
        0.0,      # wrist_flex - level
        0.0,      # wrist_roll - gripper jaws horizontal
        -0.17453, # gripper - CLOSED (low end of range)
    ], dtype=np.float32)

    print("\nZero pose (radians):", zero_pose)
    print("Zero pose (degrees):", np.degrees(zero_pose))

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"]

    while True:
        # Step sim with zero pose
        obs, _, _, _, _ = sim_env.step(zero_pose)

        # Display
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

        # Add text
        cv2.putText(frame, "CALIBRATION ZERO POSE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, "Match your physical arm to this!", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        y = 100
        for i, name in enumerate(joint_names):
            deg = np.degrees(zero_pose[i])
            cv2.putText(frame, f"{name}: {zero_pose[i]:.3f} rad ({deg:.1f}°)",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y += 25

        y += 20
        cv2.putText(frame, "Gripper: CLOSED", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        y += 25
        cv2.putText(frame, "Wrist roll: jaws VERTICAL (like gripping a bar)", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

        cv2.imshow("Zero Pose", frame)

        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("zero_pose.png", frame)
            print("Saved zero_pose.png")

    cv2.destroyAllWindows()
    sim_env.close()


if __name__ == "__main__":
    main()
