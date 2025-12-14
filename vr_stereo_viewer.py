"""
VR Stereo Viewer for Quest 3.

Renders side-by-side (SBS) stereo view of the simulation for viewing
in VR via Virtual Desktop, Quest Link, or Skybox.

Usage:
    python vr_stereo_viewer.py

    Then in Quest 3:
    - Virtual Desktop: Enable "SBS 3D" or "Half SBS" mode
    - Skybox: Select "Side by Side" 3D format
    - Quest Link: Use SteamVR with SBS viewer

Controls:
    Keyboard:
        Q/A - Shoulder pan +/-
        W/S - Shoulder lift +/-
        E/D - Elbow flex +/-
        R/F - Wrist flex +/-
        T/G - Wrist roll +/-
        Y/H - Gripper open/close

        1-4 - Preset arm poses
        V   - Toggle VR mode (stereo/mono)
        F   - Toggle fullscreen
        ESC - Quit
"""

import cv2
import numpy as np
import mujoco
import time
import json
from pathlib import Path


SCENE_XML = Path(__file__).parent / "scenes" / "so101_with_wrist_cam.xml"
SETTINGS_FILE = Path(__file__).parent / ".vr_stereo_settings.json"

# VR display settings
# Quest 3 resolution per eye: 2064x2208, but we render smaller for performance
VR_EYE_WIDTH = 1920  # Per eye width
VR_EYE_HEIGHT = 1080  # Per eye height
IPD = 0.063  # Inter-pupillary distance in meters (63mm average)

# Joint limits (radians)
JOINT_LIMITS = [
    (-1.92, 1.92),   # shoulder_pan
    (-1.75, 1.75),   # shoulder_lift
    (-1.69, 1.69),   # elbow_flex
    (-1.66, 1.66),   # wrist_flex
    (-2.74, 2.84),   # wrist_roll
    (-0.17, 1.75),   # gripper
]


class VRStereoViewer:
    def __init__(self):
        print(f"Loading {SCENE_XML}...")
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)

        # VR camera settings (will be positioned in scene)
        self.vr_pos = np.array([0.4, 0.3, 0.4])  # VR headset position
        self.vr_azimuth = -120  # Looking at robot
        self.vr_elevation = -20
        self.vr_distance = 0.0  # Not used, we use pos directly

        # Joint positions
        self.joint_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        self.joint_step = 0.05

        # Load saved settings
        self._load_settings()

        # Apply initial joint positions
        self._apply_joints()

        # Stereo renderers (one per eye)
        self.left_renderer = mujoco.Renderer(self.model, height=VR_EYE_HEIGHT, width=VR_EYE_WIDTH)
        self.right_renderer = mujoco.Renderer(self.model, height=VR_EYE_HEIGHT, width=VR_EYE_WIDTH)

        # Display state
        self.vr_mode = True  # Stereo SBS mode
        self.fullscreen = False

        # Window
        self.window_name = "VR Stereo Viewer"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1920, 540)  # Half height for SBS preview

    def _load_settings(self):
        """Load settings from file if it exists."""
        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE) as f:
                    settings = json.load(f)
                self.vr_pos = np.array(settings.get("vr_pos", self.vr_pos.tolist()))
                self.vr_azimuth = settings.get("vr_azimuth", self.vr_azimuth)
                self.vr_elevation = settings.get("vr_elevation", self.vr_elevation)
                self.joint_pos = np.array(settings.get("joint_pos", self.joint_pos.tolist()))
                print("Loaded saved VR settings.")
        except Exception:
            pass

    def _save_settings(self):
        """Save current settings to file."""
        try:
            settings = {
                "vr_pos": self.vr_pos.tolist(),
                "vr_azimuth": self.vr_azimuth,
                "vr_elevation": self.vr_elevation,
                "joint_pos": self.joint_pos.tolist(),
            }
            with open(SETTINGS_FILE, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass

    def _apply_joints(self):
        """Apply current joint positions to simulation."""
        self.data.ctrl[:] = self.joint_pos
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

    def _adjust_joint(self, idx, delta):
        """Adjust a joint by delta, respecting limits."""
        lo, hi = JOINT_LIMITS[idx]
        self.joint_pos[idx] = np.clip(self.joint_pos[idx] + delta, lo, hi)
        self._save_settings()

    def render_eye(self, renderer, eye_offset):
        """Render from one eye's perspective."""
        cam = mujoco.MjvCamera()

        # Calculate eye position with IPD offset
        # The offset is perpendicular to the viewing direction
        azimuth_rad = np.radians(self.vr_azimuth)

        # Offset perpendicular to viewing direction (in XY plane)
        offset_x = -np.sin(azimuth_rad) * eye_offset
        offset_y = np.cos(azimuth_rad) * eye_offset

        cam.lookat[0] = self.vr_pos[0] + offset_x
        cam.lookat[1] = self.vr_pos[1] + offset_y
        cam.lookat[2] = self.vr_pos[2]

        cam.distance = 0.5  # Distance from lookat to camera
        cam.azimuth = self.vr_azimuth
        cam.elevation = self.vr_elevation

        renderer.update_scene(self.data, camera=cam)
        return renderer.render()

    def render_stereo(self):
        """Render side-by-side stereo frame."""
        # Render left and right eyes
        left = self.render_eye(self.left_renderer, -IPD / 2)
        right = self.render_eye(self.right_renderer, IPD / 2)

        # Combine side by side
        stereo = np.hstack([left, right])
        return stereo

    def render_mono(self):
        """Render single mono frame (center eye)."""
        return self.render_eye(self.left_renderer, 0)

    def run(self):
        print("\n" + "="*70)
        print("VR STEREO VIEWER")
        print("="*70)
        print("Connect Quest 3 via Virtual Desktop or Quest Link")
        print("Enable 'SBS 3D' or 'Half SBS' mode in your VR app")
        print("")
        print("Controls:")
        print("  Q/A W/S E/D R/F T/G Y/H - Joint control")
        print("  Arrow keys - Move VR viewpoint")
        print("  Page Up/Down - Move up/down")
        print("  V - Toggle stereo/mono")
        print("  F - Toggle fullscreen")
        print("  1-4 - Preset poses")
        print("  ESC - Quit")
        print("="*70 + "\n")

        presets = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            [0.3, -0.5, 0.8, 0.5, 0.0, 0.5],
            [0.5, -0.7, 1.0, 0.8, 0.3, 0.2],
            [-0.5, -0.3, 0.5, 0.3, -0.5, 0.8],
        ]

        while True:
            self._apply_joints()

            # Render
            if self.vr_mode:
                frame = self.render_stereo()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame = self.render_mono()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Add mode indicator (small, won't be visible in VR)
            mode_text = "STEREO SBS" if self.vr_mode else "MONO"
            cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(16) & 0xFF  # ~60fps

            if key == 27:  # ESC
                break
            elif key == ord('v'):
                self.vr_mode = not self.vr_mode
                print(f"VR mode: {'STEREO' if self.vr_mode else 'MONO'}")
            elif key == ord('f'):
                self.fullscreen = not self.fullscreen
                if self.fullscreen:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            # Preset poses
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                self.joint_pos = np.array(presets[key - ord('1')])
                self._save_settings()
            # VR position controls (arrow keys)
            elif key == 81 or key == ord(','):  # Left arrow
                self.vr_azimuth -= 5
                self._save_settings()
            elif key == 83 or key == ord('.'):  # Right arrow
                self.vr_azimuth += 5
                self._save_settings()
            elif key == 82:  # Up arrow - move forward
                azimuth_rad = np.radians(self.vr_azimuth)
                self.vr_pos[0] += np.cos(azimuth_rad) * 0.05
                self.vr_pos[1] += np.sin(azimuth_rad) * 0.05
                self._save_settings()
            elif key == 84:  # Down arrow - move backward
                azimuth_rad = np.radians(self.vr_azimuth)
                self.vr_pos[0] -= np.cos(azimuth_rad) * 0.05
                self.vr_pos[1] -= np.sin(azimuth_rad) * 0.05
                self._save_settings()
            elif key == 85:  # Page Up
                self.vr_pos[2] += 0.05
                self._save_settings()
            elif key == 86:  # Page Down
                self.vr_pos[2] -= 0.05
                self._save_settings()
            # Joint controls
            elif key == ord('q'):
                self._adjust_joint(0, self.joint_step)
            elif key == ord('a'):
                self._adjust_joint(0, -self.joint_step)
            elif key == ord('w'):
                self._adjust_joint(1, self.joint_step)
            elif key == ord('s'):
                self._adjust_joint(1, -self.joint_step)
            elif key == ord('e'):
                self._adjust_joint(2, self.joint_step)
            elif key == ord('d'):
                self._adjust_joint(2, -self.joint_step)
            elif key == ord('r'):
                self._adjust_joint(3, self.joint_step)
            elif key == ord('f'):
                self._adjust_joint(3, -self.joint_step)
            elif key == ord('t'):
                self._adjust_joint(4, self.joint_step)
            elif key == ord('g'):
                self._adjust_joint(4, -self.joint_step)
            elif key == ord('y'):
                self._adjust_joint(5, self.joint_step)
            elif key == ord('h'):
                self._adjust_joint(5, -self.joint_step)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    VRStereoViewer().run()
