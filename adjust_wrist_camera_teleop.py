"""
Interactive wrist camera adjustment with arm teleoperation.

Adjust camera position/orientation while moving the arm around to see
how the camera view looks in different configurations.

Usage:
    python adjust_wrist_camera_teleop.py

Controls:
    Camera: X/Y/Z position sliders, Roll/Pitch/Yaw rotation sliders
    Arm: Joint sliders (J1-J6) or keyboard controls

    Keyboard:
        Q/A - Shoulder pan +/-
        W/S - Shoulder lift +/-
        E/D - Elbow flex +/-
        R/F - Wrist flex +/-
        T/G - Wrist roll +/-
        Y/H - Gripper open/close

        1-4 - Preset arm poses
        P   - Print XML config
        ESC - Quit
"""

import cv2
import numpy as np
import mujoco
from pathlib import Path


SCENE_XML = Path(__file__).parent / "scenes" / "so101_with_wrist_cam.xml"

# Joint limits (radians)
JOINT_LIMITS = [
    (-1.92, 1.92),   # shoulder_pan
    (-1.75, 1.75),   # shoulder_lift
    (-1.69, 1.69),   # elbow_flex
    (-1.66, 1.66),   # wrist_flex
    (-2.74, 2.84),   # wrist_roll
    (-0.17, 1.75),   # gripper
]

JOINT_NAMES = ["Shoulder Pan", "Shoulder Lift", "Elbow Flex", "Wrist Flex", "Wrist Roll", "Gripper"]


def euler_to_quat(roll, pitch, yaw):
    """Convert euler angles (degrees) to quaternion [w, x, y, z]."""
    r, p, y = np.radians([roll, pitch, yaw])
    cr, sr = np.cos(r/2), np.sin(r/2)
    cp, sp = np.cos(p/2), np.sin(p/2)
    cy, sy = np.cos(y/2), np.sin(y/2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_q = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y_q, z])


class CameraAdjusterTeleop:
    def __init__(self):
        print(f"Loading {SCENE_XML}...")
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)

        self.gripper_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper"
        )
        self.wrist_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam"
        )
        print(f"Wrist camera ID: {self.wrist_cam_id}")

        # Camera params
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = -0.05
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.fovy = 75
        self.pos_range = (-0.15, 0.15)

        # Joint positions (start at zero)
        self.joint_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        self.joint_step = 0.05  # radians per keypress

        # Apply initial joint positions
        self._apply_joints()

        # Renderers
        self.external_renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.wrist_renderer = mujoco.Renderer(self.model, height=480, width=640)

        # Window
        self.window_name = "Wrist Camera + Teleop"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1300, 800)
        self._create_trackbars()

    def _apply_joints(self):
        """Apply current joint positions to simulation."""
        self.data.ctrl[:] = self.joint_pos
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

    def _create_trackbars(self):
        """Create trackbars for camera and joints."""
        def pos_to_slider(val):
            return int((val - self.pos_range[0]) / (self.pos_range[1] - self.pos_range[0]) * 1000)

        def joint_to_slider(val, idx):
            lo, hi = JOINT_LIMITS[idx]
            return int((val - lo) / (hi - lo) * 1000)

        # Camera trackbars
        cv2.createTrackbar("Cam X", self.window_name, pos_to_slider(self.pos_x), 1000, lambda x: None)
        cv2.createTrackbar("Cam Y", self.window_name, pos_to_slider(self.pos_y), 1000, lambda x: None)
        cv2.createTrackbar("Cam Z", self.window_name, pos_to_slider(self.pos_z), 1000, lambda x: None)
        cv2.createTrackbar("Roll", self.window_name, int(self.roll + 180), 360, lambda x: None)
        cv2.createTrackbar("Pitch", self.window_name, int(self.pitch + 180), 360, lambda x: None)
        cv2.createTrackbar("Yaw", self.window_name, int(self.yaw + 180), 360, lambda x: None)

        # Joint trackbars
        for i, name in enumerate(JOINT_NAMES):
            cv2.createTrackbar(f"J{i+1}", self.window_name, joint_to_slider(self.joint_pos[i], i), 1000, lambda x: None)

    def _read_trackbars(self):
        """Read trackbar values."""
        def slider_to_pos(val):
            return self.pos_range[0] + (val / 1000) * (self.pos_range[1] - self.pos_range[0])

        def slider_to_joint(val, idx):
            lo, hi = JOINT_LIMITS[idx]
            return lo + (val / 1000) * (hi - lo)

        # Camera
        self.pos_x = slider_to_pos(cv2.getTrackbarPos("Cam X", self.window_name))
        self.pos_y = slider_to_pos(cv2.getTrackbarPos("Cam Y", self.window_name))
        self.pos_z = slider_to_pos(cv2.getTrackbarPos("Cam Z", self.window_name))
        self.roll = cv2.getTrackbarPos("Roll", self.window_name) - 180
        self.pitch = cv2.getTrackbarPos("Pitch", self.window_name) - 180
        self.yaw = cv2.getTrackbarPos("Yaw", self.window_name) - 180

        # Joints
        for i in range(6):
            self.joint_pos[i] = slider_to_joint(cv2.getTrackbarPos(f"J{i+1}", self.window_name), i)

    def _update_joint_sliders(self):
        """Update joint sliders to match current joint_pos (after keyboard input)."""
        for i in range(6):
            lo, hi = JOINT_LIMITS[i]
            slider_val = int((self.joint_pos[i] - lo) / (hi - lo) * 1000)
            cv2.setTrackbarPos(f"J{i+1}", self.window_name, slider_val)

    def _adjust_joint(self, idx, delta):
        """Adjust a joint by delta, respecting limits."""
        lo, hi = JOINT_LIMITS[idx]
        self.joint_pos[idx] = np.clip(self.joint_pos[idx] + delta, lo, hi)
        self._update_joint_sliders()

    def render_wrist(self):
        """Render from wrist camera (actual model camera attached to gripper body)."""
        # Update camera pose in the model (local to parent body = gripper)
        self.model.cam_pos[self.wrist_cam_id] = np.array([self.pos_x, self.pos_y, self.pos_z], dtype=float)
        self.model.cam_quat[self.wrist_cam_id] = euler_to_quat(self.roll, self.pitch, self.yaw)

        # Render using the named camera from the model
        self.wrist_renderer.update_scene(self.data, camera="wrist_cam")
        return self.wrist_renderer.render()

    def render_external(self):
        """Render external view."""
        cam = mujoco.MjvCamera()
        cam.distance = 0.8
        cam.azimuth = 120
        cam.elevation = -25
        self.external_renderer.update_scene(self.data, camera=cam)
        return self.external_renderer.render()

    def get_xml_config(self):
        """Get XML camera element."""
        quat = euler_to_quat(self.roll, self.pitch, self.yaw)
        return f'<camera name="wrist_cam" pos="{self.pos_x:.4f} {self.pos_y:.4f} {self.pos_z:.4f}" quat="{quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f}" fovy="{self.fovy}"/>'

    def run(self):
        print("\n" + "="*70)
        print("WRIST CAMERA ADJUSTMENT + TELEOP")
        print("="*70)
        print("Camera: Sliders Cam X/Y/Z, Roll/Pitch/Yaw")
        print("Arm:    Sliders J1-J6 or keyboard:")
        print("        Q/A=J1  W/S=J2  E/D=J3  R/F=J4  T/G=J5  Y/H=J6")
        print("        1-4=Preset poses  P=Print XML  ESC=Quit")
        print("="*70 + "\n")

        presets = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            [0.3, -0.5, 0.8, 0.5, 0.0, 0.5],
            [0.5, -0.7, 1.0, 0.8, 0.3, 0.2],
            [-0.5, -0.3, 0.5, 0.3, -0.5, 0.8],
        ]

        while True:
            self._read_trackbars()
            self._apply_joints()

            # Render
            external = cv2.cvtColor(self.render_external(), cv2.COLOR_RGB2BGR)
            wrist = cv2.cvtColor(self.render_wrist(), cv2.COLOR_RGB2BGR)

            # Labels
            cv2.putText(external, "External", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(wrist, "Wrist Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Camera info
            info = [
                f"cam: ({self.pos_x:.3f}, {self.pos_y:.3f}, {self.pos_z:.3f})",
                f"rot: R={self.roll:.0f} P={self.pitch:.0f} Y={self.yaw:.0f}",
            ]
            for i, line in enumerate(info):
                cv2.putText(wrist, line, (10, 60 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Joint info on external view
            joint_info = [f"J{i+1}: {np.degrees(self.joint_pos[i]):.1f}deg" for i in range(6)]
            for i, line in enumerate(joint_info):
                cv2.putText(external, line, (10, 60 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            display = np.hstack([external, wrist])
            cv2.putText(display, "P=Print XML | 1-4=Presets | Q/A W/S E/D R/F T/G Y/H=Joint control | ESC=Quit",
                       (10, display.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                break
            elif key == ord('p'):
                print("\n" + "="*60)
                print("COPY THIS INTO scenes/so101_with_wrist_cam.xml:")
                print("="*60)
                print(self.get_xml_config())
                print("="*60 + "\n")
            # Preset poses
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                self.joint_pos = np.array(presets[key - ord('1')])
                self._update_joint_sliders()
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
    CameraAdjusterTeleop().run()
