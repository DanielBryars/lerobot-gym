"""
Interactive wrist camera position/orientation adjustment tool.

Use sliders to adjust the camera position and orientation in real-time.
When you're happy with the view, the script outputs the values to paste
into scenes/so101_with_wrist_cam.xml.

Usage:
    python adjust_wrist_camera.py
"""

import cv2
import numpy as np
import mujoco
from pathlib import Path


SCENE_XML = Path(__file__).parent / "scenes" / "so101_with_wrist_cam.xml"


def quat_to_euler(quat):
    """Convert quaternion [w, x, y, z] to euler angles (degrees) [roll, pitch, yaw]."""
    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees([roll, pitch, yaw])


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


class CameraAdjuster:
    def __init__(self):
        # Load model
        print(f"Loading {SCENE_XML}...")
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)

        # Get camera and body IDs
        self.wrist_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam"
        )
        print(f"Wrist camera ID: {self.wrist_cam_id}")

        # Initialize camera parameters (in gripper's local frame)
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = -0.05  # Toward gripper tips
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.fovy = 75

        # Parameter ranges
        self.pos_range = (-0.15, 0.15)
        self.fov_range = (30, 120)

        # Move arm to an interesting position
        self._set_arm_pose([0.3, -0.5, 0.8, 0.5, 0.0, 0.5])

        # Renderers
        self.external_renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.wrist_renderer = mujoco.Renderer(self.model, height=480, width=640)

        # Setup window
        self.window_name = "Wrist Camera Adjustment"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1300, 700)
        self._create_trackbars()

    def _set_arm_pose(self, ctrl):
        """Set arm to a specific pose."""
        self.data.ctrl[:] = ctrl
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

    def _create_trackbars(self):
        """Create trackbars for camera adjustment."""
        def pos_to_slider(val):
            return int((val - self.pos_range[0]) / (self.pos_range[1] - self.pos_range[0]) * 1000)

        cv2.createTrackbar("X", self.window_name, pos_to_slider(self.pos_x), 1000, lambda x: None)
        cv2.createTrackbar("Y", self.window_name, pos_to_slider(self.pos_y), 1000, lambda x: None)
        cv2.createTrackbar("Z", self.window_name, pos_to_slider(self.pos_z), 1000, lambda x: None)
        cv2.createTrackbar("Roll", self.window_name, int(self.roll + 180), 360, lambda x: None)
        cv2.createTrackbar("Pitch", self.window_name, int(self.pitch + 180), 360, lambda x: None)
        cv2.createTrackbar("Yaw", self.window_name, int(self.yaw + 180), 360, lambda x: None)
        cv2.createTrackbar("FOV", self.window_name, int(self.fovy - self.fov_range[0]),
                          self.fov_range[1] - self.fov_range[0], lambda x: None)

    def _read_trackbars(self):
        """Read current trackbar values."""
        def slider_to_pos(val):
            return self.pos_range[0] + (val / 1000) * (self.pos_range[1] - self.pos_range[0])

        self.pos_x = slider_to_pos(cv2.getTrackbarPos("X", self.window_name))
        self.pos_y = slider_to_pos(cv2.getTrackbarPos("Y", self.window_name))
        self.pos_z = slider_to_pos(cv2.getTrackbarPos("Z", self.window_name))
        self.roll = cv2.getTrackbarPos("Roll", self.window_name) - 180
        self.pitch = cv2.getTrackbarPos("Pitch", self.window_name) - 180
        self.yaw = cv2.getTrackbarPos("Yaw", self.window_name) - 180
        self.fovy = cv2.getTrackbarPos("FOV", self.window_name) + self.fov_range[0]

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
        print("\n" + "="*60)
        print("WRIST CAMERA ADJUSTMENT")
        print("="*60)
        print("Sliders: X/Y/Z position, Roll/Pitch/Yaw rotation, FOV")
        print("Keys: S=Save XML | 1-4=Arm poses | Q=Quit")
        print("="*60 + "\n")

        arm_poses = [
            [0.3, -0.5, 0.8, 0.5, 0.0, 0.5],
            [0.0, -0.3, 0.5, 0.3, 0.0, 0.8],
            [0.5, -0.7, 1.0, 0.8, 0.3, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]

        while True:
            self._read_trackbars()

            # Render both views
            external = cv2.cvtColor(self.render_external(), cv2.COLOR_RGB2BGR)
            wrist = cv2.cvtColor(self.render_wrist(), cv2.COLOR_RGB2BGR)

            # Labels
            cv2.putText(external, "External", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(wrist, "Wrist Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Info on wrist view
            info = [
                f"pos: ({self.pos_x:.3f}, {self.pos_y:.3f}, {self.pos_z:.3f})",
                f"rot: R={self.roll:.0f} P={self.pitch:.0f} Y={self.yaw:.0f}",
                f"fov: {self.fovy}",
            ]
            for i, line in enumerate(info):
                cv2.putText(wrist, line, (10, 60 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            display = np.hstack([external, wrist])
            cv2.putText(display, "S=Print XML | Q=Quit | 1-4=Poses", (10, display.shape[0]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                print("\n" + "="*60)
                print("COPY THIS INTO scenes/so101_with_wrist_cam.xml (line ~149):")
                print("="*60)
                print(self.get_xml_config())
                print("="*60 + "\n")
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                self._set_arm_pose(arm_poses[key - ord('1')])

        cv2.destroyAllWindows()


if __name__ == "__main__":
    CameraAdjuster().run()
