"""
Interactive wrist camera position/orientation adjustment tool.

Use sliders to adjust the camera position and orientation in real-time.
When you're happy with the view, the script outputs the values to paste
into scenes/so101_with_wrist_cam.xml.

Usage:
    python adjust_wrist_camera.py

Controls:
    - Sliders: Adjust camera position (X, Y, Z) and rotation (Roll, Pitch, Yaw)
    - Press 'S' to print the camera config for the XML
    - Press 'Q' or ESC to quit
"""

import cv2
import numpy as np
import mujoco
from pathlib import Path


SCENE_XML = Path(__file__).parent / "scenes" / "so101_with_wrist_cam.xml"


class CameraAdjuster:
    def __init__(self):
        # Camera parameters (position and euler angles in degrees)
        self.pos_x = 0.0
        self.pos_y = 0.03
        self.pos_z = -0.02
        self.roll = 0.0
        self.pitch = 60.0  # Tilt down
        self.yaw = 0.0
        self.fovy = 75

        # Parameter ranges
        self.pos_range = (-0.15, 0.15)
        self.fov_range = (30, 120)

        # Load model
        print(f"Loading {SCENE_XML}...")
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)

        # Get wrist camera ID
        self.wrist_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam"
        )
        print(f"Wrist camera ID: {self.wrist_cam_id}")

        # Get gripper body ID for positioning
        self.gripper_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper"
        )

        # Move arm to an interesting position
        self._set_arm_pose([0.3, -0.5, 0.8, 0.5, 0.0, 0.5])

        # Renderers
        self.external_renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.wrist_renderer = mujoco.Renderer(self.model, height=480, width=640)

        # Setup window and trackbars
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
        # Position trackbars (scaled by 1000 for precision)
        cv2.createTrackbar("X pos (mm)", self.window_name,
                          int((self.pos_x - self.pos_range[0]) * 1000 / (self.pos_range[1] - self.pos_range[0])),
                          1000, lambda x: None)
        cv2.createTrackbar("Y pos (mm)", self.window_name,
                          int((self.pos_y - self.pos_range[0]) * 1000 / (self.pos_range[1] - self.pos_range[0])),
                          1000, lambda x: None)
        cv2.createTrackbar("Z pos (mm)", self.window_name,
                          int((self.pos_z - self.pos_range[0]) * 1000 / (self.pos_range[1] - self.pos_range[0])),
                          1000, lambda x: None)

        # Rotation trackbars (in degrees, offset by 180 to allow negative)
        cv2.createTrackbar("Roll (deg)", self.window_name, int(self.roll + 180), 360, lambda x: None)
        cv2.createTrackbar("Pitch (deg)", self.window_name, int(self.pitch + 180), 360, lambda x: None)
        cv2.createTrackbar("Yaw (deg)", self.window_name, int(self.yaw + 180), 360, lambda x: None)

        # FOV trackbar
        cv2.createTrackbar("FOV (deg)", self.window_name,
                          int(self.fovy - self.fov_range[0]),
                          self.fov_range[1] - self.fov_range[0], lambda x: None)

    def _read_trackbars(self):
        """Read current trackbar values."""
        x_raw = cv2.getTrackbarPos("X pos (mm)", self.window_name)
        y_raw = cv2.getTrackbarPos("Y pos (mm)", self.window_name)
        z_raw = cv2.getTrackbarPos("Z pos (mm)", self.window_name)

        self.pos_x = self.pos_range[0] + (x_raw / 1000) * (self.pos_range[1] - self.pos_range[0])
        self.pos_y = self.pos_range[0] + (y_raw / 1000) * (self.pos_range[1] - self.pos_range[0])
        self.pos_z = self.pos_range[0] + (z_raw / 1000) * (self.pos_range[1] - self.pos_range[0])

        self.roll = cv2.getTrackbarPos("Roll (deg)", self.window_name) - 180
        self.pitch = cv2.getTrackbarPos("Pitch (deg)", self.window_name) - 180
        self.yaw = cv2.getTrackbarPos("Yaw (deg)", self.window_name) - 180

        self.fovy = cv2.getTrackbarPos("FOV (deg)", self.window_name) + self.fov_range[0]

    def euler_to_quat(self, roll, pitch, yaw):
        """Convert euler angles (degrees) to quaternion [w, x, y, z]."""
        r, p, y = np.radians([roll, pitch, yaw])
        cr, sr = np.cos(r/2), np.sin(r/2)
        cp, sp = np.cos(p/2), np.sin(p/2)
        cy, sy = np.cos(y/2), np.sin(y/2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y_q = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return [w, x, y_q, z]

    def render_wrist_camera(self):
        """Render from the actual wrist camera, updating its position/orientation."""
        # Update the camera parameters in the model directly
        quat = self.euler_to_quat(self.roll, self.pitch, self.yaw)

        # These are body-relative coordinates, MuJoCo handles the transform
        self.model.cam_pos[self.wrist_cam_id] = [self.pos_x, self.pos_y, self.pos_z]
        self.model.cam_quat[self.wrist_cam_id] = quat
        self.model.cam_fovy[self.wrist_cam_id] = self.fovy

        # Render from the actual wrist camera
        self.wrist_renderer.update_scene(self.data, camera=self.wrist_cam_id)
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
        """Get the XML camera element to paste into the scene file."""
        quat = self.euler_to_quat(self.roll, self.pitch, self.yaw)
        return f'''<camera name="wrist_cam" pos="{self.pos_x:.4f} {self.pos_y:.4f} {self.pos_z:.4f}" quat="{quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f}" fovy="{self.fovy}"/>'''

    def run(self):
        """Main loop."""
        print("\n" + "="*60)
        print("WRIST CAMERA ADJUSTMENT TOOL")
        print("="*60)
        print("Controls:")
        print("  - Sliders: Adjust position and orientation")
        print("  - S: Print XML config to paste into scene file")
        print("  - 1-4: Change arm pose")
        print("  - Q/ESC: Quit")
        print("="*60 + "\n")

        arm_poses = [
            [0.3, -0.5, 0.8, 0.5, 0.0, 0.5],   # Default reaching
            [0.0, -0.3, 0.5, 0.3, 0.0, 0.8],   # More upright
            [0.5, -0.7, 1.0, 0.8, 0.3, 0.2],   # Extended
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],    # Home
        ]

        while True:
            self._read_trackbars()

            # Render views
            external = self.render_external()
            wrist = self.render_wrist_camera()

            # Convert to BGR for OpenCV
            external_bgr = cv2.cvtColor(external, cv2.COLOR_RGB2BGR)
            wrist_bgr = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)

            # Add labels
            cv2.putText(external_bgr, "External View", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(wrist_bgr, "Wrist Camera (Adjustable)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Show current values on wrist view
            info_lines = [
                f"pos: ({self.pos_x:.3f}, {self.pos_y:.3f}, {self.pos_z:.3f})",
                f"euler: R={self.roll:.0f} P={self.pitch:.0f} Y={self.yaw:.0f}",
                f"fovy: {self.fovy}",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(wrist_bgr, line, (10, 60 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Combine views
            display = np.hstack([external_bgr, wrist_bgr])

            # Instructions
            cv2.putText(display, "S=Print XML | Q=Quit | 1-4=Arm poses",
                       (10, display.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                print("\n" + "="*60)
                print("COPY THIS INTO scenes/so101_with_wrist_cam.xml:")
                print("="*60)
                print(self.get_xml_config())
                print("="*60 + "\n")
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                pose_idx = key - ord('1')
                self._set_arm_pose(arm_poses[pose_idx])

        cv2.destroyAllWindows()


if __name__ == "__main__":
    adjuster = CameraAdjuster()
    adjuster.run()
