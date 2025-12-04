#!/usr/bin/env python
"""
Teleoperate SO101 simulation in VR using a physical leader arm.

View the simulation in your VR headset while controlling with the physical arm.
Connect Quest 3 via Quest Link, connect leader arm, then run this script.

Usage:
    python teleop_vr.py
    python teleop_vr.py --leader-port COM8 --mirror
    python teleop_vr.py --debug  # OpenXR debug output

Requirements:
    pip install pyopenxr glfw PyOpenGL numpy
    pip install -r requirements-eval.txt  # For motor control
"""
import argparse
import ctypes
import json
import platform
import time
from pathlib import Path
from typing import Optional

import glfw
import mujoco
import numpy as np
from OpenGL import GL

try:
    import xr
except ImportError:
    print("ERROR: pyopenxr not installed. Run: pip install pyopenxr")
    raise

APP_NAME = "SO101 VR Teleop"
FRUSTUM_NEAR = 0.05
FRUSTUM_FAR = 50.0

# Sim action space bounds (radians)
SIM_ACTION_LOW = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453])
SIM_ACTION_HIGH = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533])


def normalized_to_radians(normalized_values: np.ndarray) -> np.ndarray:
    """Convert from lerobot normalized values to sim radians."""
    radians = np.zeros(6, dtype=np.float32)

    # Joints 0-4: map [-100, 100] -> [low, high]
    for i in range(5):
        t = (normalized_values[i] + 100) / 200.0
        radians[i] = SIM_ACTION_LOW[i] + t * (SIM_ACTION_HIGH[i] - SIM_ACTION_LOW[i])

    # Gripper: map [0, 100] -> [low, high]
    t = normalized_values[5] / 100.0
    radians[5] = SIM_ACTION_LOW[5] + t * (SIM_ACTION_HIGH[5] - SIM_ACTION_LOW[5])

    return radians


def load_config():
    """Load config.json for COM port settings."""
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
    """Create motor bus for leader arm with STS3250 motors."""
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


def get_scene_xml_path() -> Path:
    """Get path to scene XML file."""
    module_dir = Path(__file__).parent
    paths = [
        module_dir / "assets" / "SO-ARM100" / "Simulation" / "SO101" / "scene_with_cube.xml",
        Path("assets/SO-ARM100/Simulation/SO101/scene_with_cube.xml"),
    ]
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("scene_with_cube.xml not found. Run 'python setup_scene.py' first.")


class VRTeleop:
    """VR viewer with physical leader arm teleoperation."""

    def __init__(
        self,
        xml_path: Path,
        leader_port: str,
        mirror_window: bool = False,
        debug: bool = False,
        samples: Optional[int] = 8,
        fps: int = 30
    ):
        self._xml_path = xml_path
        self._leader_port = leader_port
        self._mirror_window = mirror_window
        self._debug = debug
        self._samples = samples
        self._fps = fps
        self._should_quit = False
        self._step_count = 0

        # Initialized later
        self._bus = None
        self._xr_instance = None
        self._xr_session = None
        self._window = None
        self._mj_model = None
        self._mj_data = None

    def __enter__(self):
        self._init_leader()
        self._init_xr()
        self._init_window()
        self._prepare_xr()
        self._prepare_mujoco()
        glfw.make_context_current(None)
        return self

    def _init_leader(self):
        """Connect to physical leader arm."""
        print(f"Connecting to leader arm on {self._leader_port}...")
        self._bus = create_leader_bus(self._leader_port)
        self._bus.connect()

        # Read calibration from EEPROM
        print("Reading calibration from motor EEPROM...")
        self._bus.calibration = self._bus.read_calibration()
        self._bus.disable_torque()
        print("Leader arm connected!")

    def _init_xr(self):
        """Initialize OpenXR."""
        extensions = [xr.KHR_OPENGL_ENABLE_EXTENSION_NAME]

        instance_create_info = xr.InstanceCreateInfo(
            application_info=xr.ApplicationInfo(
                application_name=APP_NAME,
                engine_name="pyopenxr",
                engine_version=xr.PYOPENXR_CURRENT_API_VERSION,
                api_version=xr.Version(1, 0, xr.XR_VERSION_PATCH)
            )
        )

        if self._debug:
            def debug_callback_py(severity, _type, data, _user_data):
                print(f"[XR] {data.contents.message.decode()}")
                return True

            debug_messenger = xr.DebugUtilsMessengerCreateInfoEXT(
                message_severities=(
                    xr.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                    | xr.DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
                ),
                message_types=xr.DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT,
                user_callback=xr.PFN_xrDebugUtilsMessengerCallbackEXT(debug_callback_py)
            )
            instance_create_info.next = ctypes.cast(ctypes.pointer(debug_messenger), ctypes.c_void_p)
            extensions.append(xr.EXT_DEBUG_UTILS_EXTENSION_NAME)

        instance_create_info.enabled_extension_names = extensions
        self._xr_instance = xr.create_instance(instance_create_info)
        self._xr_system = xr.get_system(
            self._xr_instance,
            xr.SystemGetInfo(xr.FormFactor.HEAD_MOUNTED_DISPLAY)
        )

        views_config = xr.enumerate_view_configuration_views(
            self._xr_instance,
            self._xr_system,
            xr.ViewConfigurationType.PRIMARY_STEREO
        )

        self._width = views_config[0].recommended_image_rect_width
        self._height = views_config[0].recommended_image_rect_height
        self._width_render = self._width * 2

        print(f"VR: {self._width}x{self._height} per eye")

        # OpenGL requirements check
        pxrGetOpenGLGraphicsRequirementsKHR = ctypes.cast(
            xr.get_instance_proc_addr(self._xr_instance, "xrGetOpenGLGraphicsRequirementsKHR"),
            xr.PFN_xrGetOpenGLGraphicsRequirementsKHR
        )
        pxrGetOpenGLGraphicsRequirementsKHR(
            self._xr_instance, self._xr_system,
            ctypes.byref(xr.GraphicsRequirementsOpenGLKHR())
        )

    def _init_window(self):
        """Create GLFW window."""
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        glfw.window_hint(glfw.DOUBLEBUFFER, False)
        glfw.window_hint(glfw.RESIZABLE, False)
        if not self._mirror_window:
            glfw.window_hint(glfw.VISIBLE, False)

        self._window_size = [self._width // 2, self._height // 2]
        self._window = glfw.create_window(*self._window_size, APP_NAME, None, None)
        glfw.make_context_current(self._window)
        glfw.swap_interval(0)

    def _prepare_xr(self):
        """Set up XR session."""
        if platform.system() == 'Windows':
            from OpenGL import WGL
            graphics_binding = xr.GraphicsBindingOpenGLWin32KHR()
            graphics_binding.h_dc = WGL.wglGetCurrentDC()
            graphics_binding.h_glrc = WGL.wglGetCurrentContext()
        else:
            from OpenGL import GLX
            graphics_binding = xr.GraphicsBindingOpenGLXlibKHR()
            graphics_binding.x_display = GLX.glXGetCurrentDisplay()
            graphics_binding.glx_context = GLX.glXGetCurrentContext()
            graphics_binding.glx_drawable = GLX.glXGetCurrentDrawable()

        self._xr_session = xr.create_session(
            self._xr_instance,
            xr.SessionCreateInfo(
                0, self._xr_system,
                next=ctypes.cast(ctypes.pointer(graphics_binding), ctypes.c_void_p)
            )
        )
        self._xr_session_state = xr.SessionState.IDLE

        self._xr_swapchain = xr.create_swapchain(
            self._xr_session,
            xr.SwapchainCreateInfo(
                usage_flags=(
                    xr.SWAPCHAIN_USAGE_TRANSFER_DST_BIT
                    | xr.SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT
                ),
                format=GL.GL_RGBA8,
                sample_count=1 if self._samples is None else self._samples,
                width=self._width_render,
                height=self._height,
                array_size=1, face_count=1, mip_count=1
            )
        )
        self._xr_swapchain_images = xr.enumerate_swapchain_images(
            self._xr_swapchain, xr.SwapchainImageOpenGLKHR
        )

        self._xr_projection_layer = xr.CompositionLayerProjection(
            space=xr.create_reference_space(self._xr_session, xr.ReferenceSpaceCreateInfo()),
            views=[
                xr.CompositionLayerProjectionView(
                    sub_image=xr.SwapchainSubImage(
                        swapchain=self._xr_swapchain,
                        image_rect=xr.Rect2Di(
                            extent=xr.Extent2Di(self._width, self._height),
                            offset=None if i == 0 else xr.Offset2Di(x=self._width)
                        )
                    )
                ) for i in range(2)
            ]
        )

        self._xr_swapchain_fbo = GL.glGenFramebuffers(1)

    def _prepare_mujoco(self):
        """Initialize MuJoCo."""
        print(f"Loading: {self._xml_path}")
        self._mj_model = mujoco.MjModel.from_xml_path(str(self._xml_path))
        self._mj_data = mujoco.MjData(self._mj_model)
        self._mj_scene = mujoco.MjvScene(self._mj_model, 1000)
        self._mj_scene.stereo = mujoco.mjtStereo.mjSTEREO_SIDEBYSIDE

        self._mj_model.vis.global_.offwidth = self._width_render
        self._mj_model.vis.global_.offheight = self._height
        self._mj_model.vis.quality.offsamples = 0 if self._samples is None else self._samples

        self._mj_context = mujoco.MjrContext(self._mj_model, mujoco.mjtFontScale.mjFONTSCALE_100)
        self._mj_camera = mujoco.MjvCamera()
        self._mj_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self._mj_option)

    def _read_leader_positions(self) -> np.ndarray:
        """Read current positions from leader arm."""
        positions = self._bus.sync_read("Present_Position")
        return np.array([
            positions["shoulder_pan"],
            positions["shoulder_lift"],
            positions["elbow_flex"],
            positions["wrist_flex"],
            positions["wrist_roll"],
            positions["gripper"],
        ], dtype=np.float32)

    def _poll_xr_events(self):
        """Handle XR events."""
        while True:
            try:
                event_buffer = xr.poll_event(self._xr_instance)
                if xr.StructureType(event_buffer.type) == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                    event = ctypes.cast(
                        ctypes.byref(event_buffer),
                        ctypes.POINTER(xr.EventDataSessionStateChanged)
                    ).contents
                    self._xr_session_state = xr.SessionState(event.state)

                    if self._xr_session_state == xr.SessionState.READY:
                        xr.begin_session(self._xr_session, xr.SessionBeginInfo(xr.ViewConfigurationType.PRIMARY_STEREO))
                    elif self._xr_session_state == xr.SessionState.STOPPING:
                        xr.end_session(self._xr_session)
                    elif self._xr_session_state in [xr.SessionState.EXITING, xr.SessionState.LOSS_PENDING]:
                        self._should_quit = True
            except xr.EventUnavailable:
                break

    def _update_views(self):
        """Update stereo views from HMD tracking."""
        _, view_states = xr.locate_views(
            self._xr_session,
            xr.ViewLocateInfo(
                xr.ViewConfigurationType.PRIMARY_STEREO,
                self._xr_frame_state.predicted_display_time,
                self._xr_projection_layer.space,
            )
        )

        for eye_index, view_state in enumerate(view_states):
            self._xr_projection_layer.views[eye_index].fov = view_state.fov
            self._xr_projection_layer.views[eye_index].pose = view_state.pose

            cam = self._mj_scene.camera[eye_index]
            cam.pos = list(view_state.pose.position)
            cam.frustum_near = FRUSTUM_NEAR
            cam.frustum_far = FRUSTUM_FAR
            cam.frustum_bottom = np.tan(view_state.fov.angle_down) * FRUSTUM_NEAR
            cam.frustum_top = np.tan(view_state.fov.angle_up) * FRUSTUM_NEAR
            cam.frustum_center = 0.5 * (
                np.tan(view_state.fov.angle_left) + np.tan(view_state.fov.angle_right)
            ) * FRUSTUM_NEAR

            rot_quat = list(view_state.pose.orientation)
            rot_quat = [rot_quat[3], *rot_quat[0:3]]

            forward, up = np.zeros(3), np.zeros(3)
            mujoco.mju_rotVecQuat(forward, [0, 0, -1], rot_quat)
            mujoco.mju_rotVecQuat(up, [0, 1, 0], rot_quat)
            cam.forward, cam.up = forward.tolist(), up.tolist()

        self._mj_scene.enabletransform = True
        self._mj_scene.rotate[0] = np.cos(0.25 * np.pi)
        self._mj_scene.rotate[1] = np.sin(-0.25 * np.pi)

    def _render(self):
        """Render to VR headset."""
        image_index = xr.acquire_swapchain_image(self._xr_swapchain, xr.SwapchainImageAcquireInfo())
        xr.wait_swapchain_image(self._xr_swapchain, xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION))

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._xr_swapchain_fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D if self._samples is None else GL.GL_TEXTURE_2D_MULTISAMPLE,
            self._xr_swapchain_images[image_index].image, 0
        )

        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._mj_context)
        mujoco.mjr_render(
            mujoco.MjrRect(0, 0, self._width_render, self._height),
            self._mj_scene, self._mj_context
        )

        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self._mj_context.offFBO)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._xr_swapchain_fbo)
        GL.glBlitFramebuffer(
            0, 0, self._width_render, self._height,
            0, 0, self._width_render, self._height,
            GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST
        )

        if self._mirror_window:
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)
            GL.glBlitFramebuffer(
                0, 0, self._width, self._height,
                0, 0, *self._window_size,
                GL.GL_COLOR_BUFFER_BIT, 0x90BA
            )

        xr.release_swapchain_image(self._xr_swapchain, xr.SwapchainImageReleaseInfo())

    def frame(self):
        """Process one frame with teleoperation."""
        glfw.poll_events()
        self._poll_xr_events()

        if glfw.window_should_close(self._window):
            self._should_quit = True

        if self._should_quit:
            return

        # Check XR frame readiness
        if self._xr_session_state not in [
            xr.SessionState.READY, xr.SessionState.FOCUSED,
            xr.SessionState.SYNCHRONIZED, xr.SessionState.VISIBLE
        ]:
            return

        self._xr_frame_state = xr.wait_frame(self._xr_session, xr.FrameWaitInfo())

        # Read leader arm and apply to simulation
        normalized = self._read_leader_positions()
        joint_radians = normalized_to_radians(normalized)
        joint_radians = np.clip(joint_radians, SIM_ACTION_LOW, SIM_ACTION_HIGH)

        # Set joint positions and step physics
        self._mj_data.ctrl[:] = joint_radians
        mujoco.mj_step(self._mj_model, self._mj_data)
        mujoco.mjv_updateScene(
            self._mj_model, self._mj_data, self._mj_option,
            None, self._mj_camera, mujoco.mjtCatBit.mjCAT_ALL, self._mj_scene
        )
        self._step_count += 1

        # Update VR views and render
        self._update_views()

        xr.begin_frame(self._xr_session, None)
        if self._xr_frame_state.should_render:
            self._render()
        xr.end_frame(
            self._xr_session,
            xr.FrameEndInfo(
                self._xr_frame_state.predicted_display_time,
                xr.EnvironmentBlendMode.OPAQUE,
                layers=[ctypes.byref(self._xr_projection_layer)] if self._xr_frame_state.should_render else []
            )
        )

    def loop(self):
        """Main loop."""
        glfw.make_context_current(self._window)
        frame_time = 1.0 / self._fps

        print("\n" + "="*50)
        print("VR TELEOPERATION STARTED")
        print("="*50)
        print("Put on your headset to view the simulation")
        print("Move your leader arm to control the robot")
        print("Press Ctrl+C or close window to exit")
        print("="*50 + "\n")

        while not self._should_quit:
            loop_start = time.time()
            self.frame()

            # Print status periodically
            if self._step_count % 100 == 0:
                print(f"Step: {self._step_count}")

            # Rate limiting
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    def __exit__(self, exc_type, exc_value, traceback):
        """Cleanup."""
        print(f"\nTeleop ended after {self._step_count} steps")

        if self._bus is not None:
            self._bus.disconnect()

        if self._window is not None:
            glfw.make_context_current(self._window)
            if self._xr_swapchain_fbo is not None:
                GL.glDeleteFramebuffers(1, [self._xr_swapchain_fbo])

        if self._xr_swapchain is not None:
            xr.destroy_swapchain(self._xr_swapchain)
        if self._xr_session is not None:
            xr.destroy_session(self._xr_session)

        glfw.terminate()


def main():
    parser = argparse.ArgumentParser(description="VR teleoperation with leader arm")
    parser.add_argument("--leader-port", "-p", type=str, default=None,
                        help="Serial port for leader arm")
    parser.add_argument("--fps", "-f", type=int, default=30,
                        help="Target frame rate")
    parser.add_argument("--mirror", "-m", action="store_true",
                        help="Show desktop mirror")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="OpenXR debug output")

    args = parser.parse_args()

    # Get leader port
    leader_port = args.leader_port
    if leader_port is None:
        config = load_config()
        if config and "leader" in config:
            leader_port = config["leader"]["port"]
            print(f"Using leader port from config: {leader_port}")
        else:
            leader_port = "COM8"
            print(f"Using default port: {leader_port}")

    xml_path = get_scene_xml_path()

    print("\nStarting VR Teleoperation...")
    print("Ensure Quest Link is connected and leader arm is powered on")

    try:
        with VRTeleop(
            xml_path=xml_path,
            leader_port=leader_port,
            mirror_window=args.mirror,
            debug=args.debug,
            fps=args.fps
        ) as teleop:
            teleop.loop()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Quest Link connected?")
        print("2. OpenXR runtime set to Oculus?")
        print("3. Leader arm connected and powered?")
        print("4. Run: pip install pyopenxr glfw PyOpenGL")
        raise


if __name__ == "__main__":
    main()
