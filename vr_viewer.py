#!/usr/bin/env python
"""
VR Viewer for SO101 Simulation using OpenXR.

View the SO101 robot simulation in VR using Meta Quest 3 (or other OpenXR headset).
Connect Quest 3 via Quest Link, then run this script.

Based on: https://gist.github.com/SkytAsul/b1a48a31c4f86b65d72bc8edcb122d3f

Usage:
    python vr_viewer.py
    python vr_viewer.py --mirror  # Show desktop mirror window
    python vr_viewer.py --debug   # Enable OpenXR debug output

Requirements:
    pip install pyopenxr glfw PyOpenGL numpy
"""
import argparse
import ctypes
import platform
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
    print("Also ensure Quest Link is running and OpenXR runtime is set to Oculus.")
    raise

APP_NAME = "SO101 VR Viewer"
FRUSTUM_NEAR = 0.05
FRUSTUM_FAR = 50.0


class MujocoXRViewer:
    """
    OpenXR-based VR viewer for MuJoCo simulations.

    Renders MuJoCo scene to VR headset with stereoscopic view.
    """

    def __init__(
        self,
        xml_path: Path,
        mirror_window: bool = False,
        debug: bool = False,
        samples: Optional[int] = 8
    ):
        self._xml_path = xml_path
        self._mirror_window = mirror_window
        self._debug = debug
        self._samples = samples
        self._should_quit = False

        # Will be initialized in __enter__
        self._xr_instance = None
        self._xr_system = None
        self._xr_session = None
        self._xr_swapchain = None
        self._xr_swapchain_fbo = None
        self._window = None
        self._mj_model = None
        self._mj_data = None

    def __enter__(self):
        self._init_xr()
        self._init_window()
        self._prepare_xr()
        self._prepare_mujoco()
        glfw.make_context_current(None)
        return self

    def _init_xr(self):
        """Initialize OpenXR instance and system."""
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
                print(f"[OpenXR {severity}] {data.contents.function_name.decode()}: {data.contents.message.decode()}")
                return True

            debug_messenger = xr.DebugUtilsMessengerCreateInfoEXT(
                message_severities=(
                    xr.DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
                    | xr.DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT
                    | xr.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                    | xr.DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
                ),
                message_types=(
                    xr.DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                    | xr.DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                    | xr.DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
                    | xr.DEBUG_UTILS_MESSAGE_TYPE_CONFORMANCE_BIT_EXT
                ),
                user_callback=xr.PFN_xrDebugUtilsMessengerCallbackEXT(debug_callback_py)
            )
            instance_create_info.next = ctypes.cast(ctypes.pointer(debug_messenger), ctypes.c_void_p)
            extensions.append(xr.EXT_DEBUG_UTILS_EXTENSION_NAME)

        instance_create_info.enabled_extension_names = extensions
        self._xr_instance = xr.create_instance(instance_create_info)

        # Get system (HMD)
        self._xr_system = xr.get_system(
            self._xr_instance,
            xr.SystemGetInfo(xr.FormFactor.HEAD_MOUNTED_DISPLAY)
        )

        # Verify stereo view configuration
        view_configs = xr.enumerate_view_configurations(self._xr_instance, self._xr_system)
        assert xr.ViewConfigurationType.PRIMARY_STEREO in view_configs, "Stereo not supported"

        views_config = xr.enumerate_view_configuration_views(
            self._xr_instance,
            self._xr_system,
            xr.ViewConfigurationType.PRIMARY_STEREO
        )
        assert len(views_config) == 2, "Expected 2 views for stereo"

        self._width = views_config[0].recommended_image_rect_width
        self._height = views_config[0].recommended_image_rect_height
        self._width_render = self._width * 2  # Side-by-side stereo

        print(f"VR resolution: {self._width}x{self._height} per eye")

        # Get OpenGL requirements
        pxrGetOpenGLGraphicsRequirementsKHR = ctypes.cast(
            xr.get_instance_proc_addr(self._xr_instance, "xrGetOpenGLGraphicsRequirementsKHR"),
            xr.PFN_xrGetOpenGLGraphicsRequirementsKHR
        )
        graphics_result = pxrGetOpenGLGraphicsRequirementsKHR(
            self._xr_instance,
            self._xr_system,
            ctypes.byref(xr.GraphicsRequirementsOpenGLKHR())
        )
        result = xr.exception.check_result(xr.Result(graphics_result))
        if result.is_exception():
            raise result

    def _init_window(self):
        """Initialize GLFW window for OpenGL context."""
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")

        glfw.window_hint(glfw.DOUBLEBUFFER, False)
        glfw.window_hint(glfw.RESIZABLE, False)
        glfw.window_hint(glfw.SAMPLES, 0)

        if not self._mirror_window:
            glfw.window_hint(glfw.VISIBLE, False)

        self._window_size = [self._width // 2, self._height // 2]
        self._window = glfw.create_window(
            *self._window_size,
            APP_NAME,
            None, None
        )

        if self._window is None:
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self._window)
        glfw.swap_interval(0)

    def _prepare_xr(self):
        """Set up OpenXR session and swapchain."""
        # Platform-specific graphics binding
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

        # Create session
        self._xr_session = xr.create_session(
            self._xr_instance,
            xr.SessionCreateInfo(
                0,
                self._xr_system,
                next=ctypes.cast(ctypes.pointer(graphics_binding), ctypes.c_void_p)
            )
        )
        self._xr_session_state = xr.SessionState.IDLE

        # Create swapchain
        self._xr_swapchain = xr.create_swapchain(
            self._xr_session,
            xr.SwapchainCreateInfo(
                usage_flags=(
                    xr.SWAPCHAIN_USAGE_TRANSFER_DST_BIT
                    | xr.SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT
                    | xr.SWAPCHAIN_USAGE_SAMPLED_BIT
                ),
                format=GL.GL_RGBA8,
                sample_count=1 if self._samples is None else self._samples,
                array_size=1,
                face_count=1,
                mip_count=1,
                width=self._width_render,
                height=self._height
            )
        )
        self._xr_swapchain_images = xr.enumerate_swapchain_images(
            self._xr_swapchain,
            xr.SwapchainImageOpenGLKHR
        )

        # Create projection layer
        self._xr_projection_layer = xr.CompositionLayerProjection(
            space=xr.create_reference_space(
                self._xr_session,
                xr.ReferenceSpaceCreateInfo()
            ),
            views=[
                xr.CompositionLayerProjectionView(
                    sub_image=xr.SwapchainSubImage(
                        swapchain=self._xr_swapchain,
                        image_rect=xr.Rect2Di(
                            extent=xr.Extent2Di(self._width, self._height),
                            offset=None if eye_index == 0 else xr.Offset2Di(x=self._width)
                        )
                    )
                )
                for eye_index in range(2)
            ]
        )

        self._xr_swapchain_fbo = GL.glGenFramebuffers(1)

    def _prepare_mujoco(self):
        """Initialize MuJoCo model and rendering."""
        print(f"Loading model: {self._xml_path}")
        self._mj_model = mujoco.MjModel.from_xml_path(str(self._xml_path))
        self._mj_data = mujoco.MjData(self._mj_model)
        self._mj_scene = mujoco.MjvScene(self._mj_model, 1000)
        self._mj_scene.stereo = mujoco.mjtStereo.mjSTEREO_SIDEBYSIDE

        # Configure offscreen rendering
        self._mj_model.vis.global_.offwidth = self._width_render
        self._mj_model.vis.global_.offheight = self._height
        self._mj_model.vis.quality.offsamples = 0 if self._samples is None else self._samples

        self._mj_context = mujoco.MjrContext(
            self._mj_model,
            mujoco.mjtFontScale.mjFONTSCALE_100
        )
        self._mj_camera = mujoco.MjvCamera()
        self._mj_option = mujoco.MjvOption()

        mujoco.mjv_defaultOption(self._mj_option)

    def _update_mujoco(self):
        """Step physics and update scene."""
        mujoco.mj_step(self._mj_model, self._mj_data)
        mujoco.mjv_updateScene(
            self._mj_model,
            self._mj_data,
            self._mj_option,
            None,
            self._mj_camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self._mj_scene
        )

    def set_joint_positions(self, positions: np.ndarray):
        """Set robot joint positions (6 DOF)."""
        if len(positions) != 6:
            raise ValueError(f"Expected 6 joint positions, got {len(positions)}")
        self._mj_data.ctrl[:] = positions

    def _wait_xr_frame(self) -> bool:
        """Wait for XR frame."""
        if self._xr_session_state in [
            xr.SessionState.READY,
            xr.SessionState.FOCUSED,
            xr.SessionState.SYNCHRONIZED,
            xr.SessionState.VISIBLE,
        ]:
            self._xr_frame_state = xr.wait_frame(self._xr_session, xr.FrameWaitInfo())
            return True
        return False

    def _end_xr_frame(self):
        """End XR frame."""
        xr.end_frame(
            self._xr_session,
            xr.FrameEndInfo(
                self._xr_frame_state.predicted_display_time,
                xr.EnvironmentBlendMode.OPAQUE,
                layers=[ctypes.byref(self._xr_projection_layer)] if self._xr_frame_state.should_render else []
            )
        )

    def _poll_xr_events(self):
        """Poll and handle OpenXR events."""
        while True:
            try:
                event_buffer = xr.poll_event(self._xr_instance)
                event_type = xr.StructureType(event_buffer.type)

                if event_type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                    event = ctypes.cast(
                        ctypes.byref(event_buffer),
                        ctypes.POINTER(xr.EventDataSessionStateChanged)
                    ).contents
                    self._xr_session_state = xr.SessionState(event.state)

                    match self._xr_session_state:
                        case xr.SessionState.READY:
                            if not self._should_quit:
                                xr.begin_session(
                                    self._xr_session,
                                    xr.SessionBeginInfo(xr.ViewConfigurationType.PRIMARY_STEREO)
                                )
                        case xr.SessionState.STOPPING:
                            xr.end_session(self._xr_session)
                        case xr.SessionState.EXITING | xr.SessionState.LOSS_PENDING:
                            self._should_quit = True
            except xr.EventUnavailable:
                break

    def _update_views(self):
        """Update stereo camera views from HMD tracking."""
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

            # Convert quaternion: OpenXR (x,y,z,w) -> MuJoCo (w,x,y,z)
            rot_quat = list(view_state.pose.orientation)
            rot_quat = [rot_quat[3], *rot_quat[0:3]]

            forward = np.zeros(3)
            up = np.zeros(3)
            mujoco.mju_rotVecQuat(forward, [0, 0, -1], rot_quat)
            mujoco.mju_rotVecQuat(up, [0, 1, 0], rot_quat)
            cam.forward = forward.tolist()
            cam.up = up.tolist()

        # Apply coordinate transform for VR viewing
        self._mj_scene.enabletransform = True
        self._mj_scene.rotate[0] = np.cos(0.25 * np.pi)
        self._mj_scene.rotate[1] = np.sin(-0.25 * np.pi)

    def _render(self):
        """Render frame to VR headset."""
        image_index = xr.acquire_swapchain_image(
            self._xr_swapchain,
            xr.SwapchainImageAcquireInfo()
        )
        xr.wait_swapchain_image(
            self._xr_swapchain,
            xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION)
        )

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._xr_swapchain_fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D if self._samples is None else GL.GL_TEXTURE_2D_MULTISAMPLE,
            self._xr_swapchain_images[image_index].image,
            0
        )

        # Render MuJoCo scene
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._mj_context)
        mujoco.mjr_render(
            mujoco.MjrRect(0, 0, self._width_render, self._height),
            self._mj_scene,
            self._mj_context
        )

        # Blit to swapchain
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self._mj_context.offFBO)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._xr_swapchain_fbo)
        GL.glBlitFramebuffer(
            0, 0, self._width_render, self._height,
            0, 0, self._width_render, self._height,
            GL.GL_COLOR_BUFFER_BIT,
            GL.GL_NEAREST
        )

        # Optional desktop mirror
        if self._mirror_window:
            if self._samples is not None:
                GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._mj_context.offFBO_r)
                GL.glBlitFramebuffer(
                    0, 0, self._width_render, self._height,
                    0, 0, self._width_render, self._height,
                    GL.GL_COLOR_BUFFER_BIT,
                    GL.GL_NEAREST
                )
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self._mj_context.offFBO_r)

            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)
            GL.glBlitFramebuffer(
                0, 0, self._width, self._height,
                0, 0, *self._window_size,
                GL.GL_COLOR_BUFFER_BIT,
                0x90BA  # GL_SCALED_RESOLVE_FASTEST_EXT
            )

        xr.release_swapchain_image(self._xr_swapchain, xr.SwapchainImageReleaseInfo())

    def __exit__(self, exc_type, exc_value, traceback):
        """Clean up resources."""
        if self._window is not None:
            glfw.make_context_current(self._window)
            if self._xr_swapchain_fbo is not None:
                GL.glDeleteFramebuffers(1, [self._xr_swapchain_fbo])

        if self._xr_swapchain is not None:
            xr.destroy_swapchain(self._xr_swapchain)
        if self._xr_session is not None:
            xr.destroy_session(self._xr_session)

        glfw.terminate()

    def frame(self):
        """Process one frame."""
        glfw.poll_events()
        self._poll_xr_events()

        if glfw.window_should_close(self._window):
            self._should_quit = True

        if self._should_quit:
            return

        if self._wait_xr_frame():
            self._update_mujoco()
            self._update_views()

            xr.begin_frame(self._xr_session, None)
            if self._xr_frame_state.should_render:
                self._render()
            self._end_xr_frame()

    def loop(self):
        """Main rendering loop."""
        glfw.make_context_current(self._window)
        print("\nVR session started!")
        print("Put on your headset to view the simulation")
        print("Press Ctrl+C or close window to exit\n")

        while not self._should_quit:
            self.frame()

    @property
    def should_quit(self) -> bool:
        return self._should_quit


def get_scene_xml_path() -> Path:
    """Get path to scene XML file."""
    # Try relative to this file first
    module_dir = Path(__file__).parent
    paths = [
        module_dir / "assets" / "SO-ARM100" / "Simulation" / "SO101" / "scene_with_cube.xml",
        Path("assets/SO-ARM100/Simulation/SO101/scene_with_cube.xml"),
    ]

    for path in paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "scene_with_cube.xml not found. Run 'python setup_scene.py' first."
    )


def main():
    parser = argparse.ArgumentParser(
        description="View SO101 simulation in VR"
    )
    parser.add_argument(
        "--mirror", "-m",
        action="store_true",
        help="Show desktop mirror window"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable OpenXR debug output"
    )
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=8,
        help="MSAA samples (default: 8)"
    )

    args = parser.parse_args()

    xml_path = get_scene_xml_path()
    print(f"Using scene: {xml_path}")

    print("\nStarting VR viewer...")
    print("Make sure Quest Link is connected and OpenXR runtime is set to Oculus")

    try:
        with MujocoXRViewer(
            xml_path=xml_path,
            mirror_window=args.mirror,
            debug=args.debug,
            samples=args.samples
        ) as viewer:
            viewer.loop()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Quest Link is running and headset is connected")
        print("2. Set OpenXR runtime to Oculus in Windows Settings > OpenXR")
        print("3. Try: pip install pyopenxr glfw PyOpenGL")
        raise


if __name__ == "__main__":
    main()
