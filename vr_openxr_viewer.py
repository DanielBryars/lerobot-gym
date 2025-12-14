"""
MuJoCo VR Viewer using OpenXR.

Proper head-tracked VR rendering with Quest Link.
Uses pyopenxr for OpenXR bindings and MuJoCo for physics/rendering.

Requirements:
    pip install pyopenxr glfw PyOpenGL numpy mujoco

Setup:
    1. Connect Quest 3 via Quest Link (USB or Air Link)
    2. Run this script
    3. Put on headset - you should see the simulation in VR

Based on pyopenxr examples by Christopher Bruns.
"""

import ctypes
import numpy as np
import mujoco
from pathlib import Path

try:
    import xr
    import xr.exception
except ImportError:
    print("ERROR: pyopenxr not installed. Run: pip install pyopenxr")
    exit(1)

try:
    import glfw
    from OpenGL import GL
except ImportError:
    print("ERROR: OpenGL dependencies missing. Run: pip install glfw PyOpenGL")
    exit(1)


SCENE_XML = Path(__file__).parent / "scenes" / "so101_with_wrist_cam.xml"

# Eye separation (IPD will come from OpenXR runtime)
DEFAULT_IPD = 0.063


class MuJoCoVRViewer:
    def __init__(self):
        print(f"Loading MuJoCo scene: {SCENE_XML}")
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)

        # Joint positions
        self.joint_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        self.data.ctrl[:] = self.joint_pos

        # Step simulation to initialize
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        # MuJoCo rendering objects (will be created after OpenGL context)
        self.mj_scene = None
        self.mj_context = None
        self.mj_camera = mujoco.MjvCamera()
        self.mj_option = mujoco.MjvOption()

        # Default camera setup
        self.mj_camera.lookat[:] = [0.1, 0.0, 0.1]
        self.mj_camera.distance = 0.6
        self.mj_camera.azimuth = -120
        self.mj_camera.elevation = -20

        # OpenXR state
        self.instance = None
        self.system_id = None
        self.session = None
        self.swapchains = []
        self.swapchain_images = []
        self.frame_buffers = []
        self.projection_views = []
        self.running = True

    def init_openxr(self):
        """Initialize OpenXR instance, system, and session."""
        print("Initializing OpenXR...")

        # Request OpenGL extension
        extensions = [xr.KHR_OPENGL_ENABLE_EXTENSION_NAME]

        # Create instance (pyopenxr uses simplified API)
        create_info = xr.InstanceCreateInfo(
            enabled_extension_names=extensions,
        )

        self.instance = xr.create_instance(create_info)
        instance_props = xr.get_instance_properties(self.instance)
        print(f"OpenXR Runtime: {instance_props.runtime_name}")

        # Get system (HMD)
        system_get_info = xr.SystemGetInfo(
            form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY,
        )
        self.system_id = xr.get_system(self.instance, system_get_info)

        # Get view configuration
        view_configs = xr.enumerate_view_configurations(self.instance, self.system_id)
        print(f"View configurations: {view_configs}")

        self.view_config_type = xr.ViewConfigurationType.PRIMARY_STEREO
        view_config_views = xr.enumerate_view_configuration_views(
            self.instance, self.system_id, self.view_config_type
        )
        print(f"View config views: {len(view_config_views)}")
        for i, view in enumerate(view_config_views):
            print(f"  Eye {i}: {view.recommended_image_rect_width}x{view.recommended_image_rect_height}")

        self.view_config_views = view_config_views

    def init_glfw(self):
        """Initialize GLFW window for OpenGL context."""
        print("Initializing GLFW...")

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # OpenGL 3.3 Compatibility Profile (MuJoCo needs compatibility)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)
        glfw.window_hint(glfw.VISIBLE, True)  # Visible window for MuJoCo

        self.window = glfw.create_window(800, 600, "MuJoCo VR", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        print(f"OpenGL Version: {GL.glGetString(GL.GL_VERSION).decode()}")

    def init_session(self):
        """Create OpenXR session with OpenGL graphics binding."""
        print("Creating OpenXR session...")

        # Get OpenGL requirements
        pfn_get_opengl_graphics_requirements = ctypes.cast(
            xr.get_instance_proc_addr(
                self.instance,
                "xrGetOpenGLGraphicsRequirementsKHR"
            ),
            xr.PFN_xrGetOpenGLGraphicsRequirementsKHR
        )

        graphics_requirements = xr.GraphicsRequirementsOpenGLKHR()
        result = pfn_get_opengl_graphics_requirements(
            self.instance,
            self.system_id,
            ctypes.byref(graphics_requirements)
        )

        # Create graphics binding (Windows-specific)
        import platform
        if platform.system() == "Windows":
            from OpenGL import WGL
            graphics_binding = xr.GraphicsBindingOpenGLWin32KHR(
                h_dc=WGL.wglGetCurrentDC(),
                h_glrc=WGL.wglGetCurrentContext(),
            )
        else:
            raise RuntimeError("Only Windows is currently supported")

        # Create session
        session_create_info = xr.SessionCreateInfo(
            system_id=self.system_id,
            next=ctypes.cast(ctypes.pointer(graphics_binding), ctypes.c_void_p),
        )

        self.session = xr.create_session(self.instance, session_create_info)
        print("Session created!")

    def init_swapchains(self):
        """Create swapchains for each eye."""
        print("Creating swapchains...")

        # Query swapchain formats
        swapchain_formats = xr.enumerate_swapchain_formats(self.session)

        # Prefer SRGB format
        chosen_format = GL.GL_SRGB8_ALPHA8
        if chosen_format not in swapchain_formats:
            chosen_format = swapchain_formats[0]

        self.swapchains = []
        self.swapchain_images = []

        for i, view_config in enumerate(self.view_config_views):
            width = view_config.recommended_image_rect_width
            height = view_config.recommended_image_rect_height

            swapchain_create_info = xr.SwapchainCreateInfo(
                usage_flags=xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT | xr.SwapchainUsageFlags.SAMPLED_BIT,
                format=chosen_format,
                sample_count=1,
                width=width,
                height=height,
                face_count=1,
                array_size=1,
                mip_count=1,
            )

            swapchain = xr.create_swapchain(self.session, swapchain_create_info)
            self.swapchains.append(swapchain)

            # Get swapchain images
            images = xr.enumerate_swapchain_images(swapchain, xr.SwapchainImageOpenGLKHR)
            self.swapchain_images.append(images)

            print(f"  Eye {i}: {width}x{height}, {len(images)} images")

        # Create framebuffers
        self.create_framebuffers()

    def create_framebuffers(self):
        """Create OpenGL framebuffers for rendering."""
        self.frame_buffers = []

        for eye_idx, (swapchain, images) in enumerate(zip(self.swapchains, self.swapchain_images)):
            view_config = self.view_config_views[eye_idx]
            width = view_config.recommended_image_rect_width
            height = view_config.recommended_image_rect_height

            eye_fbs = []
            for img in images:
                # Create framebuffer
                fbo = GL.glGenFramebuffers(1)
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)

                # Attach swapchain image as color attachment
                GL.glFramebufferTexture2D(
                    GL.GL_FRAMEBUFFER,
                    GL.GL_COLOR_ATTACHMENT0,
                    GL.GL_TEXTURE_2D,
                    img.image,
                    0
                )

                # Create depth buffer
                depth_buffer = GL.glGenRenderbuffers(1)
                GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth_buffer)
                GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH24_STENCIL8, width, height)
                GL.glFramebufferRenderbuffer(
                    GL.GL_FRAMEBUFFER,
                    GL.GL_DEPTH_STENCIL_ATTACHMENT,
                    GL.GL_RENDERBUFFER,
                    depth_buffer
                )

                status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                if status != GL.GL_FRAMEBUFFER_COMPLETE:
                    print(f"Warning: Framebuffer incomplete: {status}")

                eye_fbs.append((fbo, depth_buffer))

            self.frame_buffers.append(eye_fbs)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def init_mujoco_rendering(self):
        """Initialize MuJoCo rendering context."""
        print("Initializing MuJoCo rendering...")

        self.mj_scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.mj_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    def quat_to_forward_vector(self, quat):
        """Convert OpenXR quaternion to forward direction vector."""
        # OpenXR quaternion: (x, y, z, w)
        x, y, z, w = quat.x, quat.y, quat.z, quat.w

        # Forward vector is -Z in OpenXR (right-handed)
        # Rotate (0, 0, -1) by quaternion
        forward = np.array([
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            -(1 - 2 * (x * x + y * y))
        ])
        return forward

    def render_eye(self, eye_idx, view, fbo, width, height):
        """Render scene for one eye using HMD pose."""
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glViewport(0, 0, width, height)
        GL.glClearColor(0.2, 0.3, 0.4, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # Extract pose from OpenXR view (already includes per-eye IPD offset!)
        pos = view.pose.position
        quat = view.pose.orientation

        # Base position in MuJoCo world (VR origin)
        # This is where you "stand" in the scene
        base_pos = np.array([0.4, 0.3, 0.4])

        # OpenXR coordinate system: +X right, +Y up, -Z forward
        # MuJoCo coordinate system: +X forward, +Y left, +Z up
        # Convert OpenXR position to MuJoCo
        cam_pos = base_pos + np.array([
            -pos.z,  # OpenXR -Z -> MuJoCo +X
            -pos.x,  # OpenXR -X -> MuJoCo +Y
            pos.y    # OpenXR +Y -> MuJoCo +Z
        ])

        # Get forward direction from quaternion
        forward = self.quat_to_forward_vector(quat)

        # Convert forward to MuJoCo coords
        mj_forward = np.array([
            -forward[2],  # OpenXR -Z -> MuJoCo +X
            -forward[0],  # OpenXR -X -> MuJoCo +Y
            forward[1]    # OpenXR +Y -> MuJoCo +Z
        ])

        # Calculate lookat point (some distance in front of camera)
        look_distance = 0.5
        look_target = cam_pos + mj_forward * look_distance

        # Create MuJoCo camera
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        # Calculate distance and angles from camera to lookat
        diff = look_target - cam_pos
        distance = np.linalg.norm(diff)

        # MuJoCo uses: azimuth (rotation around Z), elevation (angle from XY plane)
        azimuth = np.degrees(np.arctan2(-diff[1], diff[0]))  # Note: MuJoCo Y is left
        horizontal_dist = np.sqrt(diff[0]**2 + diff[1]**2)
        elevation = np.degrees(np.arctan2(diff[2], horizontal_dist))

        cam.lookat[:] = look_target
        cam.distance = max(distance, 0.1)
        cam.azimuth = azimuth
        cam.elevation = elevation

        # Update scene
        mujoco.mjv_updateScene(
            self.model, self.data, self.mj_option, None,
            cam, mujoco.mjtCatBit.mjCAT_ALL, self.mj_scene
        )

        # Create viewport
        viewport = mujoco.MjrRect(0, 0, width, height)

        # Render
        mujoco.mjr_render(viewport, self.mj_scene, self.mj_context)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def run(self):
        """Main VR render loop."""
        print("\n" + "="*60)
        print("MuJoCo VR Viewer - OpenXR")
        print("="*60)
        print("Make sure Quest Link is active!")
        print("Press Ctrl+C to exit")
        print("="*60 + "\n")

        try:
            self.init_glfw()
            self.init_openxr()
            self.init_session()
            self.init_swapchains()
            self.init_mujoco_rendering()

            # Begin session
            session_begin_info = xr.SessionBeginInfo(
                primary_view_configuration_type=self.view_config_type
            )
            xr.begin_session(self.session, session_begin_info)
            print("Session started!")

            # Reference space (local = seated)
            ref_space_create_info = xr.ReferenceSpaceCreateInfo(
                reference_space_type=xr.ReferenceSpaceType.LOCAL,
                pose_in_reference_space=xr.Posef(
                    orientation=xr.Quaternionf(0, 0, 0, 1),
                    position=xr.Vector3f(0, 0, 0),
                ),
            )
            self.reference_space = xr.create_reference_space(self.session, ref_space_create_info)

            # Main loop
            while self.running:
                glfw.poll_events()

                if glfw.window_should_close(self.window):
                    break

                # Poll OpenXR events
                self.poll_events()

                if not self.running:
                    break

                # Wait for frame
                frame_state = xr.wait_frame(self.session)
                xr.begin_frame(self.session)

                # Get view poses
                view_locate_info = xr.ViewLocateInfo(
                    view_configuration_type=self.view_config_type,
                    display_time=frame_state.predicted_display_time,
                    space=self.reference_space,
                )

                view_state, views = xr.locate_views(self.session, view_locate_info)

                # Render each eye
                projection_views = []

                for eye_idx, (view, swapchain, images, fbs) in enumerate(
                    zip(views, self.swapchains, self.swapchain_images, self.frame_buffers)
                ):
                    view_config = self.view_config_views[eye_idx]
                    width = view_config.recommended_image_rect_width
                    height = view_config.recommended_image_rect_height

                    # Acquire swapchain image
                    swapchain_image_index = xr.acquire_swapchain_image(swapchain)

                    wait_info = xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION)
                    xr.wait_swapchain_image(swapchain, wait_info)

                    # Get framebuffer for this image
                    fbo, _ = fbs[swapchain_image_index]

                    # Render
                    self.render_eye(eye_idx, view, fbo, width, height)

                    # Release swapchain image
                    xr.release_swapchain_image(swapchain)

                    # Build projection view
                    projection_view = xr.CompositionLayerProjectionView(
                        pose=view.pose,
                        fov=view.fov,
                        sub_image=xr.SwapchainSubImage(
                            swapchain=swapchain,
                            image_rect=xr.Rect2Di(
                                offset=xr.Offset2Di(0, 0),
                                extent=xr.Extent2Di(width, height),
                            ),
                        ),
                    )
                    projection_views.append(projection_view)

                # Step simulation
                mujoco.mj_step(self.model, self.data)

                # Submit frame
                projection_layer = xr.CompositionLayerProjection(
                    space=self.reference_space,
                    views=projection_views,
                )

                layers = [ctypes.byref(projection_layer)]

                frame_end_info = xr.FrameEndInfo(
                    display_time=frame_state.predicted_display_time,
                    environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
                    layers=layers,
                )

                xr.end_frame(self.session, frame_end_info)

        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def poll_events(self):
        """Poll and handle OpenXR events."""
        while True:
            try:
                # pyopenxr poll_event returns the event directly
                event = xr.poll_event(self.instance)

                if event is None:
                    break

                # Check event type
                if isinstance(event, xr.EventDataSessionStateChanged):
                    print(f"Session state changed: {event.state}")

                    if event.state == xr.SessionState.STOPPING:
                        xr.end_session(self.session)
                        self.running = False
                    elif event.state == xr.SessionState.EXITING:
                        self.running = False
                    elif event.state == xr.SessionState.LOSS_PENDING:
                        self.running = False

            except xr.exception.EventUnavailable:
                break
            except Exception:
                break

    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")

        if self.session:
            try:
                xr.destroy_session(self.session)
            except:
                pass

        if self.instance:
            try:
                xr.destroy_instance(self.instance)
            except:
                pass

        if hasattr(self, 'window') and self.window:
            glfw.destroy_window(self.window)
            glfw.terminate()


if __name__ == "__main__":
    viewer = MuJoCoVRViewer()
    viewer.run()
