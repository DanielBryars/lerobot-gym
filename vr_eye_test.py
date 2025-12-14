"""
Simple VR Eye Test - Renders RED to left eye, BLUE to right eye.
Use this to verify stereo mapping is correct.
"""

import ctypes
import numpy as np
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


class VREyeTest:
    def __init__(self):
        self.instance = None
        self.system_id = None
        self.session = None
        self.swapchains = []
        self.swapchain_images = []
        self.frame_buffers = []
        self.running = True

    def init_openxr(self):
        print("Initializing OpenXR...")
        extensions = [xr.KHR_OPENGL_ENABLE_EXTENSION_NAME]
        create_info = xr.InstanceCreateInfo(enabled_extension_names=extensions)
        self.instance = xr.create_instance(create_info)

        instance_props = xr.get_instance_properties(self.instance)
        print(f"OpenXR Runtime: {instance_props.runtime_name}")

        system_get_info = xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY)
        self.system_id = xr.get_system(self.instance, system_get_info)

        self.view_config_type = xr.ViewConfigurationType.PRIMARY_STEREO
        self.view_config_views = xr.enumerate_view_configuration_views(
            self.instance, self.system_id, self.view_config_type
        )

        for i, view in enumerate(self.view_config_views):
            print(f"  Eye {i}: {view.recommended_image_rect_width}x{view.recommended_image_rect_height}")

    def init_glfw(self):
        print("Initializing GLFW...")
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)
        glfw.window_hint(glfw.VISIBLE, True)

        self.window = glfw.create_window(800, 600, "VR Eye Test", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        print(f"OpenGL Version: {GL.glGetString(GL.GL_VERSION).decode()}")

    def init_session(self):
        print("Creating OpenXR session...")

        pfn_get_opengl_graphics_requirements = ctypes.cast(
            xr.get_instance_proc_addr(self.instance, "xrGetOpenGLGraphicsRequirementsKHR"),
            xr.PFN_xrGetOpenGLGraphicsRequirementsKHR
        )

        graphics_requirements = xr.GraphicsRequirementsOpenGLKHR()
        pfn_get_opengl_graphics_requirements(
            self.instance, self.system_id, ctypes.byref(graphics_requirements)
        )

        from OpenGL import WGL
        graphics_binding = xr.GraphicsBindingOpenGLWin32KHR(
            h_dc=WGL.wglGetCurrentDC(),
            h_glrc=WGL.wglGetCurrentContext(),
        )

        session_create_info = xr.SessionCreateInfo(
            system_id=self.system_id,
            next=ctypes.cast(ctypes.pointer(graphics_binding), ctypes.c_void_p),
        )

        self.session = xr.create_session(self.instance, session_create_info)
        print("Session created!")

    def init_swapchains(self):
        print("Creating swapchains...")
        swapchain_formats = xr.enumerate_swapchain_formats(self.session)
        chosen_format = GL.GL_SRGB8_ALPHA8
        if chosen_format not in swapchain_formats:
            chosen_format = swapchain_formats[0]

        for i, view_config in enumerate(self.view_config_views):
            width = view_config.recommended_image_rect_width
            height = view_config.recommended_image_rect_height

            swapchain_create_info = xr.SwapchainCreateInfo(
                usage_flags=xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT,
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

            images = xr.enumerate_swapchain_images(swapchain, xr.SwapchainImageOpenGLKHR)
            self.swapchain_images.append(images)
            print(f"  Eye {i}: {width}x{height}, {len(images)} images")

        # Create framebuffers
        self.frame_buffers = []
        for eye_idx, images in enumerate(self.swapchain_images):
            view_config = self.view_config_views[eye_idx]
            width = view_config.recommended_image_rect_width
            height = view_config.recommended_image_rect_height

            eye_fbs = []
            for img in images:
                fbo = GL.glGenFramebuffers(1)
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
                GL.glFramebufferTexture2D(
                    GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                    GL.GL_TEXTURE_2D, img.image, 0
                )
                eye_fbs.append(fbo)
            self.frame_buffers.append(eye_fbs)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def render_eye(self, eye_idx, fbo, width, height):
        """Render solid color: RED for left (0), BLUE for right (1)."""
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glViewport(0, 0, width, height)

        if eye_idx == 0:
            # LEFT eye = RED
            GL.glClearColor(1.0, 0.0, 0.0, 1.0)
        else:
            # RIGHT eye = BLUE
            GL.glClearColor(0.0, 0.0, 1.0, 1.0)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def poll_events(self):
        while True:
            try:
                event = xr.poll_event(self.instance)
                if event is None:
                    break
                if isinstance(event, xr.EventDataSessionStateChanged):
                    print(f"Session state: {event.state}")
                    if event.state in [xr.SessionState.STOPPING, xr.SessionState.EXITING, xr.SessionState.LOSS_PENDING]:
                        self.running = False
            except xr.exception.EventUnavailable:
                break
            except Exception:
                break

    def run(self):
        print("\n" + "="*60)
        print("VR EYE TEST")
        print("="*60)
        print("You should see:")
        print("  - RED in your LEFT eye")
        print("  - BLUE in your RIGHT eye")
        print("")
        print("Close one eye at a time to check!")
        print("Press Ctrl+C to exit")
        print("="*60 + "\n")

        try:
            self.init_glfw()
            self.init_openxr()
            self.init_session()
            self.init_swapchains()

            session_begin_info = xr.SessionBeginInfo(
                primary_view_configuration_type=self.view_config_type
            )
            xr.begin_session(self.session, session_begin_info)
            print("Session started!")

            ref_space_create_info = xr.ReferenceSpaceCreateInfo(
                reference_space_type=xr.ReferenceSpaceType.LOCAL,
                pose_in_reference_space=xr.Posef(
                    orientation=xr.Quaternionf(0, 0, 0, 1),
                    position=xr.Vector3f(0, 0, 0),
                ),
            )
            self.reference_space = xr.create_reference_space(self.session, ref_space_create_info)

            while self.running:
                glfw.poll_events()
                if glfw.window_should_close(self.window):
                    break

                self.poll_events()
                if not self.running:
                    break

                frame_state = xr.wait_frame(self.session)
                xr.begin_frame(self.session)

                view_locate_info = xr.ViewLocateInfo(
                    view_configuration_type=self.view_config_type,
                    display_time=frame_state.predicted_display_time,
                    space=self.reference_space,
                )
                view_state, views = xr.locate_views(self.session, view_locate_info)

                projection_views = []

                for eye_idx, (view, swapchain, images, fbs) in enumerate(
                    zip(views, self.swapchains, self.swapchain_images, self.frame_buffers)
                ):
                    view_config = self.view_config_views[eye_idx]
                    width = view_config.recommended_image_rect_width
                    height = view_config.recommended_image_rect_height

                    swapchain_image_index = xr.acquire_swapchain_image(swapchain)
                    wait_info = xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION)
                    xr.wait_swapchain_image(swapchain, wait_info)

                    fbo = fbs[swapchain_image_index]
                    self.render_eye(eye_idx, fbo, width, height)

                    xr.release_swapchain_image(swapchain)

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

                projection_layer = xr.CompositionLayerProjection(
                    space=self.reference_space,
                    views=projection_views,
                )

                frame_end_info = xr.FrameEndInfo(
                    display_time=frame_state.predicted_display_time,
                    environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
                    layers=[ctypes.byref(projection_layer)],
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

    def cleanup(self):
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
    VREyeTest().run()
