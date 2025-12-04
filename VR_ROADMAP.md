# VR Integration Roadmap for lerobot-gym

## Goal
View the SO101 simulation in VR (Meta Quest 3) while teleoperating with the physical leader arm.

## Architecture

```
[Physical Leader Arm] --> [teleop_sim.py] --> [MuJoCo Simulation]
       (COM8)                   |                    |
                                |                    v
                                |            [VR Headset Display]
                                |              (Quest 3 via Link)
                                v
                          [Desktop Mirror]
```

## Approach: OpenXR + pyopenxr

Using OpenXR (the modern VR standard) with the pyopenxr Python bindings. Quest 3 supports OpenXR natively via Quest Link or Air Link.

### Why OpenXR over OpenVR?
- OpenXR is the industry standard (Khronos Group)
- Native Quest 3 support via Quest Link
- Cross-platform (works with other headsets too)
- Active development and support

## Implementation Phases

### Phase 1: Basic VR Viewer (Current Target)
- [x] Research VR integration options
- [ ] Port MujocoXRViewer to work with SO101 environment
- [ ] Test with Quest 3 via Quest Link
- [ ] Stereoscopic rendering of simulation

### Phase 2: Teleop Integration
- [ ] Combine VR viewer with teleop_sim.py
- [ ] Physical leader arm controls sim while viewing in VR
- [ ] Desktop mirror window for debugging

### Phase 3: Enhancements
- [ ] Head tracking for camera control
- [ ] Controller support (optional alternative to physical arm)
- [ ] Hand tracking for natural control
- [ ] Performance optimization

## Dependencies

```bash
pip install pyopenxr glfw PyOpenGL numpy
```

## Hardware Requirements
- Meta Quest 3 (or other OpenXR-compatible headset)
- Quest Link cable or Air Link (Wi-Fi 6)
- Windows PC with decent GPU
- Physical leader arm (STS3250 motors)

## Key Files
- `vr_viewer.py` - Standalone VR viewer for simulation
- `teleop_vr.py` - Combined teleoperation + VR viewing
- `teleop_sim.py` - Existing teleoperation (flat screen)

## References
- [MuJoCo Visualization Docs](https://mujoco.readthedocs.io/en/stable/programming/visualization.html)
- [pyopenxr MuJoCo example](https://gist.github.com/SkytAsul/b1a48a31c4f86b65d72bc8edcb122d3f)
- [OpenAI mjvive.py](https://github.com/openai/mujoco-py/blob/master/examples/mjvive.py)
- [XRoboToolkit](https://arxiv.org/html/2508.00097v1)

## Notes
- MuJoCo has built-in stereo rendering (`mjSTEREO_SIDEBYSIDE`)
- Quest 3 connects via Quest Link app on PC
- OpenXR runtime must be set to Oculus in Windows settings
