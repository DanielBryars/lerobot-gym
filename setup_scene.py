#!/usr/bin/env python
"""
Setup script to create the scene_with_cube.xml file.

Run this after cloning the repo:
    python setup_scene.py
"""
from pathlib import Path

SCENE_XML = '''<mujoco model="scene_with_cube">
    <include file="so101_new_calib.xml" />

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
        <rgba haze="0.15 0.25 0.35 1" />
        <global azimuth="160" elevation="-20" />
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512"
            height="3072" />
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
            rgb2="0.1 0.2 0.3"
            markrgb="0.8 0.8 0.8" width="300" height="300" />
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"
            reflectance="0.2" />
        <material name="cube_material" rgba="1 0 0 1" />
    </asset>

    <worldbody>
        <light pos="0 0 3.5" dir="0 0 -1" directional="true" />
        <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane" />

        <!-- Red cube for pick and place task -->
        <body name="cube" pos="0.2 0.0 0.025">
            <freejoint name="cube_joint"/>
            <geom name="cube_geom" type="box" size="0.025 0.025 0.025" material="cube_material" mass="0.1"/>
        </body>
    </worldbody>
</mujoco>
'''

def main():
    script_dir = Path(__file__).parent
    scene_path = script_dir / "assets" / "SO-ARM100" / "Simulation" / "SO101" / "scene_with_cube.xml"

    if scene_path.exists():
        print(f"Scene file already exists: {scene_path}")
        return

    scene_path.parent.mkdir(parents=True, exist_ok=True)
    scene_path.write_text(SCENE_XML)
    print(f"Created scene file: {scene_path}")

if __name__ == "__main__":
    main()
