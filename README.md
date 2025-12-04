# <div style="text-align:center; border-radius:30px 30px; padding:7px; color:white; margin:0; font-size:150%; font-family:Arial; background-color:#636363; overflow:hidden"><b> Basic so-101 simulation environment</b></div>

 <p align="center">
<img src="assets/media/output.gif" alt="Basic policy" width="75%"/> 
</p>

Implements a so-101 robotic arm simulaltion environment for the [EnvHub](https://huggingface.co/docs/lerobot/envhub).

- Observation is a `np.ndarray.shape = (640, 480, 3)`.
- Action is a `np.ndarray.shape = 6` where each element represents the joint control. 
- Reward is the euclidian distance between the gripper and the red block, which it needs to minimize.


## Installation

```bash
git clone https://github.com/DanielBryars/lerobot-gym.git
cd lerobot-gym
git submodule update --init
python setup_scene.py
pip install -e .
```

## Basic usage

```python
import gymnasium as gym
import env  # registers the environment

env = gym.make("base-sO101-env-v0")
try:
    obs, _ = env.reset()
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
finally:
    env.close()
```

## Evaluating LeRobot Policies

You can evaluate trained LeRobot policies (ACT, SmolVLA, etc.) in the simulation:

```bash
# Install extra dependencies
pip install -r requirements-eval.txt

# Run with a HuggingFace model
python evaluate_policy.py --policy danbhf/smolVla_so100_pick_place --episodes 5 --device cuda

# Run with random actions (for testing)
python evaluate_policy.py --policy random --episodes 3 --render
```

## Teleoperation

Control the simulation with a physical leader arm:

```bash
# Install motor control dependencies
pip install -r requirements-eval.txt

# Run teleoperation (reads COM port from config.json)
python teleop_sim.py

# Or specify port explicitly
python teleop_sim.py --leader-port COM8 --fps 30
```

Requires STS3250 motors with calibration stored in EEPROM.

## VR Support (Quest 3 / OpenXR)

View the simulation in VR while teleoperating with the physical arm:

```bash
# Install VR dependencies
pip install -r requirements-vr.txt

# Standalone VR viewer (no teleop)
python vr_viewer.py --mirror

# VR + physical leader arm teleoperation
python teleop_vr.py --mirror
```

**Setup:**
1. Connect Quest 3 via Quest Link (cable or Air Link)
2. Set OpenXR runtime to Oculus in Windows Settings
3. Run the script and put on headset

See [VR_ROADMAP.md](VR_ROADMAP.md) for details.

## ToDo
Things I want to do

- Make a hardcoded policy that can pick up the block.
- First attempt of training a model.
- Test VR teleoperation with Quest 3


# <div style="text-align:center; border-radius:30px 30px; padding:7px; color:white; margin:0; font-size:150%; font-family:Arial; background-color:#636363; overflow:hidden"><b> References</b></div>
## Assets
All robot files are from [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
```bib
@software{Knight_Standard_Open_SO-100,
    author = {Knight, Rob and Kooijmans, Pepijn and Cadene, Remi and Alibert, Simon and Aractingi, Michel and Aubakirova, Dana and Zouitine, Adil and Martino, Russi and Palma, Steven and Pascal, Caroline and Wolf, Thomas},
    title = {{Standard Open SO-100 \& SO-101 Arms}},
    url = {https://github.com/TheRobotStudio/SO-ARM100}
}
```
## MuJoCo

[MuJoCo library used](https://github.com/google-deepmind/mujoco)
```bib
@inproceedings{todorov2012mujoco,
  title={MuJoCo: A physics engine for model-based control},
  author={Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  booktitle={2012 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  pages={5026--5033},
  year={2012},
  organization={IEEE},
  doi={10.1109/IROS.2012.6386109}
}
```
