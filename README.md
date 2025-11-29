# <div style="text-align:center; border-radius:30px 30px; padding:7px; color:white; margin:0; font-size:150%; font-family:Arial; background-color:#636363; overflow:hidden"><b> Basic so-101 simulation environment</b></div>

 <p align="center">
<img src="assets/media/output.gif" alt="Basic policy" width="50%"/> 
</p>

Implements a so-101 robotic arm simulaltion environment for the [EnvHub](https://huggingface.co/docs/lerobot/envhub).

- Observation is a `np.ndarray.shape = (640, 480, 3)`.
- Action is a `np.ndarray.shape = 6` where each element represents the joint control. 
- Reward is the euclidian distance between the gripper and the red block, which it needs to minimize.


## Basic usage

```python
SO101Env(
    xml_pth=Path("assets/SO-ARM100/Simulation/SO101/scene_with_cube.xml"), 
    obs_w=640, 
    obs_h=480)
env = gym.make(
    "base-sO101-env-v0",
)
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

## ToDo
Things I want to do

- Make a hardcoded policy that can pick up the block.
- First attempt of training a model.


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