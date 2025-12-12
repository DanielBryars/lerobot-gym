"""
Replay HuggingFace LeRobot datasets in the SO101 simulation.

This script loads a LeRobot dataset from HuggingFace and replays the recorded
actions in the simulation to compare calibration between recorded and simulated
robot movements.

Usage:
    # Replay a specific dataset
    python replay_dataset.py --dataset lerobot/svla_so100_pickplace --episode 0

    # Replay with side-by-side comparison
    python replay_dataset.py --dataset lerobot/svla_so100_pickplace --episode 0 --compare

    # List available episodes
    python replay_dataset.py --dataset lerobot/svla_so100_pickplace --list-episodes

    # Replay at slower speed
    python replay_dataset.py --dataset lerobot/svla_so100_pickplace --episode 0 --fps 10
"""

import argparse
import env
import gymnasium as gym
import numpy as np
import cv2
from pathlib import Path
from typing import Optional
import time


# Sim action space bounds (radians) - same as teleop_sim.py
SIM_ACTION_LOW = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453])
SIM_ACTION_HIGH = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533])

# Config directory
CONFIG_DIR = Path(__file__).parent / "dataset_configs"


class DatasetConfig:
    """Loads and applies dataset-specific configuration for replay."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.config = self._load_config(dataset_name)

    def _load_config(self, dataset_name: str) -> dict:
        """Load config file for dataset, or return defaults."""
        import json

        # Extract short name from repo_id (e.g., "lerobot/svla_so100_pickplace" -> "svla_so100_pickplace")
        short_name = dataset_name.split("/")[-1]
        config_path = CONFIG_DIR / f"{short_name}.json"

        if config_path.exists():
            print(f"Loading config: {config_path}")
            with open(config_path) as f:
                return json.load(f)
        else:
            print(f"No config found at {config_path}, using defaults")
            return self._default_config()

    def _default_config(self) -> dict:
        """Default config if none found."""
        return {
            "joint_mapping": {
                "order": [0, 1, 2, 3, 4, 5]
            },
            "transform": {
                "unit": "degreesToRadians",
                "offset": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }
        }

    def transform(self, values: np.ndarray) -> np.ndarray:
        """
        Apply configured transform to values.

        Transform order: unit_transform(values) * scale + offset
        """
        t = self.config.get("transform", {})
        unit = t.get("unit", "unity")
        offset = np.array(t.get("offset", [0.0] * 6), dtype=np.float32)
        scale = np.array(t.get("scale", [1.0] * 6), dtype=np.float32)

        # Reorder joints if needed
        order = self.config.get("joint_mapping", {}).get("order", [0, 1, 2, 3, 4, 5])
        values = values[order]

        # Apply unit transform
        if unit == "degreesToRadians":
            result = np.deg2rad(values).astype(np.float32)
        elif unit == "unity":
            result = values.astype(np.float32)
        else:
            result = values.astype(np.float32)

        # Apply scale and offset
        result = result * scale + offset

        return result

    def print_info(self):
        """Print config info."""
        t = self.config.get("transform", {})
        jm = self.config.get("joint_mapping", {})
        print(f"Config: {self.dataset_name}")
        print(f"  Unit transform: {t.get('unit', 'unity')}")
        print(f"  Scale:  {t.get('scale', [1]*6)}")
        print(f"  Offset: {t.get('offset', [0]*6)}")
        if "dataset_names" in jm:
            print(f"  Dataset joints: {jm['dataset_names']}")
        if "order" in jm:
            print(f"  Joint order: {jm['order']}")


def load_lerobot_dataset(repo_id: str):
    """Load a LeRobot dataset from HuggingFace."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id)
    print(f"  Total frames: {len(dataset)}")
    print(f"  Episodes: {dataset.num_episodes}")
    print(f"  FPS: {dataset.fps}")

    # Print feature info
    if hasattr(dataset, 'features'):
        print(f"  Features: {list(dataset.features.keys())}")

    return dataset


def get_episode_bounds(dataset, episode_index: int) -> tuple:
    """Get start and end indices for a specific episode."""
    ep_meta = dataset.meta.episodes[episode_index]
    from_idx = ep_meta['dataset_from_index']
    to_idx = ep_meta['dataset_to_index']
    return from_idx, to_idx


def get_episode_frames(dataset, episode_index: int) -> list:
    """Get all frames for a specific episode."""
    frames = []

    # Get episode data bounds from meta.episodes
    from_idx, to_idx = get_episode_bounds(dataset, episode_index)

    print(f"Episode {episode_index}: frames {from_idx} to {to_idx} ({to_idx - from_idx} frames)")

    for idx in range(from_idx, to_idx):
        frame = dataset[idx]
        frames.append(frame)

    return frames


def list_episodes(dataset):
    """List all episodes in the dataset with basic info."""
    print(f"\nDataset has {dataset.num_episodes} episodes:")
    print("-" * 60)

    for ep in range(min(dataset.num_episodes, 50)):  # Show max 50
        from_idx, to_idx = get_episode_bounds(dataset, ep)
        n_frames = to_idx - from_idx
        duration = n_frames / dataset.fps
        print(f"  Episode {ep:3d}: {n_frames:4d} frames ({duration:.1f}s)")

    if dataset.num_episodes > 50:
        print(f"  ... and {dataset.num_episodes - 50} more episodes")


def extract_action(frame: dict) -> np.ndarray:
    """Extract action from a dataset frame."""
    action = frame.get("action")
    if action is None:
        # Try alternate keys
        for key in frame.keys():
            if "action" in key.lower():
                action = frame[key]
                break

    if action is None:
        raise ValueError(f"No action found in frame. Keys: {list(frame.keys())}")

    # Convert to numpy
    if hasattr(action, 'numpy'):
        action = action.numpy()

    return np.array(action, dtype=np.float32)


def extract_state(frame: dict) -> Optional[np.ndarray]:
    """Extract observation state from a dataset frame."""
    state = frame.get("observation.state")
    if state is None:
        for key in frame.keys():
            if "state" in key.lower() and "observation" in key.lower():
                state = frame[key]
                break

    if state is not None and hasattr(state, 'numpy'):
        state = state.numpy()

    return np.array(state, dtype=np.float32) if state is not None else None


def extract_image(frame: dict, camera_key: str = None) -> Optional[np.ndarray]:
    """Extract image from a dataset frame."""
    # Try to find an image key
    image_keys = [k for k in frame.keys() if "image" in k.lower()]

    if camera_key and camera_key in frame:
        img = frame[camera_key]
    elif image_keys:
        # Prefer 'top' camera if available
        top_keys = [k for k in image_keys if "top" in k.lower()]
        if top_keys:
            img = frame[top_keys[0]]
        else:
            img = frame[image_keys[0]]
    else:
        return None

    # Convert to numpy if tensor
    if hasattr(img, 'numpy'):
        img = img.numpy()

    # Handle different image formats
    if isinstance(img, dict):
        # Video data might be in a dict
        if 'path' in img:
            return None  # Need to load from video file
        img = img.get('array', img.get('data', None))

    if img is None:
        return None

    # Ensure proper format (H, W, C) and uint8
    img = np.array(img)

    # Handle (C, H, W) format
    if len(img.shape) == 3 and img.shape[0] in [1, 3, 4]:
        img = np.transpose(img, (1, 2, 0))

    # Convert to uint8 if float
    if img.dtype in [np.float32, np.float64]:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    return img


def replay_episode(
    dataset,
    episode_index: int,
    compare: bool = False,
    fps: int = 30,
    start_frame: int = 0,
    use_state_init: bool = True,
    config: DatasetConfig = None,
):
    """Replay a single episode from the dataset."""

    # Get episode frames
    frames = get_episode_frames(dataset, episode_index)
    if not frames:
        print(f"No frames found for episode {episode_index}")
        return

    # Load config if not provided
    if config is None:
        config = DatasetConfig(dataset.repo_id)

    # Show config info
    config.print_info()
    if 'action' in dataset.features:
        ds_names = dataset.features['action'].get('names', [])
        print(f"  Dataset joints: {ds_names}")

    print(f"\nReplaying episode {episode_index} ({len(frames)} frames)...")
    print("Controls: [SPACE] pause/play, [Q] quit, [LEFT/RIGHT] step, [R] restart")
    print("          [V] toggle wrist camera, [W/S] elev, [Z/X] azi, [-/+] zoom")

    # Create environment
    e = gym.make("base-sO101-env-v0")
    obs_sim, _ = e.reset()

    # Initialize from first frame state if available
    if use_state_init:
        first_state = extract_state(frames[0])
        if first_state is not None:
            first_state_rad = config.transform(first_state)
            first_state_rad = np.clip(first_state_rad, SIM_ACTION_LOW, SIM_ACTION_HIGH)
            print(f"Initializing sim from recorded state:")
            print(f"  Dataset: {first_state}")
            print(f"  Sim:     {first_state_rad}")
            e.unwrapped.mj_data.qpos[:6] = first_state_rad
            import mujoco
            mujoco.mj_forward(e.unwrapped.mj_model, e.unwrapped.mj_data)
            obs_sim = e.unwrapped._get_obs()

    # Setup display
    window_name = "Dataset Replay (Side-by-Side)" if compare else "Dataset Replay (Simulation)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280 if compare else 800, 600)

    frame_idx = start_frame
    paused = False
    frame_delay = int(1000 / fps)  # ms per frame
    show_wrist_cam = False  # Toggle with 'V' key

    while frame_idx < len(frames):
        frame_data = frames[frame_idx]

        # Extract action and convert using config
        try:
            action_raw = extract_action(frame_data)

            # Apply config transform
            action = config.transform(action_raw)

            # Clip action to valid range
            action = np.clip(action, e.action_space.low, e.action_space.high)
            obs_sim, reward, terminated, truncated, info = e.step(action)
        except Exception as ex:
            print(f"Error at frame {frame_idx}: {ex}")
            action = e.action_space.sample() * 0  # Zero action
            action_raw = action

        # Get recorded image if comparing
        if compare:
            recorded_img = extract_image(frame_data)

        # Get wrist camera view if enabled
        wrist_img = None
        if show_wrist_cam:
            wrist_img = e.unwrapped.get_wrist_camera_obs(width=320, height=240)

        # Prepare display frame
        sim_display = cv2.cvtColor(obs_sim, cv2.COLOR_RGB2BGR)

        # Overlay wrist camera in corner if available
        if wrist_img is not None:
            wrist_display = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
            # Add border
            wrist_display = cv2.copyMakeBorder(wrist_display, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 255, 0))
            cv2.putText(wrist_display, "WRIST", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # Place in top-right corner
            h, w = wrist_display.shape[:2]
            y_off, x_off = 10, sim_display.shape[1] - w - 10
            sim_display[y_off:y_off+h, x_off:x_off+w] = wrist_display

        if compare and recorded_img is not None:
            # Resize recorded to match sim
            recorded_resized = cv2.resize(recorded_img, (obs_sim.shape[1], obs_sim.shape[0]))
            if len(recorded_resized.shape) == 3 and recorded_resized.shape[2] == 3:
                recorded_display = cv2.cvtColor(recorded_resized, cv2.COLOR_RGB2BGR)
            else:
                recorded_display = recorded_resized

            # Add labels
            cv2.putText(recorded_display, "RECORDED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(sim_display, "SIMULATION", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Side by side
            display = np.hstack([recorded_display, sim_display])
        else:
            display = sim_display

        # Add info overlay
        info_y = display.shape[0] - 80
        cv2.putText(display, f"Episode: {episode_index}  Frame: {frame_idx}/{len(frames)}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"Raw (deg): [{action_raw[0]:.1f}, {action_raw[1]:.1f}, {action_raw[2]:.1f}, ...]",
                   (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(display, f"Sim (radians):  [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, ...]",
                   (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        if paused:
            cv2.putText(display, "PAUSED", (display.shape[1]//2 - 50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow(window_name, display)

        # Handle input
        key = cv2.waitKey(1 if not paused else 100)  # 100ms timeout when paused to stay responsive
        if key == -1:
            if not paused:
                frame_idx += 1
                time.sleep(max(0, frame_delay / 1000 - 0.01))
            continue

        key_char = key & 0xFF

        # Debug: uncomment to see key codes
        # print(f"Key pressed: {key} char: {key_char} chr: {chr(key_char) if 32 <= key_char < 127 else '?'}")

        if key_char == ord('q') or key_char == 27:  # q or ESC
            print("Quit requested")
            break
        elif key_char == ord(' ') or key_char == 32:
            paused = not paused
            print(f"{'PAUSED' if paused else 'PLAYING'}")
        elif key_char == ord('r'):
            # Restart episode
            frame_idx = 0
            e.reset()
            if use_state_init:
                first_state = extract_state(frames[0])
                if first_state is not None:
                    first_state_rad = config.transform(first_state)
                    first_state_rad = np.clip(first_state_rad, SIM_ACTION_LOW, SIM_ACTION_HIGH)
                    e.unwrapped.mj_data.qpos[:6] = first_state_rad
                    import mujoco
                    mujoco.mj_forward(e.unwrapped.mj_model, e.unwrapped.mj_data)
            print("Restarted")
            continue
        elif key_char == ord('a') or key_char == ord(',') or key_char == 81:  # a or , or left
            frame_idx = max(0, frame_idx - 1)
            continue
        elif key_char == ord('d') or key_char == ord('.') or key_char == 83:  # d or . or right
            frame_idx = min(len(frames) - 1, frame_idx + 1)
            continue
        # Camera controls
        elif key_char == ord('w'):
            e.unwrapped.cam_elev = min(89, e.unwrapped.cam_elev + 5)
        elif key_char == ord('s'):
            e.unwrapped.cam_elev = max(-89, e.unwrapped.cam_elev - 5)
        elif key_char == ord('z'):
            e.unwrapped.cam_azi -= 10
        elif key_char == ord('x'):
            e.unwrapped.cam_azi += 10
        elif key_char == ord('-'):
            e.unwrapped.cam_dis += 0.1
        elif key_char == ord('=') or key_char == ord('+'):
            e.unwrapped.cam_dis = max(0.3, e.unwrapped.cam_dis - 0.1)
        elif key_char == ord('v'):
            show_wrist_cam = not show_wrist_cam
            print(f"Wrist camera: {'ON' if show_wrist_cam else 'OFF'}")

    cv2.destroyAllWindows()
    e.close()
    print("Replay complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Replay HuggingFace LeRobot datasets in SO101 simulation"
    )
    parser.add_argument(
        "--dataset", "-d", type=str, required=True,
        help="HuggingFace dataset repo ID (e.g., lerobot/svla_so100_pickplace)"
    )
    parser.add_argument(
        "--episode", "-e", type=int, default=0,
        help="Episode index to replay (default: 0)"
    )
    parser.add_argument(
        "--list-episodes", "-l", action="store_true",
        help="List all episodes in the dataset and exit"
    )
    parser.add_argument(
        "--compare", "-c", action="store_true",
        help="Show side-by-side comparison with recorded images"
    )
    parser.add_argument(
        "--fps", "-f", type=int, default=30,
        help="Playback FPS (default: 30)"
    )
    parser.add_argument(
        "--start-frame", "-s", type=int, default=0,
        help="Start from this frame index (default: 0)"
    )
    parser.add_argument(
        "--no-state-init", action="store_true",
        help="Don't initialize simulation from recorded initial state"
    )

    args = parser.parse_args()

    # Load dataset
    dataset = load_lerobot_dataset(args.dataset)

    if args.list_episodes:
        list_episodes(dataset)
        return

    # Validate episode index
    if args.episode < 0 or args.episode >= dataset.num_episodes:
        print(f"Error: Episode {args.episode} out of range (0-{dataset.num_episodes-1})")
        return

    # Load config for this dataset
    config = DatasetConfig(args.dataset)

    # Replay
    replay_episode(
        dataset=dataset,
        episode_index=args.episode,
        compare=args.compare,
        fps=args.fps,
        start_frame=args.start_frame,
        use_state_init=not args.no_state_init,
        config=config,
    )


if __name__ == "__main__":
    main()
