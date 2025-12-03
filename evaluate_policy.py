"""
Evaluate a trained LeRobot policy in SO101 simulation.

Usage:
    python evaluate_policy.py --policy danbhf/act_so100_pick_place --episodes 10
    python evaluate_policy.py --policy danbhf/smolVla_so100_pick_place --episodes 3 --render
    python evaluate_policy.py --policy random --episodes 3 --render
"""
import argparse
import env
import gymnasium as gym
import numpy as np
import torch
import cv2
from pathlib import Path
from contextlib import nullcontext


def load_policy_and_processors(policy_path: str, device: str = "cpu"):
    """Load a LeRobot policy with its preprocessor and postprocessor."""
    from lerobot.policies.factory import make_pre_post_processors

    # Fix Windows path separators
    policy_path = str(policy_path).replace("\\", "/")
    print(f"Loading policy from: {policy_path}")

    # Determine policy type and load
    path_lower = policy_path.lower()
    if "smolvla" in path_lower:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        policy = SmolVLAPolicy.from_pretrained(policy_path)
    elif "act" in path_lower:
        from lerobot.policies.act.modeling_act import ACTPolicy
        policy = ACTPolicy.from_pretrained(policy_path)
    elif "diffusion" in path_lower:
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        policy = DiffusionPolicy.from_pretrained(policy_path)
    else:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        policy = SmolVLAPolicy.from_pretrained(policy_path)

    policy.to(device)
    policy.eval()
    print(f"Policy type: {type(policy).__name__}")

    # Create preprocessor and postprocessor
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=policy_path,
    )

    return policy, preprocessor, postprocessor


def prepare_observation(
    image: np.ndarray,
    state: np.ndarray,
    task: str,
    device: torch.device,
) -> dict:
    """
    Prepare observation dict for lerobot pipeline.

    Keys follow lerobot batch naming convention:
    - observation.images.camera1
    - observation.state
    - task (complementary data, tokenized by preprocessor)
    """
    obs = {}

    # Image: (H, W, C) uint8 -> (C, H, W) float32 normalized
    # Note: preprocessor will add batch dimension
    img_tensor = torch.from_numpy(image).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).contiguous()
    obs["observation.images.camera1"] = img_tensor

    # State: (6,)
    state_tensor = torch.from_numpy(state).float()
    obs["observation.state"] = state_tensor

    # Language task as complementary data (preprocessor handles tokenization)
    obs["task"] = task

    return obs


def get_robot_state(env) -> np.ndarray:
    """Extract robot joint positions from environment."""
    unwrapped = env.unwrapped
    return unwrapped.mj_data.qpos[:6].copy().astype(np.float32)


def run_evaluation(
    policy_path: str,
    task_instruction: str,
    episodes: int = 10,
    max_steps: int = 500,
    render: bool = False,
    device: str = "cpu",
):
    """Run policy evaluation in SO101 simulation."""

    # Create environment
    print("Creating SO101 environment...")
    e = gym.make("base-sO101-env-v0")

    # Load policy (skip if "random")
    if policy_path.lower() == "random":
        print("Using random actions")
        policy = None
        preprocessor = None
        postprocessor = None
    else:
        try:
            policy, preprocessor, postprocessor = load_policy_and_processors(policy_path, device)
        except Exception as ex:
            print(f"Error loading policy: {ex}")
            import traceback
            traceback.print_exc()
            print("\nFalling back to random actions...")
            policy = None
            preprocessor = None
            postprocessor = None

    results = []
    torch_device = torch.device(device)

    # Try to enable rendering if requested
    render_enabled = False
    if render:
        try:
            cv2.namedWindow("SO101 Simulation", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("SO101 Simulation", 800, 600)
            render_enabled = True
            print("\nVisualization enabled - press 'q' to quit")
        except cv2.error as e:
            print(f"\nWarning: Cannot create window ({e})")
            print("Running without visualization.")
            print("To enable: pip uninstall opencv-python-headless && pip install opencv-python")

    print(f"\nRunning {episodes} episodes (max {max_steps} steps each)...")
    print(f"Task instruction: '{task_instruction}'")
    print("-" * 50)

    quit_requested = False

    for ep in range(episodes):
        if quit_requested:
            break

        obs_image, _ = e.reset()

        if policy is not None:
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

        total_reward = 0

        for step in range(max_steps):
            # Render if enabled
            if render_enabled:
                frame = cv2.cvtColor(obs_image, cv2.COLOR_RGB2BGR)
                cv2.putText(frame, f"Episode: {ep+1}/{episodes}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Step: {step}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Reward: {total_reward:.2f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("SO101 Simulation", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    quit_requested = True
                    break

            if policy is not None:
                try:
                    # Get state from environment
                    state = get_robot_state(e)

                    # Prepare observation
                    observation = prepare_observation(
                        obs_image, state, task_instruction, torch_device
                    )

                    # Run through preprocessor (handles tokenization)
                    with torch.inference_mode():
                        processed_obs = preprocessor(observation)

                        # Get action from policy
                        action = policy.select_action(processed_obs)

                        # Run through postprocessor
                        action = postprocessor(action)

                    # Convert to numpy
                    action = action.squeeze().cpu().numpy()

                except Exception as ex:
                    if step < 5:  # Only print first few errors
                        print(f"  Policy error at step {step}: {ex}")
                    action = e.action_space.sample()
            else:
                action = e.action_space.sample()

            # Clip action to valid range
            action = np.clip(action, e.action_space.low, e.action_space.high)

            obs_image, reward, terminated, truncated, info = e.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        results.append({
            "episode": ep,
            "steps": step + 1,
            "total_reward": total_reward,
            "final_reward": reward,
        })

        print(f"Episode {ep + 1}/{episodes}: steps={step + 1}, reward={total_reward:.3f}")

    if render_enabled:
        cv2.destroyAllWindows()

    e.close()

    # Summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    avg_reward = np.mean([r["total_reward"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])
    print(f"Policy:       {policy_path}")
    print(f"Task:         {task_instruction}")
    print(f"Episodes:     {len(results)}")
    print(f"Avg Steps:    {avg_steps:.1f}")
    print(f"Avg Reward:   {avg_reward:.3f}")
    print("=" * 50)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LeRobot policy in SO101 sim")
    parser.add_argument("--policy", "-p", type=str, required=True,
                        help="Policy path (HuggingFace repo or local checkpoint)")
    parser.add_argument("--task", "-t", type=str, default="Pick up the red cube",
                        help="Task instruction for VLA models")
    parser.add_argument("--episodes", "-e", type=int, default=10,
                        help="Number of episodes")
    parser.add_argument("--max-steps", "-s", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--render", "-r", action="store_true",
                        help="Render visualization")
    parser.add_argument("--device", "-d", type=str, default="cpu",
                        help="Device (cuda/cpu)")

    args = parser.parse_args()

    run_evaluation(
        policy_path=args.policy,
        task_instruction=args.task,
        episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        device=args.device,
    )


if __name__ == "__main__":
    main()
