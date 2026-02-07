#!/usr/bin/env python3
"""Record chess manipulation episodes with per-move task descriptions.

Each episode is tagged with the task string the orchestrator will produce at
inference time, so pi0.5 learns to condition on the correct language input.

Usage:
    uv run python scripts/collect_episodes.py \
        --robot-type so101 \
        --follower-port /dev/tty.usbmodem... \
        --leader-port /dev/tty.usbmodem... \
        --repo-id cosmos-chessbot/chess-manipulation

Move input: source and target square, e.g. "e2e4", "e2 e4", or "e2-e4".
Keyboard controls during recording:
    Right arrow  -- end episode early
    Left arrow   -- cancel and re-record
    ESC          -- stop session
"""

import re

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import (
    SO100Follower,
    SO100FollowerConfig,
    SO101Follower,
    SO101FollowerConfig,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so_leader import (
    SO100Leader,
    SO100LeaderConfig,
    SO101Leader,
    SO101LeaderConfig,
)
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

VALID_SQUARES = {f"{col}{row}" for col in "abcdefgh" for row in "12345678"}

ROBOT_CONFIGS = {
    "so100": (SO100Follower, SO100FollowerConfig, SO100Leader, SO100LeaderConfig),
    "so101": (SO101Follower, SO101FollowerConfig, SO101Leader, SO101LeaderConfig),
}


def parse_move(raw: str) -> tuple[str, str] | None:
    """Parse 'e2e4', 'e2 e4', or 'e2-e4' into (source, target). Returns None if invalid."""
    parts = re.split(r"[\s\-]+", raw.strip().lower())
    if len(parts) == 2:
        src, dst = parts
    elif len(parts) == 1 and len(parts[0]) == 4:
        src, dst = parts[0][:2], parts[0][2:]
    else:
        return None
    if src in VALID_SQUARES and dst in VALID_SQUARES and src != dst:
        return src, dst
    return None


def move_to_task(src: str, dst: str) -> str:
    """Format task string to match orchestrator/orchestrator.py:209."""
    return f"Pick the piece at {src} and place it at {dst}"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Record chess manipulation episodes")
    parser.add_argument("--robot-type", choices=["so100", "so101"], default="so101")
    parser.add_argument("--follower-port", required=True, help="USB port for follower arm")
    parser.add_argument("--leader-port", required=True, help="USB port for leader arm")
    parser.add_argument("--follower-id", default="my_follower_arm")
    parser.add_argument("--leader-id", default="my_leader_arm")
    parser.add_argument("--repo-id", default="cosmos-chessbot/chess-manipulation")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode-time-s", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    FollowerCls, FollowerCfg, LeaderCls, LeaderCfg = ROBOT_CONFIGS[args.robot_type]

    camera_config = {
        "egocentric": OpenCVCameraConfig(index_or_path=1, width=args.width, height=args.height, fps=args.fps),
        "wrist": OpenCVCameraConfig(index_or_path=0, width=args.width, height=args.height, fps=args.fps),
    }

    follower = FollowerCls(FollowerCfg(
        port=args.follower_port,
        id=args.follower_id,
        cameras=camera_config,
    ))
    leader = LeaderCls(LeaderCfg(
        port=args.leader_port,
        id=args.leader_id,
    ))

    action_features = hw_to_dataset_features(follower.action_features, "action")
    obs_features = hw_to_dataset_features(follower.observation_features, "observation")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features={**action_features, **obs_features},
        robot_type=follower.name,
        use_videos=True,
        image_writer_threads=4,
    )

    follower.connect()
    leader.connect()
    listener, events = init_keyboard_listener()
    init_rerun(session_name="chess_recording")
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    episode_idx = 0
    try:
        while not events["stop_recording"]:
            # Prompt for move before each episode
            print(f"\n--- Episode {episode_idx + 1} ---")
            while True:
                raw = input("Enter move (e.g. e2e4) or 'q' to quit: ").strip()
                if raw.lower() == "q":
                    raise KeyboardInterrupt
                parsed = parse_move(raw)
                if parsed:
                    src, dst = parsed
                    task = move_to_task(src, dst)
                    print(f"  Task: {task}")
                    print("  Teleoperate the move. Right arrow to end early.")
                    break
                print("  Invalid. Use: e2e4, e2 e4, or e2-e4")

            record_loop(
                robot=follower,
                events=events,
                fps=args.fps,
                teleop=leader,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                dataset=dataset,
                control_time_s=args.episode_time_s,
                single_task=task,
                display_data=True,
            )

            if events["rerecord_episode"]:
                log_say("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            episode_idx += 1
            log_say(f"Saved episode {episode_idx}: {task}")

            input("\nReset board for next move. Press Enter when ready...")

    except KeyboardInterrupt:
        pass
    finally:
        log_say(f"Stopping. {episode_idx} episodes recorded.")
        follower.disconnect()
        leader.disconnect()
        listener.stop()
        dataset.finalize()
        if args.push_to_hub:
            dataset.push_to_hub()
            log_say(f"Dataset pushed: {args.repo_id}")
        else:
            log_say("Dataset saved locally. Use --push-to-hub to upload.")


if __name__ == "__main__":
    main()
