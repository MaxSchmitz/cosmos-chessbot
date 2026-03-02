#!/usr/bin/env python3
"""Record manipulation episodes with per-episode task descriptions.

Each episode is tagged with a task string so pi0.5 learns to condition on
the correct language input.

Usage:
    # Chess moves (prompted per episode):
    uv run python scripts/collect_episodes.py \
        --follower-port /dev/tty.usbmodem... \
        --leader-port /dev/tty.usbmodem...

    # Fixed task (same task string for all episodes):
    uv run python scripts/collect_episodes.py \
        --follower-port /dev/tty.usbmodem... \
        --leader-port /dev/tty.usbmodem... \
        --task "Pick up the piece and place it in the bowl" \
        --repo-id maux/chess-bowl-pick-place

Keyboard controls during recording:
    Right arrow  -- end episode early
    Left arrow   -- cancel and re-record
    ESC          -- stop session
"""

import re

import numpy as np

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
    parser.add_argument("--task", type=str, default=None,
                        help="Fixed task string for all episodes (skips move prompt)")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume adding to existing dataset")
    parser.add_argument("--hud", action="store_true",
                        help="Enable HUD overlay on egocentric camera (visual move encoding)")
    parser.add_argument("--source", type=str, default=None,
                        help="Source location for HUD (square name or 'x,y' pixels)")
    parser.add_argument("--target", type=str, default=None,
                        help="Target location for HUD (square name or 'x,y' pixels)")
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

    if args.resume:
        dataset = LeRobotDataset(args.repo_id)
        print(f"Resuming dataset: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames")
    else:
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

    # --- HUD overlay setup ---
    _hud_state = {"source": args.source, "target": args.target, "corners": None, "homography": None}

    if args.hud:
        from cosmos_chessbot.vision.hud_overlay import (
            apply_hud,
            compute_homography,
            detect_corners,
        )
        print("HUD: enabled -- corners will be auto-detected from first frame via YOLO pose")

    follower.connect()
    leader.connect()

    # Monkey-patch get_observation to inject HUD markers on egocentric image
    if args.hud:
        _orig_get_observation = follower.get_observation

        def _hud_get_observation():
            obs = _orig_get_observation()
            src = _hud_state.get("source")
            tgt = _hud_state.get("target")
            if src and tgt and "observation.images.egocentric" in obs:
                img = obs["observation.images.egocentric"]
                if not isinstance(img, np.ndarray):
                    img = np.array(img)
                # Detect corners once from first frame, cache for episode
                if _hud_state["corners"] is None:
                    corners = detect_corners(img)
                    if corners is not None:
                        _hud_state["corners"] = corners
                        _hud_state["homography"] = compute_homography(corners)
                        print(f"HUD: detected board corners from first frame")
                    else:
                        print("HUD: WARNING -- could not detect board corners")
                apply_hud(
                    img,
                    src,
                    tgt,
                    corners=_hud_state.get("corners"),
                    homography=_hud_state.get("homography"),
                )
                obs["observation.images.egocentric"] = img
            return obs

        follower.get_observation = _hud_get_observation

    listener, events = init_keyboard_listener()
    init_rerun(session_name="chess_recording")
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    episode_idx = 0
    try:
        while not events["stop_recording"]:
            print(f"\n--- Episode {episode_idx + 1} ---")

            # Re-detect corners each episode in case the board shifted
            if args.hud:
                _hud_state["corners"] = None
                _hud_state["homography"] = None

            if args.task:
                # Fixed task mode
                task = args.task
                if args.hud and args.source and args.target:
                    _hud_state["source"] = args.source
                    _hud_state["target"] = args.target
                    task = "Pick up the highlighted piece and place it at the target"
                print(f"  Task: {task}")
                input("  Press Enter to start recording...")
            else:
                # Chess move prompt mode
                while True:
                    raw = input("Enter move (e.g. e2e4) or 'q' to quit: ").strip()
                    if raw.lower() == "q":
                        raise KeyboardInterrupt
                    parsed = parse_move(raw)
                    if parsed:
                        src, dst = parsed
                        if args.hud:
                            _hud_state["source"] = src
                            _hud_state["target"] = dst
                            task = "Pick up the highlighted piece and place it at the target"
                        else:
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

            # Guard against empty episodes (e.g. immediate exit)
            if not dataset.episode_buffer or not dataset.episode_buffer.get("size", 0):
                log_say("No frames recorded, skipping episode")
                dataset.clear_episode_buffer()
                events["exit_early"] = False
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
