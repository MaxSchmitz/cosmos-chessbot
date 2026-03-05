#!/usr/bin/env python3
"""Run a pi0.5 episode using lerobot's built-in async inference system.

Uses PolicyServer (on brev GPU) + RobotClient (local) with gRPC transport
and weighted_average action blending for smooth chunk transitions.

Server setup (brev):
    uv run python -m lerobot.async_inference.policy_server --port 8002

SSH tunnel:
    ssh -f -N -L 8002:localhost:8002 ubuntu@isaacsim

Usage:
    uv run python scripts/hardware/run_lerobot_async.py \
        --port /dev/tty.usbmodem58FA0962531 \
        --source b2 --target b4

    # More sequential-like:
    uv run python scripts/hardware/run_lerobot_async.py \
        --port /dev/tty.usbmodem58FA0962531 \
        --source b2 --target b4 --chunk-size-threshold 0.1

Press Ctrl+C to stop. Robot parks automatically on exit.
"""

import argparse
import builtins
import threading
import time
from pathlib import Path

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import TimedObservation
from lerobot.async_inference.robot_client import RobotClient
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101FollowerConfig

from cosmos_chessbot.vision.hud_overlay import (
    apply_hud,
    compute_homography,
    detect_corners,
)


class HUDRobotClient(RobotClient):
    """RobotClient with HUD overlay injection."""

    def __init__(self, config, source, target):
        super().__init__(config)
        self.source = source
        self.target = target
        self.hud_corners = None
        self.hud_H = None

    def control_loop_observation(self, task, verbose=False):
        """Inject HUD overlay on egocentric image before sending."""
        try:
            start_time = time.perf_counter()
            raw_observation = self.robot.get_observation()

            # Apply HUD to egocentric camera image
            if "egocentric" in raw_observation:
                img = raw_observation["egocentric"]
                if self.hud_corners is None:
                    self.hud_corners = detect_corners(img)
                    if self.hud_corners is not None:
                        self.hud_H = compute_homography(self.hud_corners)
                        print(f"Board corners detected, HUD active: {self.source} -> {self.target}")
                    else:
                        print("WARNING: could not detect board corners for HUD")
                if self.hud_corners is not None:
                    apply_hud(img, self.source, self.target,
                              corners=self.hud_corners, homography=self.hud_H)

            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                self.must_go.clear()

            if verbose:
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())
                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")


def park_robot(robot):
    """Smoothly return arm to park position."""
    PARK = {
        "shoulder_pan.pos": -80.0,
        "shoulder_lift.pos": -99.1,
        "elbow_flex.pos": 95.2,
        "wrist_flex.pos": 71.7,
        "wrist_roll.pos": -78.0,
        "gripper.pos": 3.1,
    }
    obs = robot.get_observation()
    current = {k: obs[k] for k in PARK}
    steps = 60
    print("Returning to park...")
    for i in range(1, steps + 1):
        t = i / steps
        action = {k: current[k] + t * (PARK[k] - current[k]) for k in PARK}
        robot.send_action(action)
        time.sleep(1.0 / 30)
    print("Parked")


def main():
    parser = argparse.ArgumentParser(description="Run pi0.5 episode via lerobot async inference")
    parser.add_argument("--port", required=True, help="Robot serial port")
    parser.add_argument("--source", required=True, help="Source square (e.g. e2)")
    parser.add_argument("--target", required=True, help="Target square (e.g. e4)")
    parser.add_argument("--server-address", default="localhost:8002", help="gRPC server address")
    parser.add_argument("--model-path",
                        default="outputs/pi05_chess_hud/checkpoints/012000/pretrained_model/",
                        help="Model path on server")
    parser.add_argument("--chunk-size-threshold", type=float, default=0.2,
                        help="Queue ratio threshold for sending observations (0-1)")
    parser.add_argument("--aggregate-fn", default="weighted_average",
                        choices=["weighted_average", "latest_only", "average", "conservative"])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--actions-per-chunk", type=int, default=50)
    parser.add_argument("--overhead-cam", type=int, default=1)
    parser.add_argument("--wrist-cam", type=int, default=0)
    parser.add_argument("--calibration-dir", default=str(
        Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower"))
    parser.add_argument("--debug-visualize", action="store_true",
                        help="Plot queue size after episode")
    args = parser.parse_args()

    task = "Move the small object from the green circle to the magenta circle"
    print(f"HUD: {args.source} -> {args.target}")
    print(f"Server: {args.server_address}, model: {args.model_path}")
    print(f"Async params: threshold={args.chunk_size_threshold}, "
          f"aggregate={args.aggregate_fn}, actions/chunk={args.actions_per_chunk}")

    # Bypass calibration prompt
    _real_input = builtins.input
    builtins.input = lambda *a, **kw: ""

    camera_config = {
        "egocentric": OpenCVCameraConfig(
            index_or_path=args.overhead_cam, width=640, height=480, fps=args.fps),
        "wrist": OpenCVCameraConfig(
            index_or_path=args.wrist_cam, width=640, height=480, fps=args.fps),
    }

    robot_config = SO101FollowerConfig(
        port=args.port,
        id="my_follower_arm",
        cameras=camera_config,
        calibration_dir=Path(args.calibration_dir),
    )

    client_config = RobotClientConfig(
        robot=robot_config,
        server_address=args.server_address,
        policy_type="pi05",
        pretrained_name_or_path=args.model_path,
        policy_device="cuda",
        client_device="cpu",
        actions_per_chunk=args.actions_per_chunk,
        chunk_size_threshold=args.chunk_size_threshold,
        aggregate_fn_name=args.aggregate_fn,
        fps=args.fps,
        task=task,
        debug_visualize_queue_size=args.debug_visualize,
    )

    builtins.input = _real_input

    print("Creating client...")
    client = HUDRobotClient(client_config, source=args.source, target=args.target)

    if client.start():
        print("Connected to policy server, starting episode...")
        print("Press Ctrl+C to stop")
        action_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_thread.start()

        try:
            client.control_loop(task=task)
        except KeyboardInterrupt:
            print("\nStopping...")
            client.shutdown_event.set()
        finally:
            try:
                park_robot(client.robot)
            except Exception as e:
                print(f"Park failed: {e}")

            client.stop()
            action_thread.join(timeout=5)

            if args.debug_visualize and client.action_queue_size:
                from lerobot.async_inference.helpers import visualize_action_queue_size
                visualize_action_queue_size(client.action_queue_size)

        print("Done")
    else:
        print("Failed to connect to policy server")
        client.robot.disconnect()


if __name__ == "__main__":
    main()
