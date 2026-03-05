#!/usr/bin/env python3
"""Run a pi0.5 episode on the real robot via remote inference server.

Uses async inference with RTC (Real-Time Chunking) for smooth, continuous
motion. Two threads decouple action prediction from execution:
  - Inference thread: captures observations, sends to server, updates queue
  - Execution thread: pulls actions from queue, sends to robot at target fps

The robot never idles waiting for inference. RTC guides each new action chunk
to smoothly continue from the previous one, eliminating jerky transitions.

Usage:
    # Bowl pick-and-place:
    uv run python scripts/hardware/run_pi05_episode.py \
        --port /dev/tty.usbmodem58FA0962531 \
        --task "Pick up the piece and place it in the bowl"

    # With HUD overlay:
    uv run python scripts/hardware/run_pi05_episode.py \
        --port /dev/tty.usbmodem58FA0962531 \
        --source e2 --target e4

    # Sequential mode (no async, for debugging):
    uv run python scripts/hardware/run_pi05_episode.py \
        --port /dev/tty.usbmodem58FA0962531 --sequential
"""

import argparse
import math
import time
from collections import deque
from pathlib import Path
from threading import Event, Lock, Thread

import msgpack
import numpy as np
import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


# ---------------------------------------------------------------------------
# msgpack-numpy helpers
# ---------------------------------------------------------------------------

def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


# ---------------------------------------------------------------------------
# Thread-safe robot wrapper (prevents serial port contention)
# ---------------------------------------------------------------------------

class RobotWrapper:
    """Wraps robot with a Lock for safe access from inference + execution threads."""

    def __init__(self, robot):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self):
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action_dict):
        with self.lock:
            self.robot.send_action(action_dict)

    def disconnect(self):
        with self.lock:
            self.robot.disconnect()


# ---------------------------------------------------------------------------
# Action queue for async inference with RTC
# ---------------------------------------------------------------------------

class ActionQueue:
    """Thread-safe queue managing action chunks for async execution.

    Maintains both original (pre-postprocessing) actions for RTC guidance
    and processed actions for robot execution.

    Two merge modes (caller decides which to use):
    - replace(): RTC mode -- skip inference_delay actions, replace queue
    - append(): Non-RTC mode -- trim consumed, concatenate new chunk
    """

    def __init__(self):
        self.queue = None          # processed actions (N, action_dim) numpy
        self.original = None       # original actions for RTC (N, action_dim) numpy
        self.index = 0
        self.lock = Lock()

    def get(self):
        """Get next action for execution. Returns None if queue is empty."""
        with self.lock:
            if self.queue is None or self.index >= len(self.queue):
                return None
            action = self.queue[self.index].copy()
            self.index += 1
            return action

    def qsize(self):
        """Number of remaining actions."""
        if self.queue is None:
            return 0
        return max(0, len(self.queue) - self.index)

    def get_action_index(self):
        """Current consumption index."""
        return self.index

    def get_left_over(self):
        """Get unconsumed original actions for RTC prev_chunk_left_over."""
        with self.lock:
            if self.original is None:
                return None
            # Map interpolated queue index back to original action index
            # (original and queue may differ in length when slowdown > 1)
            if self.queue is not None and len(self.queue) > 0 and len(self.original) > 0:
                ratio = len(self.original) / len(self.queue)
                orig_idx = min(int(self.index * ratio), len(self.original))
            else:
                orig_idx = self.index
            left = self.original[orig_idx:]
            return left.copy() if len(left) > 0 else None

    def replace(self, original_actions, processed_actions, inference_delay,
                action_index_before_inference=None):
        """Replace queue with new chunk, skipping actions consumed during inference (RTC mode)."""
        with self.lock:
            skip = min(inference_delay, len(processed_actions))
            # Scale skip for original actions (may differ in length from processed
            # when slowdown interpolation is applied)
            if len(processed_actions) > 0 and len(original_actions) > 0:
                ratio = len(original_actions) / len(processed_actions)
                skip_original = min(int(skip * ratio), len(original_actions))
            else:
                skip_original = skip
            self.original = original_actions[skip_original:].copy()
            self.queue = processed_actions[skip:].copy()
            self.index = 0

    def append(self, original_actions, processed_actions):
        """Append new actions to queue (non-RTC mode). Preserves remaining actions."""
        with self.lock:
            if self.queue is None:
                self.original = original_actions.copy()
                self.queue = processed_actions.copy()
                self.index = 0
                return
            # Trim consumed, then append new chunk
            self.original = np.concatenate([self.original[self.index:], original_actions])
            self.queue = np.concatenate([self.queue[self.index:], processed_actions])
            self.index = 0


# ---------------------------------------------------------------------------
# Latency tracker
# ---------------------------------------------------------------------------

class LatencyTracker:
    """Track inference latency for computing RTC inference_delay."""

    def __init__(self, maxlen=10):
        self.values = deque(maxlen=maxlen)
        self._max = 0.0

    def add(self, latency):
        self.values.append(latency)
        self._max = max(self._max, latency)

    def max(self):
        return self._max

    def mean(self):
        """Rolling average of recent latencies (avoids cold-start inflation)."""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def p95(self):
        """95th percentile of recent latencies (conservative but not inflated)."""
        if not self.values:
            return 0.0
        sorted_vals = sorted(self.values)
        idx = min(int(len(sorted_vals) * 0.95), len(sorted_vals) - 1)
        return sorted_vals[idx]


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def capture_observation(robot_wrapper, log_first=False):
    """Capture joints and images from a single robot observation snapshot."""
    obs_raw = robot_wrapper.get_observation()

    if log_first:
        print(f"Observation keys: {list(obs_raw.keys())}")
        for k, v in obs_raw.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape} dtype={v.dtype}")

    # Extract joint positions
    joints = np.array([
        float(obs_raw[n].item() if hasattr(obs_raw[n], "item") else obs_raw[n])
        for n in JOINT_NAMES
    ], dtype=np.float32)

    # Extract images
    ego_key = next((k for k in obs_raw if "egocentric" in k), None)
    wrist_key = next((k for k in obs_raw if "wrist" in k and "pos" not in k), None)
    if ego_key is None:
        raise KeyError(f"No egocentric image in: {list(obs_raw.keys())}")

    overhead = np.array(obs_raw[ego_key], dtype=np.uint8)
    if overhead.ndim == 3 and overhead.shape[0] == 3:
        overhead = np.transpose(overhead, (1, 2, 0))

    if wrist_key and obs_raw[wrist_key] is not None:
        wrist = np.array(obs_raw[wrist_key], dtype=np.uint8)
        if wrist.ndim == 3 and wrist.shape[0] == 3:
            wrist = np.transpose(wrist, (1, 2, 0))
        if wrist.ndim < 2:
            wrist = np.zeros_like(overhead)
    else:
        wrist = np.zeros_like(overhead)

    return joints, overhead, wrist


def send_joints(robot_wrapper, targets):
    """Send joint targets to robot."""
    action_dict = {}
    for i, name in enumerate(JOINT_NAMES):
        if i < len(targets):
            action_dict[name] = torch.tensor([float(targets[i])], dtype=torch.float32)
    robot_wrapper.send_action(action_dict)


# ---------------------------------------------------------------------------
# Server communication
# ---------------------------------------------------------------------------

def interpolate_chunk(chunk, factor):
    """Interpolate between consecutive actions to slow down motion.

    A factor of 2 inserts one intermediate step between each pair,
    doubling the chunk length (and halving effective speed at same fps).
    """
    if factor <= 1:
        return chunk
    n, dim = chunk.shape
    indices = np.arange(n)
    new_indices = np.linspace(0, n - 1, (n - 1) * factor + 1)
    result = np.zeros((len(new_indices), dim), dtype=chunk.dtype)
    for d in range(dim):
        result[:, d] = np.interp(new_indices, indices, chunk[:, d])
    return result


def request_chunk(ws, packer, obs):
    """Send observation to server, receive action chunk and original actions."""
    ws.send(packer.pack(obs))
    response = ws.recv()
    result = msgpack.unpackb(response, object_hook=_unpack_array)

    chunk = result.get("action_chunk")
    if chunk is None:
        chunk = result.get("action")
    chunk = np.array(chunk, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk.reshape(1, -1)

    # Original (pre-postprocessing) actions for RTC
    original = result.get("original_actions")
    if original is not None:
        original = np.array(original, dtype=np.float32)
        if original.ndim == 1:
            original = original.reshape(1, -1)
    else:
        original = chunk.copy()  # fallback if server doesn't return originals

    return chunk, original, result.get("inference_time", 0)


# ---------------------------------------------------------------------------
# Inference thread
# ---------------------------------------------------------------------------

def inference_loop(robot_wrapper, ws, packer, action_queue, latency_tracker,
                   shutdown_event, fps, use_hud, hud_args, task, slowdown=1,
                   no_rtc=False, queue_threshold=30):
    """Background thread: capture observations, request chunks, update queue."""
    time_per_action = 1.0 / fps
    hud_corners = None
    hud_H = None
    chunk_count = 0
    first_obs = True

    # Non-RTC: trigger only when queue empty (like reference), unless overridden
    effective_threshold = 0 if no_rtc and queue_threshold == 30 else queue_threshold

    try:
        while not shutdown_event.is_set():
            if action_queue.qsize() <= effective_threshold:
                t0 = time.perf_counter()
                action_index_before = action_queue.get_action_index()
                prev_left_over = action_queue.get_left_over()

                # Compute inference delay from tracked latency (p95 is a
                # conservative estimate without cold-start inflation)
                latency_est = latency_tracker.p95()
                inference_delay = math.ceil(latency_est / time_per_action) if latency_est > 0 else 0

                # Capture observation (thread-safe via RobotWrapper lock)
                joints, overhead, wrist = capture_observation(robot_wrapper, log_first=first_obs)
                first_obs = False

                # Apply HUD overlay if needed
                if use_hud:
                    from cosmos_chessbot.vision.hud_overlay import (
                        apply_hud, compute_homography, detect_corners,
                    )
                    if hud_corners is None:
                        hud_corners = detect_corners(overhead)
                        if hud_corners is not None:
                            hud_H = compute_homography(hud_corners)
                            print("HUD: detected board corners")
                        else:
                            print("HUD: WARNING - could not detect board corners")
                    apply_hud(overhead, hud_args["source"], hud_args["target"],
                              hud_corners, hud_H)

                    # Save debug frame with board grid on first HUD application
                    if chunk_count == 0:
                        import cv2 as _cv2
                        from cosmos_chessbot.vision.hud_overlay import resolve_location
                        debug_img = overhead.copy()
                        if hud_corners is not None:
                            labels = ["TL(a8)", "TR(h8)", "BR(h1)", "BL(a1)"]
                            for ci, (cx, cy) in enumerate(hud_corners):
                                pt = (int(cx), int(cy))
                                _cv2.circle(debug_img, pt, 6, (0, 0, 255), -1, _cv2.LINE_AA)
                                _cv2.putText(debug_img, labels[ci], (pt[0]+8, pt[1]-4),
                                             _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                            for i in range(4):
                                p1 = (int(hud_corners[i][0]), int(hud_corners[i][1]))
                                p2 = (int(hud_corners[(i+1)%4][0]), int(hud_corners[(i+1)%4][1]))
                                _cv2.line(debug_img, p1, p2, (0, 0, 255), 2, _cv2.LINE_AA)
                            for file in "abcdefgh":
                                for rank in "12345678":
                                    sq = f"{file}{rank}"
                                    px = resolve_location(sq, hud_corners, hud_H)
                                    _cv2.circle(debug_img, px, 3, (0, 255, 255), -1)
                                    _cv2.putText(debug_img, sq, (px[0]+4, px[1]-2),
                                                 _cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1)
                        _cv2.imwrite("/tmp/hud_debug.png", debug_img)
                        _cv2.imwrite("/tmp/hud_policy_frame.png", overhead)
                        print("HUD: saved debug frame to /tmp/hud_debug.png")

                # Save policy frames periodically
                import cv2 as _cv2
                if chunk_count < 5 or chunk_count % 10 == 0:
                    _cv2.imwrite(f"/tmp/policy_frame_{chunk_count:03d}.png", overhead)
                    _cv2.imwrite(f"/tmp/policy_wrist_{chunk_count:03d}.png", wrist)

                obs = {
                    "observation.images.egocentric": overhead,
                    "observation.images.wrist": wrist,
                    "observation.state": joints,
                    "task": task,
                }

                # Add RTC params if we have prior chunk data
                if not no_rtc and prev_left_over is not None:
                    obs["prev_chunk_left_over"] = prev_left_over
                    obs["inference_delay"] = inference_delay

                chunk, original, inf_time = request_chunk(ws, packer, obs)
                chunk_count += 1

                # Interpolate to slow motion (don't interpolate originals -- RTC
                # needs raw policy outputs for guidance)
                chunk = interpolate_chunk(chunk, slowdown)

                new_latency = time.perf_counter() - t0
                new_delay = math.ceil(new_latency / time_per_action)
                latency_tracker.add(new_latency)

                # Delay validation (like reference _check_delays)
                consumed = action_queue.get_action_index() - action_index_before
                if chunk_count > 1 and consumed != new_delay:
                    print(f"  Delay mismatch: predicted={new_delay}, actual={consumed}")

                if no_rtc:
                    action_queue.append(original, chunk)
                else:
                    action_queue.replace(original, chunk, new_delay, action_index_before)

                if chunk_count <= 3 or chunk_count % 10 == 0:
                    print(f"  Chunk {chunk_count}: {chunk.shape}, "
                          f"latency={new_latency:.2f}s, delay={new_delay}, "
                          f"mode={'append' if no_rtc else 'rtc-replace'}, "
                          f"queue={action_queue.qsize()}")
            else:
                time.sleep(0.01)

    except Exception as e:
        print(f"Inference thread error: {e}")
        import traceback
        traceback.print_exc()
        shutdown_event.set()

    print(f"Inference thread done: {chunk_count} chunks")


# ---------------------------------------------------------------------------
# Execution thread
# ---------------------------------------------------------------------------

def execution_loop(robot_wrapper, action_queue, shutdown_event, fps,
                   max_steps, step_counter, dry_run):
    """Pull actions from queue and send to robot at target fps."""
    action_interval = 1.0 / fps

    try:
        while not shutdown_event.is_set():
            if step_counter[0] >= max_steps:
                shutdown_event.set()
                break

            t0 = time.perf_counter()
            action = action_queue.get()

            if action is not None:
                if not dry_run:
                    send_joints(robot_wrapper, action)
                step_counter[0] += 1

                if step_counter[0] <= 5 or step_counter[0] % 100 == 0:
                    print(f"Step {step_counter[0]:4d}: "
                          f"[{action[0]:6.1f} {action[1]:7.1f} {action[2]:6.1f} "
                          f"{action[3]:6.1f} {action[4]:6.1f} {action[5]:5.1f}] "
                          f"q={action_queue.qsize()}")

            dt = time.perf_counter() - t0
            time.sleep(max(0, action_interval - dt - 0.001))

    except Exception as e:
        print(f"Execution thread error: {e}")
        import traceback
        traceback.print_exc()
        shutdown_event.set()

    print(f"Execution thread done: {step_counter[0]} steps")


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_async(robot_wrapper, ws, packer, args, use_hud, task, metadata):
    """Run with async inference + RTC."""
    action_queue = ActionQueue()
    latency_tracker = LatencyTracker()
    shutdown_event = Event()
    step_counter = [0]  # mutable for thread access

    hud_args = {"source": args.source, "target": args.target}

    inf_thread = Thread(
        target=inference_loop,
        args=(robot_wrapper, ws, packer, action_queue, latency_tracker,
              shutdown_event, args.fps, use_hud, hud_args, task, args.slowdown,
              args.no_rtc, args.queue_threshold),
        daemon=True, name="Inference",
    )
    exec_thread = Thread(
        target=execution_loop,
        args=(robot_wrapper, action_queue, shutdown_event, args.fps,
              args.num_steps, step_counter, args.dry_run),
        daemon=True, name="Execution",
    )

    rtc_server = metadata.get(b"rtc_enabled") or metadata.get("rtc_enabled")
    if args.no_rtc:
        rtc_label = "disabled (append mode)"
    else:
        rtc_label = "on" if rtc_server else "off on server"
    effective_threshold = 0 if args.no_rtc and args.queue_threshold == 30 else args.queue_threshold
    print(f"Mode: async inference (RTC {rtc_label}, threshold={effective_threshold})")
    print("-" * 60)
    inf_thread.start()
    exec_thread.start()

    try:
        while not shutdown_event.is_set():
            time.sleep(2)
            print(f"  Status: step={step_counter[0]}/{args.num_steps}, "
                  f"queue={action_queue.qsize()}, "
                  f"latency={latency_tracker.max():.2f}s")
    except KeyboardInterrupt:
        print("\nInterrupted")

    shutdown_event.set()
    inf_thread.join(timeout=5)
    exec_thread.join(timeout=5)

    print(f"\n{'=' * 60}")
    print(f"Episode complete: {step_counter[0]} steps")


def run_sequential(robot_wrapper, ws, packer, args, use_hud, task):
    """Run with sequential chunking (fallback, no async)."""
    hud_corners = None
    hud_H = None
    total_steps = 0
    chunk_count = 0
    step_delay = 1.0 / args.fps
    first_obs = True

    print("Mode: sequential chunking (no async)")
    print("-" * 60)

    try:
        while total_steps < args.num_steps:
            joints, overhead, wrist = capture_observation(robot_wrapper, log_first=first_obs)
            first_obs = False

            if use_hud:
                from cosmos_chessbot.vision.hud_overlay import (
                    apply_hud, compute_homography, detect_corners,
                )
                # Save raw frame before any HUD modification
                if hud_corners is None:
                    import cv2 as _cv2
                    _cv2.imwrite("/tmp/hud_raw_frame.png", overhead.copy())
                    print("HUD: saved raw frame to /tmp/hud_raw_frame.png")
                if hud_corners is None:
                    hud_corners = detect_corners(overhead)
                    if hud_corners is not None:
                        hud_H = compute_homography(hud_corners)
                        print("HUD: detected board corners")
                    else:
                        print("HUD: WARNING - could not detect board corners")
                apply_hud(overhead, args.source, args.target, hud_corners, hud_H)

                # Save debug frame with board grid on first HUD application
                if chunk_count == 0:
                    import cv2 as _cv2
                    from cosmos_chessbot.vision.hud_overlay import resolve_location
                    debug_img = overhead.copy()
                    # Draw detected corners
                    if hud_corners is not None:
                        labels = ["TL(a8)", "TR(h8)", "BR(h1)", "BL(a1)"]
                        for ci, (cx, cy) in enumerate(hud_corners):
                            pt = (int(cx), int(cy))
                            _cv2.circle(debug_img, pt, 6, (0, 0, 255), -1, _cv2.LINE_AA)
                            _cv2.putText(debug_img, labels[ci], (pt[0]+8, pt[1]-4),
                                         _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        # Draw board outline
                        for i in range(4):
                            p1 = (int(hud_corners[i][0]), int(hud_corners[i][1]))
                            p2 = (int(hud_corners[(i+1)%4][0]), int(hud_corners[(i+1)%4][1]))
                            _cv2.line(debug_img, p1, p2, (0, 0, 255), 2, _cv2.LINE_AA)
                        # Draw all 64 square centers
                        for file in "abcdefgh":
                            for rank in "12345678":
                                sq = f"{file}{rank}"
                                px = resolve_location(sq, hud_corners, hud_H)
                                _cv2.circle(debug_img, px, 3, (0, 255, 255), -1)
                                _cv2.putText(debug_img, sq, (px[0]+4, px[1]-2),
                                             _cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1)
                    _cv2.imwrite("/tmp/hud_debug.png", debug_img)
                    # Also save the actual frame being sent to the policy
                    _cv2.imwrite("/tmp/hud_policy_frame.png", overhead)
                    print("HUD: saved debug frame to /tmp/hud_debug.png")

            obs = {
                "observation.images.egocentric": overhead,
                "observation.images.wrist": wrist,
                "observation.state": joints,
                "task": task,
            }

            chunk, _, inf_time = request_chunk(ws, packer, obs)
            chunk = interpolate_chunk(chunk, args.slowdown)
            chunk_count += 1

            if chunk_count <= 2:
                print(f"Chunk {chunk_count}: {chunk.shape}, inference: {inf_time:.3f}s")

            for i in range(len(chunk)):
                if total_steps >= args.num_steps:
                    break
                action = chunk[i]
                if not args.dry_run:
                    send_joints(robot_wrapper, action)
                if total_steps < 5 or total_steps % 50 == 0:
                    print(f"Step {total_steps:4d} (chunk {chunk_count}, {i}/{len(chunk)}): "
                          f"[{action[0]:6.1f} {action[1]:7.1f} {action[2]:6.1f} "
                          f"{action[3]:6.1f} {action[4]:6.1f} {action[5]:5.1f}]")
                total_steps += 1
                time.sleep(step_delay)

            print(f"  Chunk {chunk_count}: {len(chunk)} actions")

    except KeyboardInterrupt:
        print("\nInterrupted")

    print(f"\n{'=' * 60}")
    print(f"Episode complete: {total_steps} steps, {chunk_count} chunks")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run pi0.5 episode on real robot")
    parser.add_argument("--port", type=str, default="/dev/tty.usbmodem58FA0962531")
    parser.add_argument("--calibration-dir", type=str,
                        default="/Users/max/.cache/huggingface/lerobot/calibration/robots/so101_follower")
    parser.add_argument("--server-url", type=str, default="ws://localhost:8001")
    parser.add_argument("--task", type=str, default="Pick up the piece and place it in the bowl")
    parser.add_argument("--num-steps", type=int, default=1000,
                        help="Max action steps to execute")
    parser.add_argument("--fps", type=float, default=30, help="Action execution rate")
    parser.add_argument("--source", type=str, default=None, help="HUD source square (e.g. e2)")
    parser.add_argument("--target", type=str, default=None, help="HUD target square (e.g. e4)")
    parser.add_argument("--slowdown", type=int, default=1,
                        help="Interpolation factor to slow motion (2=half speed, 3=third)")
    parser.add_argument("--dry-run", action="store_true", help="Inference only, don't execute")
    parser.add_argument("--no-rtc", action="store_true",
                        help="Disable sending RTC guidance to server")
    parser.add_argument("--queue-threshold", type=int, default=30,
                        help="Request new chunk when queue drops to this size (default 30, "
                             "non-RTC defaults to 0 unless overridden)")
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential chunking (no async, for debugging)")
    parser.add_argument("--overhead-cam", type=int, default=1)
    parser.add_argument("--wrist-cam", type=int, default=0)
    args = parser.parse_args()

    use_hud = args.source and args.target
    task = args.task
    if use_hud:
        task = "Move the small object from the green circle to the magenta circle"
        print(f"HUD enabled: source={args.source} target={args.target}")

    # Corner detection happens from the first observation frame after robot
    # connection -- same pattern as collect_episodes.py during data collection.
    # The arm should already be at park pose (clear view of the board).

    # Connect robot (patch out interactive calibration prompt)
    import builtins
    _real_input = builtins.input
    builtins.input = lambda *a, **kw: ""
    print("Connecting robot...")
    camera_config = {
        "egocentric": OpenCVCameraConfig(
            index_or_path=args.overhead_cam, width=640, height=480, fps=30),
        "wrist": OpenCVCameraConfig(
            index_or_path=args.wrist_cam, width=640, height=480, fps=30),
    }
    robot = SO101Follower(SO101FollowerConfig(
        port=args.port,
        id="my_follower_arm",
        cameras=camera_config,
        calibration_dir=Path(args.calibration_dir),
    ))
    robot.connect()
    builtins.input = _real_input
    robot_wrapper = RobotWrapper(robot)
    print("Robot connected")

    # Show initial state (arm should already be at park)
    joints, _, _ = capture_observation(robot_wrapper, log_first=True)
    print(f"Initial joints: pan={joints[0]:.1f} lift={joints[1]:.1f} elbow={joints[2]:.1f} "
          f"wrist={joints[3]:.1f} roll={joints[4]:.1f} grip={joints[5]:.1f}")

    # Connect to pi0.5 server
    import websockets.sync.client
    print(f"Connecting to pi0.5 server at {args.server_url}...")
    ws = websockets.sync.client.connect(args.server_url, compression=None, max_size=None)
    packer = msgpack.Packer(default=_pack_array)
    metadata = msgpack.unpackb(ws.recv(), object_hook=_unpack_array)
    print(f"Server: {metadata}")

    # Reset policy
    ws.send(packer.pack({"command": "reset"}))
    reset_resp = msgpack.unpackb(ws.recv(), object_hook=_unpack_array)
    print(f"Policy reset: {reset_resp}")

    print(f"\nRunning episode: {task}")
    print(f"Max steps: {args.num_steps}, fps: {args.fps}, dry_run: {args.dry_run}")

    if args.sequential:
        run_sequential(robot_wrapper, ws, packer, args, use_hud, task)
    else:
        run_async(robot_wrapper, ws, packer, args, use_hud, task, metadata)

    ws.close()

    # Return to park position
    PARK = [-80.0, -99.1, 95.2, 71.7, -78.0, 3.1]
    print("Returning to park...")
    current, _, _ = capture_observation(robot_wrapper)
    steps = 60  # ~2s at 30fps
    for i in range(1, steps + 1):
        t = i / steps
        interp = [current[j] + t * (PARK[j] - current[j]) for j in range(6)]
        send_joints(robot_wrapper, interp)
        time.sleep(1.0 / 30)
    print("Parked")

    robot_wrapper.disconnect()
    print("Done")


if __name__ == "__main__":
    main()
