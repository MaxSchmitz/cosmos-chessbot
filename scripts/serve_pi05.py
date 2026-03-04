#!/usr/bin/env python3
"""Serve fine-tuned pi0.5 for chess pick-and-place over WebSocket.

Supports Real-Time Chunking (RTC) for smooth async inference.

Usage:
    # On brev GPU server (with RTC):
    uv run python scripts/serve_pi05.py \
        --checkpoint models/pi05_chess_hud --port 8001 --rtc

    # Without RTC:
    uv run python scripts/serve_pi05.py \
        --checkpoint models/pi05_chess_hud --port 8001

    # SSH tunnel from local Mac:
    ssh -f -N -L 8001:localhost:8001 ubuntu@isaacsim
"""

import argparse
import asyncio
import functools
import logging
import time

import msgpack
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# msgpack-numpy helpers (same as client)
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


_packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cuda", rtc: bool = False):
    """Load fine-tuned pi0.5 with preprocessor/postprocessor."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors

    logger.info(f"Loading model from: {checkpoint_path}")
    policy = PI05Policy.from_pretrained(checkpoint_path)
    policy = policy.to(device).eval()

    # Enable RTC if requested (inference-time only, no training changes needed)
    if rtc:
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        from lerobot.configs.types import RTCAttentionSchedule
        policy.config.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=10,
            max_guidance_weight=10.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
        policy.init_rtc_processor()
        logger.info("RTC enabled: execution_horizon=10, max_guidance_weight=10.0, schedule=EXP")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=checkpoint_path,
    )

    logger.info(f"Model loaded on {device}. Max action dim: {policy.config.max_action_dim}")
    return policy, preprocessor, postprocessor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(policy, preprocessor, postprocessor, obs: dict, device: str = "cuda") -> dict:
    """Run one inference step. Returns action_chunk, original_actions, and timing."""
    t0 = time.time()

    # Convert msgpack-decoded obs to clean Python/numpy types
    clean_obs = {}
    for key, val in obs.items():
        # msgpack decodes string keys as bytes
        k = key.decode("utf-8") if isinstance(key, bytes) else key
        if isinstance(val, np.ndarray):
            clean_obs[k] = np.array(val, copy=True)  # fresh writable copy
        elif isinstance(val, bytes):
            clean_obs[k] = val.decode("utf-8")
        else:
            clean_obs[k] = val
    obs = clean_obs

    # Extract RTC params before observation preprocessing
    inference_delay = obs.pop("inference_delay", None)
    prev_chunk_left_over = obs.pop("prev_chunk_left_over", None)

    logger.info(f"Clean obs types: {[(k, type(v).__name__, v.shape if hasattr(v, 'shape') else '') for k, v in obs.items()]}")
    if inference_delay is not None:
        plco_shape = prev_chunk_left_over.shape if prev_chunk_left_over is not None else None
        logger.info(f"RTC params: inference_delay={inference_delay}, prev_chunk_left_over={plco_shape}")

    # Task must be a list of strings (not bare string) -- see lerobot reference
    if "task" in obs and isinstance(obs["task"], str):
        obs["task"] = [obs["task"]]

    # Add batch dimension to state if needed (preprocessor expects 2D)
    if "observation.state" in obs and obs["observation.state"].ndim == 1:
        obs["observation.state"] = obs["observation.state"].reshape(1, -1)

    # Convert images from HWC uint8 to CHW float32 [0,1] (lerobot dataset format)
    for key in list(obs.keys()):
        if "image" in key and isinstance(obs[key], np.ndarray) and obs[key].ndim == 3:
            img = obs[key]
            if img.shape[-1] in (1, 3, 4):  # HWC format
                img = np.transpose(img, (2, 0, 1))  # -> CHW
            obs[key] = img.astype(np.float32) / 255.0

    batch = preprocessor(obs)
    logger.info(f"Preprocessed batch keys: {list(batch.keys()) if isinstance(batch, dict) else type(batch)}")

    # Build RTC kwargs for predict_action_chunk
    rtc_kwargs = {}
    if inference_delay is not None:
        rtc_kwargs["inference_delay"] = int(inference_delay)
    if prev_chunk_left_over is not None:
        plco = torch.from_numpy(np.array(prev_chunk_left_over, dtype=np.float32)).to(device)
        # Don't add batch dim -- reference passes (T, action_dim) directly
        rtc_kwargs["prev_chunk_left_over"] = plco

    # Don't use torch.inference_mode() -- RTC's guidance step needs
    # torch.autograd.grad internally. The policy's own @torch.no_grad()
    # decorator handles the non-RTC path.
    action_chunk = policy.predict_action_chunk(batch, **rtc_kwargs)

    logger.info(f"Raw action_chunk type: {type(action_chunk)}, "
                f"shape: {action_chunk.shape if hasattr(action_chunk, 'shape') else 'N/A'}")

    # Save original (pre-postprocessing) actions for RTC on next call
    original_np = action_chunk.squeeze(0).cpu().numpy().astype(np.float32)

    action_chunk_np = postprocessor(action_chunk)
    if isinstance(action_chunk_np, torch.Tensor):
        action_chunk_np = action_chunk_np.cpu().numpy()
    action_chunk_np = np.asarray(action_chunk_np, dtype=np.float32)
    # Remove batch dim if present: (1, n_steps, action_dim) -> (n_steps, action_dim)
    if action_chunk_np.ndim == 3 and action_chunk_np.shape[0] == 1:
        action_chunk_np = action_chunk_np[0]
    elif action_chunk_np.ndim == 1:
        action_chunk_np = action_chunk_np.reshape(1, -1)

    dt = time.time() - t0
    first_action = action_chunk_np[0, :6]
    rtc_info = f", RTC delay={rtc_kwargs.get('inference_delay')}" if rtc_kwargs else ""
    logger.info(f"Inference: {dt:.3f}s, chunk shape: {action_chunk_np.shape}, "
                f"first: [{', '.join(f'{float(v):.1f}' for v in first_action)}]{rtc_info}")

    return {
        "action_chunk": action_chunk_np,
        "original_actions": original_np,
        "inference_time": dt,
    }


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------

async def handle_client(websocket, policy, preprocessor, postprocessor, device):
    """Handle a single WebSocket client connection."""
    packer = _packer()

    rtc_enabled = (policy.config.rtc_config is not None
                   and policy.config.rtc_config.enabled)

    # Send metadata on connect
    metadata = {
        "model": "pi05_chess_finetuned",
        "action_dim": 6,
        "chunk_size": 50,
        "rtc_enabled": rtc_enabled,
        "obs_keys": [
            "observation.state",
            "observation.images.egocentric",
            "observation.images.wrist",
            "task",
        ],
    }
    await websocket.send(packer.pack(metadata))
    logger.info(f"Client connected: {websocket.remote_address} (RTC={'on' if rtc_enabled else 'off'})")

    try:
        async for message in websocket:
            obs = _unpackb(message)

            # Log what we received
            keys = list(obs.keys()) if isinstance(obs, dict) else []
            logger.info(f"Received obs with keys: {keys}")

            # Handle reset command (msgpack encodes keys as bytes)
            cmd = obs.get("command") or obs.get(b"command")
            if cmd in ("reset", b"reset"):
                policy.reset()
                logger.info("Policy reset (action queue cleared)")
                await websocket.send(packer.pack({"status": "reset"}))
                continue

            # Run inference
            result = run_inference(policy, preprocessor, postprocessor, obs, device)
            await websocket.send(packer.pack(result))

    except Exception as e:
        import traceback
        logger.error(f"Client error: {e}\n{traceback.format_exc()}")
    finally:
        logger.info(f"Client disconnected: {websocket.remote_address}")


async def main(args):
    import websockets.asyncio.server

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, preprocessor, postprocessor = load_model(args.checkpoint, device, rtc=args.rtc)

    handler = functools.partial(
        handle_client,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=device,
    )

    logger.info(f"Starting WebSocket server on port {args.port}...")
    async with websockets.asyncio.server.serve(
        handler, "0.0.0.0", args.port,
        compression=None, max_size=None,
    ) as server:
        logger.info(f"Server ready on ws://0.0.0.0:{args.port}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve pi0.5 chess policy")
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained_model dir")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--rtc", action="store_true",
                        help="Enable Real-Time Chunking for smooth async inference")
    args = parser.parse_args()
    asyncio.run(main(args))
