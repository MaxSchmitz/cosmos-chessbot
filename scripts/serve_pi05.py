#!/usr/bin/env python3
"""Serve fine-tuned pi0.5 for chess pick-and-place over WebSocket.

Usage:
    # On brev GPU server:
    uv run python scripts/serve_pi05.py --checkpoint outputs/pi05_chess/checkpoints/last/pretrained_model --port 8001

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

def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load fine-tuned pi0.5 with preprocessor/postprocessor."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors

    logger.info(f"Loading model from: {checkpoint_path}")
    policy = PI05Policy.from_pretrained(checkpoint_path)
    policy = policy.to(device).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=checkpoint_path,
    )

    logger.info(f"Model loaded on {device}. Action dim: {policy.config.action_dim}")
    return policy, preprocessor, postprocessor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(policy, preprocessor, postprocessor, obs: dict) -> dict:
    """Run one inference step. Returns dict with 'action' array."""
    t0 = time.time()

    batch = preprocessor(obs)

    with torch.inference_mode():
        action = policy.select_action(batch)

    action_np = postprocessor(action)
    if isinstance(action_np, torch.Tensor):
        action_np = action_np.cpu().numpy()
    action_np = np.asarray(action_np, dtype=np.float32)

    dt = time.time() - t0
    logger.info(f"Inference: {dt:.3f}s, action shape: {action_np.shape}, "
                f"values: [{', '.join(f'{v:.1f}' for v in action_np[:6])}]")

    return {"action": action_np, "inference_time": dt}


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------

async def handle_client(websocket, policy, preprocessor, postprocessor):
    """Handle a single WebSocket client connection."""
    packer = _packer()

    # Send metadata on connect
    metadata = {
        "model": "pi05_chess_finetuned",
        "action_dim": 6,
        "chunk_size": 50,
        "obs_keys": [
            "observation.state",
            "observation.images.egocentric",
            "observation.images.wrist",
            "task",
        ],
    }
    await websocket.send(packer.pack(metadata))
    logger.info(f"Client connected: {websocket.remote_address}")

    try:
        async for message in websocket:
            obs = _unpackb(message)

            # Log what we received
            keys = list(obs.keys()) if isinstance(obs, dict) else []
            logger.info(f"Received obs with keys: {keys}")

            # Handle reset command
            if obs.get("command") == "reset":
                policy.reset()
                await websocket.send(packer.pack({"status": "reset"}))
                continue

            # Run inference
            result = run_inference(policy, preprocessor, postprocessor, obs)
            await websocket.send(packer.pack(result))

    except Exception as e:
        logger.error(f"Client error: {e}")
    finally:
        logger.info(f"Client disconnected: {websocket.remote_address}")


async def main(args):
    import websockets.asyncio.server

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, preprocessor, postprocessor = load_model(args.checkpoint, device)

    handler = functools.partial(
        handle_client,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
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
    args = parser.parse_args()
    asyncio.run(main(args))
