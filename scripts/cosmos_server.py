#!/usr/bin/env python3
"""Remote Cosmos inference server.

Run this on your GPU server (H100/GB200) to serve Cosmos-Reason2 inference
for both perception AND reasoning. The local orchestrator connects via HTTP.

Endpoints:
    GET  /health              — Health check
    POST /perceive            — Board state perception (FEN extraction)
    POST /reason/action       — Pre-action reasoning (obstacles, grasp strategy)
    POST /reason/trajectory   — Action CoT trajectory planning (2D pixel waypoints)
    POST /reason/verify_goal  — Post-action visual goal verification
    POST /reason/analyze_game — Turn detection from video frames
    POST /reason/detect_move  — Opponent move detection from video frames
    POST /reason/correction   — Post-failure correction planning
"""

import argparse
import base64
import io
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from cosmos_chessbot.vision import CosmosPerception
from cosmos_chessbot.reasoning import ChessGameReasoning


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class InferenceRequest(BaseModel):
    """Request for perception inference."""
    image_base64: str
    max_new_tokens: int = 2048
    temperature: float = 0.1


class InferenceResponse(BaseModel):
    """Response from perception inference."""
    fen: str
    confidence: float
    anomalies: list[str]
    raw_response: str


class ActionReasoningRequest(BaseModel):
    """Request for action reasoning."""
    image_base64: str
    move_uci: str
    from_square: str
    to_square: str
    max_new_tokens: int = 512
    temperature: float = 0.1
    wrist_image_base64: Optional[str] = None


class VideoReasoningRequest(BaseModel):
    """Request for video-based reasoning (turn/move detection)."""
    frames_base64: list[str]
    max_new_tokens: int = 512
    temperature: float = 0.1


class TrajectoryRequest(BaseModel):
    """Request for trajectory planning (Action CoT)."""
    image_base64: str
    move_uci: str
    from_square: str
    to_square: str
    piece_type: str = "piece"
    max_new_tokens: int = 1024
    temperature: float = 0.1
    wrist_image_base64: Optional[str] = None


class GoalVerificationRequest(BaseModel):
    """Request for post-action goal verification."""
    image_base64: str
    move_uci: str
    from_square: str
    to_square: str
    piece_type: str = "piece"
    max_new_tokens: int = 512
    temperature: float = 0.1
    wrist_image_base64: Optional[str] = None


class CorrectionRequest(BaseModel):
    """Request for correction planning."""
    image_base64: str
    expected_fen: str
    actual_fen: str
    differences: list[str]
    max_new_tokens: int = 512
    temperature: float = 0.1
    wrist_image_base64: Optional[str] = None


class EpisodeCritiqueRequest(BaseModel):
    """Request for episode video critique (RL critic)."""
    frames_base64: list[str]
    from_square: str
    to_square: str
    piece_type: str = "piece"
    max_new_tokens: int = 1024
    temperature: float = 0.1


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Cosmos Inference Server")
perception: Optional[CosmosPerception] = None
reasoning: Optional[ChessGameReasoning] = None


def _decode_image(b64: str) -> Image.Image:
    """Decode a base64 string to a PIL Image."""
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _decode_images(b64_list: list[str]) -> list[Image.Image]:
    """Decode a list of base64 strings to PIL Images."""
    return [_decode_image(b) for b in b64_list]


@app.on_event("startup")
async def load_model():
    """Load Cosmos model on startup."""
    global perception, reasoning
    print("Loading Cosmos-Reason2 model (perception)...")
    perception = CosmosPerception()
    print("Loading Cosmos-Reason2 model (reasoning)...")
    reasoning = ChessGameReasoning()
    print("Models loaded and ready!")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "perception_loaded": perception is not None,
        "reasoning_loaded": reasoning is not None,
    }


# ---------------------------------------------------------------------------
# Perception endpoint
# ---------------------------------------------------------------------------

@app.post("/perceive", response_model=InferenceResponse)
async def perceive(request: InferenceRequest):
    """Run perception on an image."""
    if perception is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        image = _decode_image(request.image_base64)
        board_state = perception.perceive(
            image,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
        return InferenceResponse(
            fen=board_state.fen,
            confidence=board_state.confidence,
            anomalies=board_state.anomalies,
            raw_response=board_state.raw_response,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Perception failed: {e}")


# ---------------------------------------------------------------------------
# Reasoning endpoints
# ---------------------------------------------------------------------------

@app.post("/reason/action")
async def reason_action(request: ActionReasoningRequest):
    """Reason about how to physically execute a chess move."""
    if reasoning is None:
        raise HTTPException(status_code=503, detail="Reasoning model not loaded")

    try:
        image = _decode_image(request.image_base64)
        wrist = _decode_image(request.wrist_image_base64) if request.wrist_image_base64 else None
        result = reasoning.reason_about_action(
            image=image,
            move_uci=request.move_uci,
            from_square=request.from_square,
            to_square=request.to_square,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            wrist_image=wrist,
        )
        return {
            "obstacles": result.obstacles,
            "adjacent_pieces": result.adjacent_pieces,
            "grasp_strategy": result.grasp_strategy,
            "trajectory_advice": result.trajectory_advice,
            "risks": result.risks,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Action reasoning failed: {e}")


@app.post("/reason/trajectory")
async def reason_trajectory(request: TrajectoryRequest):
    """Plan a 2D pixel-space trajectory for a chess move (Action CoT)."""
    if reasoning is None:
        raise HTTPException(status_code=503, detail="Reasoning model not loaded")

    try:
        image = _decode_image(request.image_base64)
        wrist = _decode_image(request.wrist_image_base64) if request.wrist_image_base64 else None
        result = reasoning.plan_trajectory(
            image=image,
            move_uci=request.move_uci,
            from_square=request.from_square,
            to_square=request.to_square,
            piece_type=request.piece_type,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            wrist_image=wrist,
        )
        return {
            "waypoints": [
                {"point_2d": list(wp.point_2d), "label": wp.label}
                for wp in result.waypoints
            ],
            "move_uci": result.move_uci,
            "reasoning": result.reasoning,
            "confidence": result.confidence,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trajectory planning failed: {e}")


@app.post("/reason/verify_goal")
async def reason_verify_goal(request: GoalVerificationRequest):
    """Verify physical outcome of a chess move from post-action image."""
    if reasoning is None:
        raise HTTPException(status_code=503, detail="Reasoning model not loaded")

    try:
        image = _decode_image(request.image_base64)
        wrist = _decode_image(request.wrist_image_base64) if request.wrist_image_base64 else None
        result = reasoning.verify_goal(
            image=image,
            move_uci=request.move_uci,
            from_square=request.from_square,
            to_square=request.to_square,
            piece_type=request.piece_type,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            wrist_image=wrist,
        )
        return {
            "success": result.success,
            "reason": result.reason,
            "physical_issues": result.physical_issues,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Goal verification failed: {e}")


@app.post("/reason/analyze_game")
async def reason_analyze_game(request: VideoReasoningRequest):
    """Analyze game state from video frames (turn detection)."""
    if reasoning is None:
        raise HTTPException(status_code=503, detail="Reasoning model not loaded")

    try:
        frames = _decode_images(request.frames_base64)
        result = reasoning.analyze_game_state(
            video_frames=frames,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
        return {
            "whose_turn": result.whose_turn.value,
            "opponent_moving": result.opponent_moving,
            "should_robot_act": result.should_robot_act,
            "reasoning": result.reasoning,
            "confidence": result.confidence,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Game analysis failed: {e}")


@app.post("/reason/detect_move")
async def reason_detect_move(request: VideoReasoningRequest):
    """Detect what move the opponent made from video frames."""
    if reasoning is None:
        raise HTTPException(status_code=503, detail="Reasoning model not loaded")

    try:
        frames = _decode_images(request.frames_base64)
        result = reasoning.detect_move(
            video_frames=frames,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
        return {
            "move_occurred": result.move_occurred,
            "from_square": result.from_square,
            "to_square": result.to_square,
            "piece_type": result.piece_type,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Move detection failed: {e}")


@app.post("/reason/correction")
async def reason_correction(request: CorrectionRequest):
    """Plan correction for a failed move execution."""
    if reasoning is None:
        raise HTTPException(status_code=503, detail="Reasoning model not loaded")

    try:
        image = _decode_image(request.image_base64)
        wrist = _decode_image(request.wrist_image_base64) if request.wrist_image_base64 else None
        result = reasoning.plan_correction(
            image=image,
            expected_fen=request.expected_fen,
            actual_fen=request.actual_fen,
            differences=request.differences,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            wrist_image=wrist,
        )
        return {
            "physical_cause": result.physical_cause,
            "correction_needed": result.correction_needed,
            "obstacles": result.obstacles,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Correction planning failed: {e}")


@app.post("/reason/critique_episode")
async def reason_critique_episode(request: EpisodeCritiqueRequest):
    """Critique a full RL episode from video frames."""
    if reasoning is None:
        raise HTTPException(status_code=503, detail="Reasoning model not loaded")

    try:
        frames = _decode_images(request.frames_base64)
        result = reasoning.critique_episode(
            video_frames=frames,
            from_square=request.from_square,
            to_square=request.to_square,
            piece_type=request.piece_type,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
        return {
            "overall_score": result.overall_score,
            "success": result.success,
            "approach_safe": result.approach_safe,
            "grasp_stable": result.grasp_stable,
            "trajectory_safe": result.trajectory_safe,
            "placement_stable": result.placement_stable,
            "physical_issues": result.physical_issues,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Episode critique failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Cosmos inference server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    args = parser.parse_args()

    print(f"Starting Cosmos inference server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
