"""Pydantic schemas for Cosmos server request/response types.

Shared between the server (``scripts/cosmos_server.py``) and the remote
client (``reasoning/remote_reasoning.py``) so both sides agree on the
wire format.
"""

from pydantic import BaseModel
from typing import Optional


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------

class PerceptionRequest(BaseModel):
    """Request for board-state perception."""
    image_base64: str
    max_new_tokens: int = 2048
    temperature: float = 0.1


class PerceptionResponse(BaseModel):
    """Response from board-state perception."""
    fen: str
    confidence: float
    anomalies: list[str]
    raw_response: str


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------

class ActionReasoningRequest(BaseModel):
    """Request for pre-action physical reasoning."""
    image_base64: str
    move_uci: str
    from_square: str
    to_square: str
    max_new_tokens: int = 512
    temperature: float = 0.1


class ActionReasoningResponse(BaseModel):
    """Response from pre-action reasoning."""
    obstacles: list[str]
    adjacent_pieces: list[str]
    grasp_strategy: str
    trajectory_advice: str
    risks: list[str]
    confidence: float
    reasoning: str


class VideoReasoningRequest(BaseModel):
    """Request for video-based reasoning (turn/move detection)."""
    frames_base64: list[str]
    max_new_tokens: int = 512
    temperature: float = 0.1


class GameStateResponse(BaseModel):
    """Response from game state analysis."""
    whose_turn: str
    opponent_moving: bool
    should_robot_act: bool
    reasoning: str
    confidence: float


class MoveDetectionResponse(BaseModel):
    """Response from move detection."""
    move_occurred: bool
    from_square: Optional[str] = None
    to_square: Optional[str] = None
    piece_type: Optional[str] = None
    confidence: float
    reasoning: str


class CorrectionRequest(BaseModel):
    """Request for correction planning."""
    image_base64: str
    expected_fen: str
    actual_fen: str
    differences: list[str]
    max_new_tokens: int = 512
    temperature: float = 0.1


class CorrectionResponse(BaseModel):
    """Response from correction planning."""
    physical_cause: str
    correction_needed: str
    obstacles: list[str]
    confidence: float
    reasoning: str


# ---------------------------------------------------------------------------
# Trajectory Planning (Action CoT)
# ---------------------------------------------------------------------------

class TrajectoryRequest(BaseModel):
    """Request for trajectory planning via Action CoT."""
    image_base64: str
    move_uci: str
    from_square: str
    to_square: str
    piece_type: str = "piece"
    max_new_tokens: int = 1024
    temperature: float = 0.1


class TrajectoryWaypointResponse(BaseModel):
    """A single waypoint in a trajectory."""
    point_2d: list[int]
    label: str


class TrajectoryPlanResponse(BaseModel):
    """Response from trajectory planning."""
    waypoints: list[TrajectoryWaypointResponse]
    move_uci: str
    reasoning: str
    confidence: float


# ---------------------------------------------------------------------------
# Goal Verification
# ---------------------------------------------------------------------------

class GoalVerificationRequest(BaseModel):
    """Request for post-action goal verification."""
    image_base64: str
    move_uci: str
    from_square: str
    to_square: str
    piece_type: str = "piece"
    max_new_tokens: int = 512
    temperature: float = 0.1


class GoalVerificationResponse(BaseModel):
    """Response from goal verification."""
    success: bool
    reason: str
    physical_issues: list[str]
    confidence: float
    reasoning: str


__all__ = [
    "PerceptionRequest",
    "PerceptionResponse",
    "ActionReasoningRequest",
    "ActionReasoningResponse",
    "VideoReasoningRequest",
    "GameStateResponse",
    "MoveDetectionResponse",
    "CorrectionRequest",
    "CorrectionResponse",
    "TrajectoryRequest",
    "TrajectoryWaypointResponse",
    "TrajectoryPlanResponse",
    "GoalVerificationRequest",
    "GoalVerificationResponse",
]
