"""Remote Cosmos game reasoning client.

Mirrors the ``ChessGameReasoning`` interface but delegates inference
to the GPU server via HTTP, so the local machine (with robot hardware)
doesn't need a GPU for reasoning.
"""

import base64
import io
import logging
from typing import Optional

import httpx
from PIL import Image

from .game_reasoning import (
    ActionReasoning,
    CorrectionPlan,
    EpisodeCritique,
    GameState,
    GoalVerification,
    MoveDetection,
    TrajectoryPlan,
    TrajectoryWaypoint,
    Turn,
)

logger = logging.getLogger(__name__)


def _encode_image(image: Image.Image) -> str:
    """Encode a PIL Image to base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _encode_images(images: list[Image.Image]) -> list[str]:
    """Encode a list of PIL Images to base64 strings."""
    return [_encode_image(img) for img in images]


class RemoteChessGameReasoning:
    """Client for remote Cosmos game reasoning server.

    Drop-in replacement for ``ChessGameReasoning`` that forwards all
    calls to the remote server's ``/reason/*`` endpoints.
    """

    def __init__(self, server_url: str, timeout: float = 120.0):
        self.server_url = server_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def __del__(self):
        self.client.close()

    def analyze_game_state(
        self,
        video_frames: list[Image.Image],
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> GameState:
        """Analyze game state via remote server."""
        response = self.client.post(
            f"{self.server_url}/reason/analyze_game",
            json={
                "frames_base64": _encode_images(video_frames),
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        data = response.json()

        turn_str = data.get("whose_turn", "unknown").lower()
        whose_turn = Turn.ROBOT if turn_str == "robot" else (
            Turn.OPPONENT if turn_str == "opponent" else Turn.UNKNOWN
        )
        return GameState(
            whose_turn=whose_turn,
            opponent_moving=data.get("opponent_moving", False),
            should_robot_act=data.get("should_robot_act", False),
            reasoning=data.get("reasoning", ""),
            confidence=float(data.get("confidence", 0.0)),
        )

    def detect_move(
        self,
        video_frames: list[Image.Image],
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> MoveDetection:
        """Detect opponent move via remote server."""
        response = self.client.post(
            f"{self.server_url}/reason/detect_move",
            json={
                "frames_base64": _encode_images(video_frames),
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        data = response.json()

        return MoveDetection(
            move_occurred=data.get("move_occurred", False),
            from_square=data.get("from_square"),
            to_square=data.get("to_square"),
            piece_type=data.get("piece_type"),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=data.get("reasoning", ""),
        )

    def reason_about_action(
        self,
        image: Image.Image,
        move_uci: str,
        from_square: str,
        to_square: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> ActionReasoning:
        """Get action reasoning via remote server."""
        response = self.client.post(
            f"{self.server_url}/reason/action",
            json={
                "image_base64": _encode_image(image),
                "move_uci": move_uci,
                "from_square": from_square,
                "to_square": to_square,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        data = response.json()

        return ActionReasoning(
            obstacles=data.get("obstacles", []),
            adjacent_pieces=data.get("adjacent_pieces", []),
            grasp_strategy=data.get("grasp_strategy", ""),
            trajectory_advice=data.get("trajectory_advice", ""),
            risks=data.get("risks", []),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=data.get("reasoning", ""),
        )

    def plan_correction(
        self,
        image: Image.Image,
        expected_fen: str,
        actual_fen: str,
        differences: list,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> CorrectionPlan:
        """Get correction plan via remote server."""
        response = self.client.post(
            f"{self.server_url}/reason/correction",
            json={
                "image_base64": _encode_image(image),
                "expected_fen": expected_fen,
                "actual_fen": actual_fen,
                "differences": [str(d) for d in differences],
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        data = response.json()

        return CorrectionPlan(
            physical_cause=data.get("physical_cause", ""),
            correction_needed=data.get("correction_needed", ""),
            obstacles=data.get("obstacles", []),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=data.get("reasoning", ""),
        )

    def plan_trajectory(
        self,
        image: Image.Image,
        move_uci: str,
        from_square: str,
        to_square: str,
        piece_type: str = "piece",
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> TrajectoryPlan:
        """Plan trajectory via remote server."""
        response = self.client.post(
            f"{self.server_url}/reason/trajectory",
            json={
                "image_base64": _encode_image(image),
                "move_uci": move_uci,
                "from_square": from_square,
                "to_square": to_square,
                "piece_type": piece_type,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        data = response.json()

        waypoints = [
            TrajectoryWaypoint(
                point_2d=tuple(wp["point_2d"]),
                label=wp.get("label", ""),
            )
            for wp in data.get("waypoints", [])
        ]

        return TrajectoryPlan(
            waypoints=waypoints,
            move_uci=data.get("move_uci", move_uci),
            reasoning=data.get("reasoning", ""),
            confidence=float(data.get("confidence", 0.0)),
        )

    def verify_goal(
        self,
        image: Image.Image,
        move_uci: str,
        from_square: str,
        to_square: str,
        piece_type: str = "piece",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> GoalVerification:
        """Verify goal via remote server."""
        response = self.client.post(
            f"{self.server_url}/reason/verify_goal",
            json={
                "image_base64": _encode_image(image),
                "move_uci": move_uci,
                "from_square": from_square,
                "to_square": to_square,
                "piece_type": piece_type,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        data = response.json()

        return GoalVerification(
            success=data.get("success", False),
            reason=data.get("reason", ""),
            physical_issues=data.get("physical_issues", []),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=data.get("reasoning", ""),
        )

    def critique_episode(
        self,
        video_frames: list[Image.Image],
        from_square: str,
        to_square: str,
        piece_type: str = "piece",
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> EpisodeCritique:
        """Critique a full RL episode via remote server."""
        response = self.client.post(
            f"{self.server_url}/reason/critique_episode",
            json={
                "frames_base64": _encode_images(video_frames),
                "from_square": from_square,
                "to_square": to_square,
                "piece_type": piece_type,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        data = response.json()

        return EpisodeCritique(
            overall_score=float(data.get("overall_score", 0.0)),
            success=data.get("success", False),
            approach_safe=data.get("approach_safe", False),
            grasp_stable=data.get("grasp_stable", False),
            trajectory_safe=data.get("trajectory_safe", False),
            placement_stable=data.get("placement_stable", False),
            physical_issues=data.get("physical_issues", []),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=data.get("reasoning", ""),
        )
