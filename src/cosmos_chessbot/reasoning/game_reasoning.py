"""Chess game reasoning using Cosmos Reason2 for embodied AI.

This module uses Cosmos Reason2's embodied reasoning capabilities to understand
the chess game flow: whose turn it is, when moves are complete, and what actions
to take. This is the key differentiator from traditional chess AI.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import json
import logging
import re

from PIL import Image
import torch
import transformers

from ..utils import extract_json

logger = logging.getLogger(__name__)


def _extract_answer(text: str) -> str:
    """Extract content from <answer> tags if present, otherwise return full text."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def _extract_thinking(text: str) -> str | None:
    """Extract content from <think> tags if present."""
    match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL)
    if match:
        return match.group(1)
    return None


class Turn(Enum):
    """Whose turn it is."""
    ROBOT = "robot"
    OPPONENT = "opponent"
    UNKNOWN = "unknown"


@dataclass
class MoveDetection:
    """Result of move detection from video."""
    move_occurred: bool
    """Whether a move was detected."""

    from_square: Optional[str] = None
    """Starting square in algebraic notation (e.g., 'e2')."""

    to_square: Optional[str] = None
    """Ending square in algebraic notation (e.g., 'e4')."""

    piece_type: Optional[str] = None
    """Type of piece that moved (e.g., 'pawn', 'knight')."""

    confidence: float = 0.0
    """Confidence in the detection."""

    reasoning: str = ""
    """Step-by-step reasoning from Cosmos."""


@dataclass
class GameState:
    """Current state of the chess game."""
    whose_turn: Turn
    """Whose turn it is."""

    opponent_moving: bool
    """Whether opponent is currently making a move."""

    should_robot_act: bool
    """Whether the robot should make its move now."""

    reasoning: str
    """Step-by-step reasoning from Cosmos."""

    confidence: float
    """Confidence in the game state analysis."""


class ChessGameReasoning:
    """Chess game reasoning using Cosmos Reason2.

    Uses Cosmos Reason2's embodied reasoning to understand game flow:
    - Whose turn is it?
    - Is the opponent currently moving?
    - What piece did they move?
    - Should the robot act now?

    This demonstrates Cosmos Reason2's key strength: embodied AI reasoning
    about multi-agent interactions and temporal dynamics.
    """

    PIXELS_PER_TOKEN = 32**2

    SYSTEM_PROMPT = (
        "You are an embodied chess-playing robot arm (SO-101, 5-DOF with gripper) "
        "mounted beside a chess board. You have two cameras: an overhead egocentric "
        "camera looking down at the board, and a wrist camera mounted on your gripper. "
        "The chess pieces are standard Staunton style on a regulation board. "
        "Your gripper can open wide enough to grasp any piece from the top. "
        "Reason about the physical scene from your embodied perspective -- you are "
        "the robot, these are your views, and you must plan and execute moves safely."
    )

    COT_FORMAT = """

Answer using the following format:

<think>
Your step-by-step reasoning about the physical scene.
</think>

<answer>
Your JSON response here.
</answer>"""

    TURN_DETECTION_PROMPT = """I am watching the chess board through my overhead camera. I am playing against a human opponent sitting across from me.

Analyze what I see and determine:
1. Is my opponent's hand near the board or moving a piece?
2. Has my opponent just completed a move (hand withdrawn, piece settled)?
3. Is it safe for me to move now, or should I wait for them to finish?

Respond in JSON:
{
    "whose_turn": "robot" or "opponent",
    "opponent_moving": true/false,
    "should_robot_act": true/false,
    "confidence": 0.0-1.0
}"""

    MOVE_DETECTION_PROMPT = """I just observed my opponent make a chess move. This video is from my overhead camera looking down at the board.

The board is oriented with white pieces (my pieces) closest to me and black pieces on the far side. Files run a-h from my left to right, ranks run 1-8 from my side outward.

Identify what move my opponent made:
1. Which piece moved? Look for which square is now empty that was occupied, and which square is now occupied that was empty.
2. What are the from and to squares in algebraic notation (e.g., e7 to e5)?
3. What type of piece is it?

Respond in JSON:
{
    "move_occurred": true/false,
    "from_square": "starting square or null",
    "to_square": "ending square or null",
    "piece_type": "pawn/knight/bishop/rook/queen/king or null",
    "confidence": 0.0-1.0
}"""

    TRAJECTORY_PLAN_PROMPT = """I need to pick up the {piece_type} on {from_square} and place it on {to_square} (move: {move_uci}).

Looking at my overhead camera view, I can see the board and all pieces. My gripper approaches from above. I need to plan my gripper's path as a sequence of 2D positions in the image.

Specify the trajectory as pixel coordinates in the image where my gripper should move. Use the actual pixel positions of the squares you can see.

Plan these waypoints:
1. Hover above {from_square} -- position my gripper over the piece
2. Descend to grasp the {piece_type} on {from_square}
3. Lift to clearance height above surrounding pieces
4. Transit to above {to_square} -- if pieces are in the way, arc around them
5. Descend to place on {to_square}
6. Release and retract upward

Respond in JSON:
{{
    "waypoints": [
        {{"point_2d": [x, y], "label": "description of this waypoint"}}
    ],
    "confidence": 0.0-1.0
}}"""

    EPISODE_CRITIQUE_PROMPT = """This video shows my arm attempting to pick up a {piece_type} from {from_square} and place it on {to_square}. I am reviewing my own execution.

Evaluate each phase of my movement:
1. Approach: Did I approach the piece cleanly without hitting adjacent pieces?
2. Grasp: Did I grip the piece securely? Did it slip or tilt?
3. Lift: Did I raise the piece high enough to clear other pieces?
4. Transport: Did I avoid knocking any pieces during transit?
5. Placement: Did I place the piece squarely on {to_square}? Is it stable and upright?
6. Collateral: Did I disturb any non-target pieces at any point?

Rate from 0 (complete failure) to 10 (perfect execution).

Respond in JSON:
{{
    "overall_score": 0-10,
    "success": true or false,
    "approach_safe": true or false,
    "grasp_stable": true or false,
    "trajectory_safe": true or false,
    "placement_stable": true or false,
    "physical_issues": ["list of specific issues"],
    "confidence": 0.0-1.0
}}"""

    GOAL_VERIFICATION_PROMPT = """I just tried to move the {piece_type} from {from_square} to {to_square} (move: {move_uci}). This is my overhead camera view of the board AFTER my attempt.

Check the physical result:
1. Is there a {piece_type} on {to_square}? Is it centered and upright?
2. Is {from_square} now empty (piece successfully picked up)?
3. Are any adjacent pieces knocked over, displaced, or leaning?
4. Is my gripper clear of the board (not still holding a piece)?

Respond in JSON:
{{
    "success": true or false,
    "reason": "brief explanation",
    "physical_issues": ["list of issues, empty if none"],
    "confidence": 0.0-1.0
}}"""

    BOARD_ANALYSIS_PROMPT = """I am looking at a chess board through my overhead camera. Describe what I see:

1. What is the current position? Identify where the major pieces are (kings, queens, rooks, bishops, knights) and the general pawn structure.
2. What phase of the game is this? (opening, middlegame, endgame)
3. Are there any pieces that appear to be under attack or in danger?
4. Is the board set up correctly, or are any pieces knocked over, displaced, or off-center on their squares?
5. Can I see my robot arm or gripper in the image? If so, where is it relative to the board?

Respond in JSON:
{
    "position_summary": "brief description of the position",
    "game_phase": "opening/middlegame/endgame",
    "pieces_at_risk": ["any pieces under immediate threat"],
    "board_condition": "clean/pieces_displaced/pieces_knocked",
    "physical_observations": ["anything notable about the physical scene"],
    "confidence": 0.0-1.0
}"""

    @staticmethod
    def _build_image_content(
        image: Image.Image,
        wrist_image: Optional[Image.Image] = None,
    ) -> list[dict]:
        """Build image content entries for a Qwen3VL conversation.

        When only an overhead image is provided, returns a single image entry.
        When both images are provided, labels each so the model can
        distinguish between them.

        Args:
            image: Primary overhead / egocentric camera image.
            wrist_image: Optional wrist camera image.

        Returns:
            List of content dict entries suitable for a Qwen3VL user message.
        """
        if wrist_image is None:
            return [{"type": "image", "image": image}]
        return [
            {"type": "text", "text": "Overhead camera view:"},
            {"type": "image", "image": image},
            {"type": "text", "text": "Wrist camera view:"},
            {"type": "image", "image": wrist_image},
        ]

    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-8B",
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize Cosmos Reason2 for chess game reasoning.

        Args:
            model_name: Cosmos model identifier
            device: Device to run on ('auto', 'cuda', 'cpu')
            dtype: Data type for model weights
        """
        self.model_name = model_name

        # Load model
        self.model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
            attn_implementation="sdpa"
        )

        self.processor = transformers.Qwen3VLProcessor.from_pretrained(model_name)

    def analyze_board(
        self,
        image: Image.Image,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        wrist_image: Optional[Image.Image] = None,
    ) -> "BoardAnalysis":
        """Analyze the chess board scene from camera images.

        Uses Cosmos Reason2 to describe the position, game phase,
        and physical state of the board. Useful for demo visualization.

        Args:
            image: Overhead camera view
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            wrist_image: Optional wrist camera view

        Returns:
            BoardAnalysis with scene description and reasoning
        """
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    *self._build_image_content(image, wrist_image),
                    {"type": "text", "text": self.BOARD_ANALYSIS_PROMPT + self.COT_FORMAT},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return self._parse_board_analysis(output_text)

    def _parse_board_analysis(self, response: str) -> "BoardAnalysis":
        """Parse Cosmos response into BoardAnalysis."""
        thinking = _extract_thinking(response)
        answer_text = _extract_answer(response)
        data = extract_json(answer_text)

        if data is not None:
            try:
                return BoardAnalysis(
                    position_summary=data.get("position_summary", ""),
                    game_phase=data.get("game_phase", "unknown"),
                    pieces_at_risk=data.get("pieces_at_risk", []),
                    board_condition=data.get("board_condition", "unknown"),
                    physical_observations=data.get("physical_observations", []),
                    confidence=float(data.get("confidence", 0.0)),
                    reasoning=thinking or response,
                )
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        return BoardAnalysis(
            position_summary="",
            game_phase="unknown",
            pieces_at_risk=[],
            board_condition="unknown",
            physical_observations=[],
            confidence=0.0,
            reasoning=response,
        )

    def analyze_game_state(
        self,
        video_frames: list[Image.Image],
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> GameState:
        """Analyze game state from video frames.

        Uses embodied reasoning to determine whose turn it is and whether
        the robot should act.

        Args:
            video_frames: List of PIL Images (video frames)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            GameState with turn information and reasoning
        """
        # Create conversation with robot-centric framing
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_frames},
                    {"type": "text", "text": self.TURN_DETECTION_PROMPT + self.COT_FORMAT},
                ],
            },
        ]

        # Process inputs
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Run inference
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Parse JSON response
        return self._parse_game_state(output_text)

    def detect_move(
        self,
        video_frames: list[Image.Image],
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> MoveDetection:
        """Detect what move the opponent made from video.

        Uses embodied reasoning to identify which piece moved and where.

        Args:
            video_frames: List of PIL Images showing the move
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            MoveDetection with move information and reasoning
        """
        # Create conversation
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_frames},
                    {"type": "text", "text": self.MOVE_DETECTION_PROMPT + self.COT_FORMAT},
                ],
            },
        ]

        # Process and generate
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Parse JSON response
        return self._parse_move_detection(output_text)

    def _parse_game_state(self, response: str) -> GameState:
        """Parse Cosmos response into GameState.

        Args:
            response: Raw text response from model

        Returns:
            Parsed GameState
        """
        thinking = _extract_thinking(response)
        answer_text = _extract_answer(response)
        data = extract_json(answer_text)

        if data is not None:
            try:

                # Parse whose turn
                turn_str = data.get("whose_turn", "unknown").lower()
                whose_turn = Turn.ROBOT if turn_str == "robot" else (
                    Turn.OPPONENT if turn_str == "opponent" else Turn.UNKNOWN
                )

                return GameState(
                    whose_turn=whose_turn,
                    opponent_moving=data.get("opponent_moving", False),
                    should_robot_act=data.get("should_robot_act", False),
                    reasoning=thinking or data.get("reasoning", response),
                    confidence=float(data.get("confidence", 0.0)),
                )
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # Fallback if JSON parsing fails
        return GameState(
            whose_turn=Turn.UNKNOWN,
            opponent_moving=False,
            should_robot_act=False,
            reasoning=response,
            confidence=0.0,
        )

    def _parse_move_detection(self, response: str) -> MoveDetection:
        """Parse Cosmos response into MoveDetection.

        Args:
            response: Raw text response from model

        Returns:
            Parsed MoveDetection
        """
        thinking = _extract_thinking(response)
        answer_text = _extract_answer(response)
        data = extract_json(answer_text)

        if data is not None:
            try:
                return MoveDetection(
                    move_occurred=data.get("move_occurred", False),
                    from_square=data.get("from_square"),
                    to_square=data.get("to_square"),
                    piece_type=data.get("piece_type"),
                    confidence=float(data.get("confidence", 0.0)),
                    reasoning=thinking or data.get("reasoning", response),
                )
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # Fallback
        return MoveDetection(
            move_occurred=False,
            confidence=0.0,
            reasoning=response,
        )

    def plan_correction(
        self,
        image: Image.Image,
        expected_fen: str,
        actual_fen: str,
        differences: list,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        wrist_image: Optional[Image.Image] = None,
    ) -> "CorrectionPlan":
        """Plan how to correct a move that didn't execute as expected.

        Uses Cosmos Reason2 to understand what went wrong physically and
        how to correct it.

        Args:
            image: Current egocentric camera view
            expected_fen: What the board should look like
            actual_fen: What the board actually looks like
            differences: List of SquareDifference objects from FEN comparison
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            CorrectionPlan with reasoning and suggested actions
        """
        # Format differences for prompt
        diff_text = "\n".join([str(d) for d in differences])

        prompt = f"""My move did not execute correctly. I can see the board through my overhead camera.

Expected board (FEN): {expected_fen}
Actual board (FEN): {actual_fen}

Differences:
{diff_text}

Looking at the current board state, analyze:
1. What went wrong physically? Did the piece slip from my gripper, land on the wrong square, knock another piece, or fail to be picked up at all?
2. What correction do I need to make? Which piece needs to move where?
3. Are there any pieces in the way of the correction?

Respond in JSON:
{{
    "physical_cause": "what went wrong",
    "correction_needed": "what to do to fix it",
    "obstacles": ["any obstacles to the correction"],
    "confidence": 0.0-1.0
}}"""

        # Create conversation
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    *self._build_image_content(image, wrist_image),
                    {"type": "text", "text": prompt + self.COT_FORMAT},
                ],
            },
        ]

        # Process and generate
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Parse JSON response
        return self._parse_correction_plan(output_text)

    def reason_about_action(
        self,
        image: Image.Image,
        move_uci: str,
        from_square: str,
        to_square: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        wrist_image: Optional[Image.Image] = None,
    ) -> "ActionReasoning":
        """Reason about how to physically execute a chess move.

        Uses Cosmos Reason2 to plan the physical execution before acting.

        Args:
            image: Current egocentric camera view
            move_uci: Move in UCI format (e.g., 'e2e4')
            from_square: Starting square
            to_square: Target square
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            ActionReasoning with obstacles, grasp strategy, and risks
        """
        prompt = f"""I need to move a piece from {from_square} to {to_square} (move: {move_uci}). This is my overhead camera view of the board.

Before I move, I need to plan carefully. Analyze the physical scene:
1. What pieces are between {from_square} and {to_square} that I could collide with during transit?
2. What pieces are immediately adjacent to {from_square} (risk of knocking when grasping) and {to_square} (risk of knocking when placing)?
3. How should I approach the grasp? My gripper opens to about 4cm and descends from above. The piece is roughly 3-5cm tall.
4. What is the safest trajectory -- should I go straight, or arc to avoid crowded squares?

Respond in JSON:
{{
    "obstacles": ["pieces in the transit path"],
    "adjacent_pieces": ["pieces next to source and target squares"],
    "grasp_strategy": "how to approach and grasp",
    "trajectory_advice": "recommended path",
    "risks": ["potential problems"],
    "confidence": 0.0-1.0
}}"""

        # Create conversation
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    *self._build_image_content(image, wrist_image),
                    {"type": "text", "text": prompt + self.COT_FORMAT},
                ],
            },
        ]

        # Process and generate
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Parse JSON response
        return self._parse_action_reasoning(output_text)

    def _parse_correction_plan(self, response: str) -> "CorrectionPlan":
        """Parse Cosmos response into CorrectionPlan."""
        thinking = _extract_thinking(response)
        answer_text = _extract_answer(response)
        data = extract_json(answer_text)

        if data is not None:
            try:
                return CorrectionPlan(
                    physical_cause=data.get("physical_cause", ""),
                    correction_needed=data.get("correction_needed", ""),
                    obstacles=data.get("obstacles", []),
                    confidence=float(data.get("confidence", 0.0)),
                    reasoning=thinking or data.get("reasoning", response),
                )
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # Fallback
        return CorrectionPlan(
            physical_cause="Unknown",
            correction_needed="Unknown",
            obstacles=[],
            confidence=0.0,
            reasoning=response,
        )

    def _parse_action_reasoning(self, response: str) -> "ActionReasoning":
        """Parse Cosmos response into ActionReasoning."""
        thinking = _extract_thinking(response)
        answer_text = _extract_answer(response)
        data = extract_json(answer_text)

        if data is not None:
            try:
                return ActionReasoning(
                    obstacles=data.get("obstacles", []),
                    adjacent_pieces=data.get("adjacent_pieces", []),
                    grasp_strategy=data.get("grasp_strategy", ""),
                    trajectory_advice=data.get("trajectory_advice", ""),
                    risks=data.get("risks", []),
                    confidence=float(data.get("confidence", 0.0)),
                    reasoning=thinking or data.get("reasoning", response),
                )
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # Fallback
        return ActionReasoning(
            obstacles=[],
            adjacent_pieces=[],
            grasp_strategy="",
            trajectory_advice="",
            risks=[],
            confidence=0.0,
            reasoning=response,
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
        wrist_image: Optional[Image.Image] = None,
    ) -> "TrajectoryPlan":
        """Plan a 2D pixel-space trajectory for executing a chess move.

        Uses Cosmos Reason2's Action CoT to output normalized pixel
        coordinates (0-1000) as waypoints for the gripper trajectory.

        Args:
            image: Current egocentric camera view
            move_uci: Move in UCI format (e.g., 'e2e4')
            from_square: Starting square
            to_square: Target square
            piece_type: Type of piece being moved (e.g., 'pawn', 'knight')
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            TrajectoryPlan with ordered waypoints and reasoning
        """
        prompt = self.TRAJECTORY_PLAN_PROMPT.format(
            move_uci=move_uci,
            from_square=from_square,
            to_square=to_square,
            piece_type=piece_type,
        )

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    *self._build_image_content(image, wrist_image),
                    {"type": "text", "text": prompt + self.COT_FORMAT},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return self._parse_trajectory_plan(output_text, move_uci)

    def verify_goal(
        self,
        image: Image.Image,
        move_uci: str,
        from_square: str,
        to_square: str,
        piece_type: str = "piece",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        wrist_image: Optional[Image.Image] = None,
    ) -> "GoalVerification":
        """Verify physical outcome of a chess move from post-action image.

        Uses Cosmos Reason2 to visually check whether the move succeeded
        physically, catching issues that FEN comparison misses (tilted
        pieces, adjacent pieces bumped, gripper not released, etc.).

        Args:
            image: Post-action egocentric camera view
            move_uci: Move that was attempted
            from_square: Source square
            to_square: Target square
            piece_type: Type of piece moved
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            GoalVerification with success status and physical issue details
        """
        prompt = self.GOAL_VERIFICATION_PROMPT.format(
            move_uci=move_uci,
            from_square=from_square,
            to_square=to_square,
            piece_type=piece_type,
        )

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    *self._build_image_content(image, wrist_image),
                    {"type": "text", "text": prompt + self.COT_FORMAT},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return self._parse_goal_verification(output_text)

    def critique_episode(
        self,
        video_frames: list[Image.Image],
        from_square: str,
        to_square: str,
        piece_type: str = "piece",
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> "EpisodeCritique":
        """Critique a full RL episode from video frames.

        Watches the entire pick-and-place execution and evaluates approach
        safety, grasp stability, trajectory quality, and placement precision.

        Args:
            video_frames: Ordered frames from the episode
            from_square: Source square
            to_square: Target square
            piece_type: Type of piece being moved
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            EpisodeCritique with score and detailed evaluation
        """
        prompt = self.EPISODE_CRITIQUE_PROMPT.format(
            from_square=from_square,
            to_square=to_square,
            piece_type=piece_type,
        )

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_frames},
                    {"type": "text", "text": prompt + self.COT_FORMAT},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return self._parse_episode_critique(output_text)

    def _parse_episode_critique(self, response: str) -> "EpisodeCritique":
        """Parse Cosmos response into EpisodeCritique."""
        thinking = _extract_thinking(response)
        answer_text = _extract_answer(response)
        data = extract_json(answer_text)

        if data is not None:
            try:
                return EpisodeCritique(
                    overall_score=float(data.get("overall_score", 0.0)),
                    success=data.get("success", False),
                    approach_safe=data.get("approach_safe", False),
                    grasp_stable=data.get("grasp_stable", False),
                    trajectory_safe=data.get("trajectory_safe", False),
                    placement_stable=data.get("placement_stable", False),
                    physical_issues=data.get("physical_issues", []),
                    confidence=float(data.get("confidence", 0.0)),
                    reasoning=thinking or data.get("reasoning", response),
                )
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # Fallback
        return EpisodeCritique(
            overall_score=0.0,
            success=False,
            approach_safe=False,
            grasp_stable=False,
            trajectory_safe=False,
            placement_stable=False,
            physical_issues=["parse_failure"],
            confidence=0.0,
            reasoning=response,
        )

    def _parse_trajectory_plan(self, response: str, move_uci: str) -> "TrajectoryPlan":
        """Parse Cosmos response into TrajectoryPlan."""
        thinking = _extract_thinking(response)
        answer_text = _extract_answer(response)
        data = extract_json(answer_text)

        if data is not None:
            try:
                waypoints = []
                for wp in data.get("waypoints", []):
                    waypoints.append(TrajectoryWaypoint(
                        point_2d=tuple(wp["point_2d"]),
                        label=wp.get("label", ""),
                    ))
                return TrajectoryPlan(
                    waypoints=waypoints,
                    move_uci=move_uci,
                    reasoning=thinking or data.get("reasoning", response),
                    confidence=float(data.get("confidence", 0.0)),
                )
            except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                pass

        # Fallback
        return TrajectoryPlan(
            waypoints=[],
            move_uci=move_uci,
            reasoning=response,
            confidence=0.0,
        )

    def _parse_goal_verification(self, response: str) -> "GoalVerification":
        """Parse Cosmos response into GoalVerification."""
        thinking = _extract_thinking(response)
        answer_text = _extract_answer(response)
        data = extract_json(answer_text)

        if data is not None:
            try:
                return GoalVerification(
                    success=data.get("success", False),
                    reason=data.get("reason", ""),
                    physical_issues=data.get("physical_issues", []),
                    confidence=float(data.get("confidence", 0.0)),
                    reasoning=thinking or data.get("reasoning", response),
                )
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # Fallback: assume failure when parsing fails
        return GoalVerification(
            success=False,
            reason="Failed to parse verification response",
            physical_issues=["parse_failure"],
            confidence=0.0,
            reasoning=response,
        )


@dataclass
class BoardAnalysis:
    """Scene analysis of the chess board."""

    position_summary: str
    """Brief description of the position."""

    game_phase: str
    """Game phase: opening, middlegame, or endgame."""

    pieces_at_risk: list[str]
    """Pieces under immediate threat."""

    board_condition: str
    """Physical condition: clean, pieces_displaced, pieces_knocked."""

    physical_observations: list[str]
    """Notable physical observations about the scene."""

    confidence: float
    """Confidence in the analysis."""

    reasoning: str
    """Step-by-step reasoning from Cosmos."""


@dataclass
class CorrectionPlan:
    """Plan for correcting a failed move."""

    physical_cause: str
    """What physically went wrong."""

    correction_needed: str
    """Description of the correction needed."""

    obstacles: list[str]
    """Obstacles to making the correction."""

    confidence: float
    """Confidence in the analysis."""

    reasoning: str
    """Step-by-step reasoning from Cosmos."""


@dataclass
class ActionReasoning:
    """Physical reasoning about how to execute an action."""

    obstacles: list[str]
    """Obstacles in the path."""

    adjacent_pieces: list[str]
    """Pieces near the from/to squares."""

    grasp_strategy: str
    """How to grasp the piece."""

    trajectory_advice: str
    """Safest path to take."""

    risks: list[str]
    """Potential issues to watch for."""

    confidence: float
    """Confidence in the analysis."""

    reasoning: str
    """Step-by-step reasoning from Cosmos."""


@dataclass
class TrajectoryWaypoint:
    """A single 2D pixel waypoint in a gripper trajectory."""

    point_2d: tuple[int, int]
    """Normalized pixel coordinates (0-1000 range, origin top-left)."""

    label: str
    """Semantic label (e.g., 'above e2', 'grasp e2', 'lift')."""


@dataclass
class TrajectoryPlan:
    """Planned 2D pixel-space trajectory for a chess move (Action CoT)."""

    waypoints: list[TrajectoryWaypoint]
    """Ordered list of waypoints the end-effector should follow."""

    move_uci: str
    """The UCI move this trajectory executes."""

    reasoning: str
    """Step-by-step reasoning about trajectory choices."""

    confidence: float
    """Confidence in the trajectory plan (0.0-1.0)."""


@dataclass
class GoalVerification:
    """Result of post-action visual goal verification."""

    success: bool
    """Whether the move was physically successful."""

    reason: str
    """Brief explanation of the verification result."""

    physical_issues: list[str]
    """Detected physical issues (e.g., 'piece_unstable', 'adjacent_bumped')."""

    confidence: float
    """Confidence in the verification (0.0-1.0)."""

    reasoning: str
    """Step-by-step reasoning from Cosmos."""


@dataclass
class EpisodeCritique:
    """Critique of a full RL episode from video."""

    overall_score: float
    """Overall execution quality score (0-10)."""

    success: bool
    """Whether the task was completed successfully."""

    approach_safe: bool
    """Whether the approach phase avoided collisions."""

    grasp_stable: bool
    """Whether the grasp was stable throughout."""

    trajectory_safe: bool
    """Whether the transport trajectory was safe."""

    placement_stable: bool
    """Whether the placement was gentle and stable."""

    physical_issues: list[str]
    """Specific physical issues detected."""

    confidence: float
    """Confidence in the critique (0.0-1.0)."""

    reasoning: str
    """Step-by-step evaluation from Cosmos."""
