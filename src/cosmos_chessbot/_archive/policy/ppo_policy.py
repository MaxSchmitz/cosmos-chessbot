"""PPO policy trained in Isaac Sim for chess piece pick-and-place.

Loads an ActorCritic checkpoint trained via PPO in Isaac Sim and runs
inference on real robot observations.  The policy expects a 21-dim
observation and outputs 6-dim normalized actions (5 arm joints + gripper).

The observation vector is constructed from:
  - arm joint positions (5, radians)
  - gripper position (1, radians)
  - end-effector position (3, metres)
  - end-effector quaternion (4, wxyz)
  - target piece relative position (3, metres)
  - target square relative position (3, metres)
  - is_grasped flag (1)
  - phase (1)
"""

import logging
import re
import time
from pathlib import Path
from typing import Optional

import chess
import numpy as np
import torch
import torch.nn as nn

from .base_policy import BasePolicy, PolicyAction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Network architecture (must match Isaac Sim training)
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Actor-Critic network from Isaac Sim PPO training."""

    def __init__(self, obs_dim: int = 21, act_dim: int = 6, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ELU(),
        )
        self.actor_branch = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic_branch = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        shared_features = self.shared(obs)
        actor_features = self.actor_branch(shared_features)
        action_mean = self.actor_mean(actor_features)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        critic_input = shared_features + 0.5 * (shared_features.detach() - shared_features)
        critic_features = self.critic_branch(critic_input)
        value = self.critic_head(critic_features).squeeze(-1)
        return action_mean, action_std, value


class RunningMeanStd:
    """Tracks running mean and std for observation normalization."""

    def __init__(self, shape, device, epsilon=1e-4):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / (self.var.sqrt() + 1e-8)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

DEFAULT_JOINT_LIMITS = {
    "shoulder_pan": (-90.0, 90.0),
    "shoulder_lift": (-120.0, 0.0),
    "elbow_flex": (0.0, 150.0),
    "wrist_flex": (0.0, 120.0),
    "wrist_roll": (-90.0, 90.0),
    "gripper": (0.0, 5.0),
}

# Rough link lengths for SO-101 (approximate)
_L1, _L2, _L3, _L4 = 0.15, 0.15, 0.15, 0.10


def _compute_ee_pose(joint_rad: np.ndarray):
    """Simplified FK returning (ee_pos(3,), ee_quat(4,))."""
    q = joint_rad
    x = _L1 + _L2 * np.cos(q[1]) + _L3 * np.cos(q[1] + q[2]) + _L4 * np.cos(q[1] + q[2] + q[3])
    z = 0.1 + _L2 * np.sin(q[1]) + _L3 * np.sin(q[1] + q[2]) + _L4 * np.sin(q[1] + q[2] + q[3])
    y = 0.0
    return (
        np.array([x, y, z], dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def _square_to_world(square_name: str, board_size: float, board_height: float,
                     board_center=(0.3, 0.0, 0.0)) -> np.ndarray:
    """Chess square name -> 3-D world position."""
    sq = chess.parse_square(square_name)
    file_coord = chess.square_file(sq) + 0.5
    rank_coord = chess.square_rank(sq) + 0.5
    fn = (file_coord / 8.0) - 0.5
    rn = (rank_coord / 8.0) - 0.5
    return np.array([
        board_center[0] + fn * board_size,
        board_center[1] + rn * board_size,
        board_center[2] + board_height,
    ], dtype=np.float32)


def _construct_obs(arm_rad, gripper_rad, ee_pos, ee_quat,
                   piece_pos, square_pos, grasped, phase):
    """Build 21-dim observation vector."""
    return np.concatenate([
        arm_rad,                       # 5
        [gripper_rad],                 # 1
        ee_pos,                        # 3
        ee_quat,                       # 4
        piece_pos - ee_pos,            # 3
        square_pos - ee_pos,           # 3
        [float(grasped)],              # 1
        [phase],                       # 1
    ])  # 21


def _denormalize(action_norm: np.ndarray, joint_limits: dict) -> np.ndarray:
    """Map actions from [-1,1] to joint angle ranges in degrees."""
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
             "wrist_flex", "wrist_roll", "gripper"]
    out = np.zeros_like(action_norm)
    for i, n in enumerate(names):
        lo, hi = joint_limits[n]
        out[i] = (action_norm[i] + 1.0) / 2.0 * (hi - lo) + lo
    return out


def _parse_squares(instruction: Optional[str]):
    """Extract source and target square names from an instruction string.

    Accepts formats like:
        "Pick the piece at e2 and place it at e4"
        "e2 e4"
    """
    if instruction is None:
        return None, None
    squares = re.findall(r'\b([a-h][1-8])\b', instruction)
    if len(squares) >= 2:
        return squares[0], squares[1]
    return None, None


# ---------------------------------------------------------------------------
# PPOPolicy
# ---------------------------------------------------------------------------

class PPOPolicy(BasePolicy):
    """Isaac-Sim PPO policy for chess piece manipulation.

    Wraps the ActorCritic checkpoint in the BasePolicy interface.
    Supports both single-step and multi-step (continuous) control.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: str = "cpu",
        board_size: float = 0.4,
        board_height: float = 0.0,
        joint_limits: Optional[dict] = None,
        max_steps: int = 100,
        control_hz: float = 20.0,
    ):
        self.device = torch.device(device)
        self.board_size = board_size
        self.board_height = board_height
        self.joint_limits = joint_limits or DEFAULT_JOINT_LIMITS
        self.max_steps = max_steps
        self.control_period = 1.0 / control_hz

        self.net: Optional[ActorCritic] = None
        self.obs_norm: Optional[RunningMeanStd] = None

        self._is_grasped = False
        self._phase = 0.0

        if checkpoint_path and Path(checkpoint_path).exists():
            self._load(Path(checkpoint_path))
        else:
            logger.warning("No PPO checkpoint provided or file missing -- policy will return zeros")

    # ----- loading --------------------------------------------------------

    def _load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        logger.info("PPO checkpoint: step=%s  episodes=%s",
                     ckpt.get("step", "?"), ckpt.get("episode_count", "?"))

        self.net = ActorCritic(obs_dim=21, act_dim=6, hidden=256).to(self.device)
        self.net.load_state_dict(ckpt["policy_state_dict"])
        self.net.eval()

        self.obs_norm = RunningMeanStd(21, self.device)
        if "obs_normalizer" in ckpt:
            ns = ckpt["obs_normalizer"]
            self.obs_norm.mean = ns["mean"].to(self.device)
            self.obs_norm.var = ns["var"].to(self.device)
            self.obs_norm.count = ns["count"]
            logger.info("  Obs normalizer loaded (count=%d)", self.obs_norm.count)

        params = sum(p.numel() for p in self.net.parameters())
        logger.info("  PPO policy loaded (%d params)", params)

    # ----- BasePolicy interface -------------------------------------------

    def reset(self):
        self._is_grasped = False
        self._phase = 0.0

    def select_action(
        self,
        images: dict,
        robot_state: np.ndarray,
        instruction: Optional[str] = None,
    ) -> PolicyAction:
        """Compute a single action step.

        ``robot_state`` is expected to be (6,) joint angles **in degrees**
        as returned by the orchestrator's ``_get_robot_state()``.
        """
        if self.net is None:
            return PolicyAction(
                actions=np.zeros(6),
                success_probability=0.0,
                metadata={"policy": "ppo", "error": "no checkpoint"},
            )

        source_sq, target_sq = _parse_squares(instruction)
        obs, value = self._infer_step(robot_state, source_sq, target_sq)
        return PolicyAction(
            actions=obs,          # denormalized joint targets in degrees (6,)
            success_probability=float(value),
            metadata={"policy": "ppo", "source": source_sq, "target": target_sq},
        )

    def plan_action(
        self,
        images: dict,
        robot_state: np.ndarray,
        instruction: Optional[str] = None,
    ) -> list[PolicyAction]:
        """Return single-step result (PPO has no multi-candidate planning)."""
        return [self.select_action(images, robot_state, instruction)]

    # ----- Continuous control loop ----------------------------------------

    def run_control_loop(
        self,
        robot,
        source_square: str,
        target_square: str,
        get_state_fn,
        send_action_fn,
    ) -> bool:
        """Run the PPO policy in a closed-loop on the real robot.

        This is the main execution method for PPO -- it runs at
        ``control_hz`` calling the policy every step with fresh
        observations until ``max_steps`` is reached.

        Args:
            robot: lerobot robot instance (or None for dry-run).
            source_square: Pick square (e.g. 'e2').
            target_square: Place square (e.g. 'e4').
            get_state_fn: Callable returning (6,) joint angles in degrees.
            send_action_fn: Callable accepting (6,) joint targets in degrees.

        Returns:
            True if the loop completed without error.
        """
        if self.net is None:
            logger.error("PPO policy not loaded -- cannot run control loop")
            return False

        self.reset()
        logger.info("PPO control loop: %s -> %s  (%d steps max, %.0f Hz)",
                     source_square, target_square, self.max_steps,
                     1.0 / self.control_period)

        try:
            for step in range(self.max_steps):
                t0 = time.time()

                robot_state_deg = get_state_fn()
                action_deg, value = self._infer_step(
                    robot_state_deg, source_square, target_square,
                )
                send_action_fn(action_deg)

                if step % 20 == 0:
                    logger.info("  step %3d  value=%.3f  action=%s",
                                step, value, np.array2string(action_deg, precision=1))

                # Maintain control frequency
                elapsed = time.time() - t0
                sleep = max(0.0, self.control_period - elapsed)
                if sleep > 0:
                    time.sleep(sleep)

            logger.info("PPO control loop finished (%d steps)", self.max_steps)
            return True

        except Exception as e:
            logger.error("PPO control loop error: %s", e)
            return False

    # ----- Internal -------------------------------------------------------

    def _infer_step(
        self,
        robot_state_deg: np.ndarray,
        source_square: Optional[str],
        target_square: Optional[str],
    ) -> tuple[np.ndarray, float]:
        """Run one inference step.

        Args:
            robot_state_deg: (6,) joint angles in degrees
            source_square: Pick square name
            target_square: Place square name

        Returns:
            (action_deg, value) -- denormalized joint targets in degrees and
            critic value estimate.
        """
        # Convert degrees -> radians (Isaac Sim trains with radians)
        state_rad = np.deg2rad(robot_state_deg.astype(np.float32))
        arm_rad = state_rad[:5]
        gripper_rad = float(state_rad[5])

        ee_pos, ee_quat = _compute_ee_pose(arm_rad)

        # Target positions from square names
        if source_square:
            piece_pos = _square_to_world(source_square, self.board_size, self.board_height)
            piece_pos[2] += 0.03  # piece height offset
        else:
            piece_pos = np.zeros(3, dtype=np.float32)

        if target_square:
            square_pos = _square_to_world(target_square, self.board_size, self.board_height)
        else:
            square_pos = np.zeros(3, dtype=np.float32)

        obs = _construct_obs(
            arm_rad, gripper_rad, ee_pos, ee_quat,
            piece_pos, square_pos, self._is_grasped, self._phase,
        )

        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device).float()
        obs_n = self.obs_norm.normalize(obs_t) if self.obs_norm else obs_t

        with torch.no_grad():
            action_mean, _, value = self.net(obs_n)
            action_clipped = action_mean.clamp(-1.0, 1.0)

        action_norm = action_clipped.cpu().numpy()[0]
        action_deg = _denormalize(action_norm, self.joint_limits)

        return action_deg, float(value.item())
