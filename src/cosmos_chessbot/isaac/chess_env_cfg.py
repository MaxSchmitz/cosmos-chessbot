"""Environment configuration for ChessPickPlaceEnv.

Extends LeIsaac's SingleArmTaskDirectEnvCfg with chess-specific reward
weights, success thresholds, and dataset paths. The base class already
provides action_space=6 (5 arm joints + 1 gripper) and default observation
handling.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

from leisaac.tasks.template.direct.single_arm_env import SingleArmTaskDirectEnvCfg

from .chess_scene_cfg import ChessSceneCfg


@configclass
class ChessPickPlaceEnvCfg(SingleArmTaskDirectEnvCfg):
    """Configuration for the chess pick-and-place RL environment."""

    # -- Scene ---------------------------------------------------------------
    scene: ChessSceneCfg = ChessSceneCfg(num_envs=64, env_spacing=2.5)

    # -- Task description (required by parent) --------------------------------
    task_description: str = "Pick a chess piece and place it on a target square."

    # -- Observation & action spaces -----------------------------------------
    # Inherited action_space = 6 (5 arm joints + 1 gripper)
    # We add our own RL obs space: a flat 21-dim vector
    # arm_joint_pos(5) + gripper(1) + ee_pos(3) + ee_quat(4) +
    # target_piece_rel(3) + target_square_rel(3) + is_grasped(1) + phase(1)
    num_rl_observations: int = 21
    num_arm_joints: int = 5
    num_gripper_joints: int = 1

    # -- Reward weights ------------------------------------------------------
    approach_weight: float = -1.0     # dense: -||ee - piece||
    placement_weight: float = -10.0   # dense: -||piece - target_square|| (when grasped)
    collision_weight: float = -5.0    # dense: per non-target contact
    success_bonus: float = 100.0      # sparse: piece placed within tolerance
    grasp_bonus: float = 10.0         # sparse: piece first grasped

    # -- Piece physics -------------------------------------------------------
    piece_mass_kg: float = 0.010
    piece_friction: float = 0.8
    piece_restitution: float = 0.1
    piece_linear_damping: float = 5.0
    piece_angular_damping: float = 5.0

    # -- Grasp parameters ----------------------------------------------------
    grasp_threshold: float = 0.05       # 5cm max proximity for grasp trigger
    grasp_quality_sigma: float = 0.01   # exponential decay for grasp quality reward
    grasp_offset_z: float = -0.01       # piece hangs 1cm below EE when grasped

    # -- Collision parameters ------------------------------------------------
    collision_displacement_threshold: float = 0.005  # 5mm noise filter
    knocked_off_terminates: bool = True

    # -- Success / termination thresholds ------------------------------------
    placement_tolerance: float = 0.02   # 2 cm from target square center
    grasp_height_threshold: float = 0.05  # piece must be lifted 5 cm to count
    board_drop_z: float = -0.10       # if piece drops this far below board, fail

    # -- Piece USD directory (relative to project root) ----------------------
    usd_dir: str = "data/usd"

    # -- FEN dataset (JSONL with conversations containing FEN strings) -------
    fen_dataset_path: str = "data/chess_fen_train.jsonl"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 5.0
        self.decimation = 2  # control at 30 Hz (sim at 60 Hz)
