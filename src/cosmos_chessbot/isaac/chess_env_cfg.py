"""Environment configuration for ChessPickPlaceEnv.

Extends LeIsaac's SingleArmTaskDirectEnvCfg with chess-specific reward
weights, success thresholds, and dataset paths. The base class already
provides action_space=6 (5 arm joints + 1 gripper) and default observation
handling.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from leisaac.tasks.template.direct.single_arm_env import SingleArmTaskDirectEnvCfg

from .chess_scene_cfg import ChessSceneCfg


@configclass
class ChessPickPlaceEnvCfg(SingleArmTaskDirectEnvCfg):
    """Configuration for the chess pick-and-place RL environment."""

    # -- Scene ---------------------------------------------------------------
    scene: ChessSceneCfg = ChessSceneCfg(num_envs=96, env_spacing=2.5)

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

    # -- Reward: approach (tanh kernel, two scales) ---------------------------
    approach_weight: float = 1.0        # coarse: 1 - tanh(dist / std)
    approach_std: float = 0.1           # 10cm std — guides from far
    approach_fine_weight: float = 0.5   # fine-grained precision near piece
    approach_fine_std: float = 0.03     # 3cm std — sharp near piece

    # -- Reward: lift (binary per-step, gated on grasp) ---------------------
    lift_weight: float = 15.0           # strong per-step signal when lifted
    lift_threshold: float = 0.03        # 3cm above initial piece Z

    # -- Reward: transport (tanh kernel, gated on lift) ---------------------
    transport_weight: float = 16.0      # coarse: guides toward target
    transport_std: float = 0.3          # 30cm std — broad attraction
    transport_fine_weight: float = 5.0  # fine precision at target
    transport_fine_std: float = 0.05    # 5cm std — sharp near target

    # -- Reward: success bonus (sparse) ------------------------------------
    success_bonus: float = 100.0        # piece placed within tolerance

    # -- Reward: grasp bonus (one-time) ------------------------------------
    grasp_bonus: float = 5.0            # one-time when first grasped

    # -- Reward: collision penalty (milestone curriculum) -------------------
    # Ramps based on GRASP COUNT, not time.  Stays near zero until the
    # policy learns to grasp, then increases so it learns precision.
    collision_weight: float = -0.5      # final weight (ramped to)
    collision_weight_start: float = -0.001  # initial weight (near zero)
    collision_milestone_grasps: int = 500  # full penalty after this many grasps

    # -- Reward: action smoothness (milestone curriculum) ------------------
    # Penalizes jerk (sudden direction changes), NOT speed.
    action_rate_weight: float = -0.005  # final ||a_t - a_{t-1}||² weight
    action_rate_start: float = -1e-5    # initial (near zero)
    action_rate_milestone_grasps: int = 500  # full penalty after this many grasps

    # -- Piece physics -------------------------------------------------------
    piece_mass_kg: float = 0.010
    piece_friction: float = 0.8
    piece_restitution: float = 0.1
    piece_linear_damping: float = 5.0
    piece_angular_damping: float = 5.0

    # -- Grasp parameters ----------------------------------------------------
    grasp_threshold: float = 0.05       # 5cm max proximity for grasp trigger
    grasp_quality_sigma: float = 0.03   # exp decay (widened from 0.01)
    grasp_offset_z: float = -0.01       # piece hangs 1cm below EE when grasped

    # -- Collision parameters ------------------------------------------------
    collision_displacement_threshold: float = 0.005  # 5mm noise filter
    knocked_off_terminates: bool = True

    # -- Success / termination thresholds ------------------------------------
    placement_tolerance: float = 0.02   # 2 cm from target square center
    board_drop_z: float = -0.10         # if piece drops below board, fail

    # -- Piece USD directory (relative to project root) ----------------------
    usd_dir: str = "data/usd"

    # -- FEN dataset (JSONL with conversations containing FEN strings) -------
    fen_dataset_path: str = "data/chess_fen_train.jsonl"

    # -- PhysX GPU buffers (64 envs × 32 pieces = 2048 rigid bodies) ----------
    sim: SimulationCfg = SimulationCfg(
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,           # 8M (default 2^21)
            gpu_max_rigid_patch_count=2**19,                # 512K (default 5*2^15)
            gpu_found_lost_aggregate_pairs_capacity=2**26,  # 64M (default 2^25)
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 5.0
        self.decimation = 2  # control at 30 Hz (sim at 60 Hz)
