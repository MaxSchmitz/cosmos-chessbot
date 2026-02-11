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

    # -- Reward: approach (single tanh kernel, matching isaac_so_arm101) ------
    approach_weight: float = 1.0        # 1 - tanh(dist / std)
    approach_std: float = 0.05          # 5cm std (isaac_so_arm101 default)

    # -- Reward: lift (height-gated, matching isaac_so_arm101) ----------------
    # Gates on piece HEIGHT only — no grasp detection.  If the piece is above
    # the threshold, gravity guarantees something is holding it.
    lift_weight: float = 15.0           # binary per-step bonus (isaac_so_arm101: 15.0)
    lift_threshold: float = 0.025       # 2.5cm above initial Z (isaac_so_arm101: 0.025)

    # -- Reward: transport (tanh kernel, gated on piece lifted) -------------
    transport_weight: float = 16.0      # coarse: guides toward target
    transport_std: float = 0.3          # 30cm std — broad attraction
    transport_fine_weight: float = 5.0  # fine precision at target
    transport_fine_std: float = 0.05    # 5cm std — sharp near target

    # -- Reward: success bonus (sparse) ------------------------------------
    success_bonus: float = 100.0        # piece placed within tolerance

    # -- Reward: collision penalty (milestone curriculum) -------------------
    # Ramps based on GRASP COUNT, not time.  Stays near zero until the
    # policy learns to grasp, then increases so it learns precision.
    collision_weight: float = -0.3      # final weight (ramped to)
    collision_weight_start: float = -0.001  # initial weight (near zero)
    collision_milestone_grasps: int = 1000  # full penalty after this many grasps

    # -- Reward: action smoothness (milestone curriculum) ------------------
    # Penalizes jerk (sudden direction changes), NOT speed.
    action_rate_weight: float = -0.003  # final ||a_t - a_{t-1}||² weight
    action_rate_start: float = -1e-5    # initial (near zero)
    action_rate_milestone_grasps: int = 1000  # full penalty after this many grasps

    # -- Piece count (reduce for training throughput) -------------------------
    num_pieces: int = 32  # 32 = full board; 1 = single piece for fast training

    # -- Camera (disable to save VRAM for large env counts) ------------------
    enable_camera: bool = True

    # -- Piece physics -------------------------------------------------------
    piece_mass_kg: float = 0.010
    piece_friction: float = 0.8
    piece_restitution: float = 0.1
    piece_linear_damping: float = 5.0
    piece_angular_damping: float = 5.0

    # -- Grasp parameters ----------------------------------------------------
    grasp_threshold: float = 0.02       # 2cm jaw-to-piece (LeIsaac default)
    gripper_closed_threshold: float = 0.26  # joint angle below this = closed (LeIsaac default)
    piece_grasp_z_offset: float = 0.007 # 7mm above base = ~center of scaled pawn

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

    # -- Reason2 critic (episode video evaluation) ----------------------------
    critic_render_interval: int = 5  # capture camera frame every N control steps

    # -- PhysX GPU buffers ---------------------------------------------------
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
        self.sim.render_interval = self.decimation
