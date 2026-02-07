"""Chess pick-and-place RL environment with rigid body physics.

An SO-101 robot arm learns to pick a chess piece and place it at a random
target location within its workspace. Built on LeIsaac's SingleArmTaskDirectEnv
(which extends IsaacLab's DirectRLEnv via RecorderEnhanceDirectRLEnv).

Physics features:
- Pieces are rigid bodies with sphere colliders
- Kinematic-override grasp: piece becomes kinematic and follows EE when grasped
- On release: piece becomes dynamic, drops under gravity
- Collision penalty for displacing non-target pieces
- Termination if a non-target piece is knocked off the table
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema

from leisaac.tasks.template.direct.single_arm_env import SingleArmTaskDirectEnv

from .chess_env_cfg import ChessPickPlaceEnvCfg
from .fen_placement import (
    BOARD_SQUARE_SIZE,
    FEN_TO_PIECE_TYPE,
    PIECE_COLLISION_RADII,
    fen_to_board_state,
    get_square_position,
    validate_fen,
)
from .chess_scene_cfg import TABLE_HEIGHT, RL_SQUARE_SIZE, BOARD_CENTER_OFFSET_Y

# Maximum number of pieces on a chess board
MAX_PIECES = 32

# Robot workspace for random target sampling
_MAX_REACH = 0.30   # 30cm conservative (within 35cm SO-101 max)
_MIN_REACH = 0.05   # 5cm minimum from robot base


class ChessPickPlaceEnv(SingleArmTaskDirectEnv):
    """RL environment for chess piece pick-and-place with an SO-101 robot arm.

    Observation space (21-dim):
        - Arm joint positions (5)
        - Gripper state (1)
        - End-effector position (3)
        - End-effector quaternion (4)
        - Target piece position relative to EE (3)
        - Target square position relative to EE (3)
        - Is-grasped flag (1)
        - Phase: 0=approach, 1=transport (1)

    Action space (6-dim, inherited from SingleArmTaskDirectEnv):
        - Arm joint position targets (5)
        - Gripper command (1)
    """

    cfg: ChessPickPlaceEnvCfg

    def __init__(self, cfg: ChessPickPlaceEnvCfg, render_mode: Optional[str] = None, **kwargs):
        # Resolve project root (three levels up from this file's package)
        self._project_root = Path(__file__).resolve().parents[3]
        self._usd_dir = self._project_root / cfg.usd_dir

        # Load FEN dataset before super().__init__ (which calls _setup_scene)
        self._fen_list = self._load_fen_dataset(
            self._project_root / cfg.fen_dataset_path
        )

        # Per-env episode state (initialized in _reset_idx)
        self._target_piece_pos: Optional[torch.Tensor] = None   # (num_envs, 3)
        self._target_square_pos: Optional[torch.Tensor] = None  # (num_envs, 3)
        self._is_grasped: Optional[torch.Tensor] = None         # (num_envs,) bool
        self._has_been_grasped: Optional[torch.Tensor] = None   # (num_envs,) bool
        self._phase: Optional[torch.Tensor] = None              # (num_envs,) float
        self._gripper_cmd: Optional[torch.Tensor] = None        # (num_envs,) float

        # Piece physics state
        self._piece_pos: Optional[torch.Tensor] = None          # (num_envs, MAX_PIECES, 3)
        self._piece_initial_pos: Optional[torch.Tensor] = None  # (num_envs, MAX_PIECES, 3)
        self._piece_active_mask: Optional[torch.Tensor] = None  # (num_envs, MAX_PIECES) bool
        self._num_active_pieces: List[int] = []                  # per-env count

        # Piece prim management
        self._piece_pool_paths: List[List[str]] = []  # per-env list of prim paths
        self._target_piece_idx: List[int] = []        # index into piece pool per env
        self._target_piece_idx_tensor: Optional[torch.Tensor] = None  # (num_envs,) long

        # Board scale factor for collision radii
        self._board_scale = RL_SQUARE_SIZE / BOARD_SQUARE_SIZE

        # Track per-piece fen chars for collision radius updates
        self._piece_fen_chars: List[List[str]] = []  # per-env, per-slot

        super().__init__(cfg, render_mode=render_mode, **kwargs)

    # ------------------------------------------------------------------ #
    # Dataset loading
    # ------------------------------------------------------------------ #

    def _load_fen_dataset(self, path: Path) -> List[str]:
        """Load FEN strings from the JSONL dataset."""
        fens: List[str] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                for turn in record.get("conversations", []):
                    if turn.get("role") == "assistant":
                        fen_candidate = turn["content"].strip()
                        if validate_fen(fen_candidate):
                            fens.append(fen_candidate)
                        else:
                            full_fen = fen_candidate + " w - - 0 1"
                            if validate_fen(full_fen):
                                fens.append(full_fen)
        if not fens:
            fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
        return fens

    # ------------------------------------------------------------------ #
    # Scene setup
    # ------------------------------------------------------------------ #

    def _setup_scene(self):
        """Set up the chess scene: board USD + piece prim pool with rigid body physics.

        The parent class handles robot, ee_frame, cameras, and lights via
        the scene config. We add the chess board and pre-create a pool of
        piece prims with rigid body and collision APIs.
        """
        super()._setup_scene()

        stage = omni.usd.get_context().get_stage()
        board_usd = self._usd_dir / "board.usd"
        cfg = self.cfg
        board_scale = self._board_scale

        for env_idx in range(self.num_envs):
            env_ns = f"/World/envs/env_{env_idx}"

            # -- Load chess board USD ----------------------------------------
            board_path = f"{env_ns}/Board"
            board_prim = stage.DefinePrim(board_path, "Xform")
            board_prim.GetReferences().AddReference(str(board_usd))
            xform = UsdGeom.Xformable(board_prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(
                Gf.Vec3d(0.0, BOARD_CENTER_OFFSET_Y, TABLE_HEIGHT)
            )
            xform.AddScaleOp().Set(Gf.Vec3f(board_scale, board_scale, board_scale))

            # -- Pre-create piece pool with physics --------------------------
            env_pieces: List[str] = []
            env_fen_chars: List[str] = []
            for piece_idx in range(MAX_PIECES):
                piece_path = f"{env_ns}/Pieces/piece_{piece_idx}"
                piece_prim = stage.DefinePrim(piece_path, "Xform")
                UsdGeom.Imageable(piece_prim).MakeInvisible()

                # Apply rigid body API (start kinematic = inactive)
                rb_api = UsdPhysics.RigidBodyAPI.Apply(piece_prim)
                rb_api.CreateRigidBodyEnabledAttr(True)
                rb_api.CreateKinematicEnabledAttr(True)

                # Mass
                mass_api = UsdPhysics.MassAPI.Apply(piece_prim)
                mass_api.CreateMassAttr(cfg.piece_mass_kg)

                # Sphere collision as child prim
                col_path = f"{piece_path}/CollisionSphere"
                col_prim = stage.DefinePrim(col_path, "Sphere")
                col_prim.GetAttribute("radius").Set(0.025)  # default, updated per piece
                UsdPhysics.CollisionAPI.Apply(col_prim)

                # Material for friction/restitution
                mat_path = f"{piece_path}/PhysicsMaterial"
                mat_prim = stage.DefinePrim(mat_path, "Material")
                phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
                phys_mat.CreateStaticFrictionAttr(cfg.piece_friction)
                phys_mat.CreateDynamicFrictionAttr(cfg.piece_friction)
                phys_mat.CreateRestitutionAttr(cfg.piece_restitution)

                # Bind material to collision
                col_prim_ref = stage.GetPrimAtPath(col_path)
                if col_prim_ref.IsValid():
                    binding_api = UsdPhysics.MaterialAPI.Apply(col_prim_ref)
                    # PhysX will pick up the material from the parent hierarchy

                # PhysX damping
                physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(piece_prim)
                physx_rb.CreateLinearDampingAttr(cfg.piece_linear_damping)
                physx_rb.CreateAngularDampingAttr(cfg.piece_angular_damping)

                env_pieces.append(piece_path)
                env_fen_chars.append("")  # empty until reset

            self._piece_pool_paths.append(env_pieces)
            self._piece_fen_chars.append(env_fen_chars)

    # ------------------------------------------------------------------ #
    # Piece management helpers
    # ------------------------------------------------------------------ #

    def _configure_piece(
        self,
        prim_path: str,
        piece_type: str,
        position: np.ndarray,
        visible: bool = True,
        fen_char: str = "",
    ):
        """Configure a pooled piece prim: set USD reference, position, visibility, physics."""
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return

        prim.GetReferences().ClearReferences()
        piece_usd = self._usd_dir / f"{piece_type}.usd"
        if piece_usd.exists():
            prim.GetReferences().AddReference(str(piece_usd))

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(
            Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
        )

        imageable = UsdGeom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

        # Update collision sphere radius based on piece type
        if fen_char and fen_char in PIECE_COLLISION_RADII:
            col_path = f"{prim_path}/CollisionSphere"
            col_prim = stage.GetPrimAtPath(col_path)
            if col_prim.IsValid():
                radius = PIECE_COLLISION_RADII[fen_char] * self._board_scale
                radius_attr = col_prim.GetAttribute("radius")
                if radius_attr:
                    radius_attr.Set(radius)

        # Set kinematic mode: visible (active) pieces are dynamic,
        # invisible (inactive) pieces stay kinematic
        rb_api = UsdPhysics.RigidBodyAPI(prim)
        kinematic_attr = rb_api.GetKinematicEnabledAttr()
        if kinematic_attr:
            kinematic_attr.Set(not visible)

    def _hide_all_pieces(self, env_idx: int):
        """Hide all pieces and set them kinematic (inactive) for a given environment."""
        stage = omni.usd.get_context().get_stage()
        for prim_path in self._piece_pool_paths[env_idx]:
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()
                # Set kinematic so inactive pieces don't participate in physics
                rb_api = UsdPhysics.RigidBodyAPI(prim)
                kinematic_attr = rb_api.GetKinematicEnabledAttr()
                if kinematic_attr:
                    kinematic_attr.Set(True)

    def _set_piece_kinematic(self, prim_path: str, kinematic: bool):
        """Set a piece's kinematic mode via USD."""
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            rb_api = UsdPhysics.RigidBodyAPI(prim)
            kinematic_attr = rb_api.GetKinematicEnabledAttr()
            if kinematic_attr:
                kinematic_attr.Set(kinematic)

    # ------------------------------------------------------------------ #
    # Random target sampling
    # ------------------------------------------------------------------ #

    def _pick_random_target(self, env_origin: np.ndarray) -> np.ndarray:
        """Sample a random target position within the robot's reachable workspace.

        The SO-101 arm has ~35cm reach from base at env_origin + (0, 0, TABLE_HEIGHT).
        Samples uniformly within a disk on the table surface, biased toward
        the board area (positive Y from robot base).
        """
        angle = random.uniform(0, 2 * math.pi)
        # sqrt for uniform area distribution within annulus
        radius = math.sqrt(random.uniform(_MIN_REACH**2, _MAX_REACH**2))
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        # Robot base is at env_origin + (0, 0, TABLE_HEIGHT)
        target = np.array([
            env_origin[0] + x,
            env_origin[1] + y,
            TABLE_HEIGHT,
        ])
        return target

    # ------------------------------------------------------------------ #
    # Episode reset
    # ------------------------------------------------------------------ #

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments: sample FEN, place pieces, pick random target."""
        super()._reset_idx(env_ids)

        device = self.device

        # Lazy-init per-env tensors
        if self._target_piece_pos is None:
            self._target_piece_pos = torch.zeros(
                (self.num_envs, 3), device=device
            )
            self._target_square_pos = torch.zeros(
                (self.num_envs, 3), device=device
            )
            self._is_grasped = torch.zeros(
                self.num_envs, dtype=torch.bool, device=device
            )
            self._has_been_grasped = torch.zeros(
                self.num_envs, dtype=torch.bool, device=device
            )
            self._phase = torch.zeros(
                self.num_envs, dtype=torch.float32, device=device
            )
            self._gripper_cmd = torch.zeros(
                self.num_envs, dtype=torch.float32, device=device
            )
            # Physics state tensors
            self._piece_pos = torch.zeros(
                (self.num_envs, MAX_PIECES, 3), device=device
            )
            self._piece_initial_pos = torch.zeros(
                (self.num_envs, MAX_PIECES, 3), device=device
            )
            self._piece_active_mask = torch.zeros(
                (self.num_envs, MAX_PIECES), dtype=torch.bool, device=device
            )
            self._target_piece_idx_tensor = torch.zeros(
                self.num_envs, dtype=torch.long, device=device
            )
            self._num_active_pieces = [0] * self.num_envs

        for env_idx in env_ids.tolist():
            # 1. Sample a random FEN
            fen = random.choice(self._fen_list)
            board_state = fen_to_board_state(fen)

            # 2. Hide all existing pieces (sets them kinematic)
            self._hide_all_pieces(env_idx)

            # 3. Compute board center and env origin in WORLD coordinates
            env_origin = self.scene.env_origins[env_idx].cpu().numpy()
            board_center = env_origin + np.array(
                [0.0, BOARD_CENTER_OFFSET_Y, TABLE_HEIGHT]
            )

            # 4. Place pieces from FEN onto the board
            piece_squares: List[Tuple[str, str, np.ndarray]] = []
            self._piece_active_mask[env_idx] = False  # reset mask
            for pool_idx, (square, fen_char) in enumerate(board_state.items()):
                if pool_idx >= MAX_PIECES:
                    break
                piece_type = FEN_TO_PIECE_TYPE[fen_char]
                pos = get_square_position(
                    square, board_center=board_center, square_size=RL_SQUARE_SIZE
                )
                prim_path = self._piece_pool_paths[env_idx][pool_idx]
                self._configure_piece(
                    prim_path, piece_type, pos, visible=True, fen_char=fen_char
                )
                piece_squares.append((square, fen_char, pos))

                # Store position in physics tensors
                self._piece_pos[env_idx, pool_idx] = torch.tensor(
                    pos, dtype=torch.float32, device=device
                )
                self._piece_initial_pos[env_idx, pool_idx] = torch.tensor(
                    pos, dtype=torch.float32, device=device
                )
                self._piece_active_mask[env_idx, pool_idx] = True
                self._piece_fen_chars[env_idx][pool_idx] = fen_char

            # Hide remaining unused slots and move them far away
            num_active = len(piece_squares)
            self._num_active_pieces[env_idx] = num_active
            far_pos = np.array([env_origin[0], env_origin[1], -100.0])
            for pool_idx in range(num_active, MAX_PIECES):
                prim_path = self._piece_pool_paths[env_idx][pool_idx]
                self._configure_piece(
                    prim_path, "pawn_w", far_pos, visible=False, fen_char=""
                )
                self._piece_pos[env_idx, pool_idx] = torch.tensor(
                    far_pos, dtype=torch.float32, device=device
                )
                self._piece_initial_pos[env_idx, pool_idx] = torch.tensor(
                    far_pos, dtype=torch.float32, device=device
                )
                self._piece_fen_chars[env_idx][pool_idx] = ""

            # 5. Pick a random piece and random target location
            if piece_squares:
                src_idx = random.randrange(len(piece_squares))
                src_sq, src_char, src_pos = piece_squares[src_idx]
                target_pool_idx = src_idx
            else:
                src_pos = get_square_position(
                    "E2", board_center=board_center, square_size=RL_SQUARE_SIZE
                )
                target_pool_idx = 0

            target_pos = self._pick_random_target(env_origin)

            # 6. Store episode state
            self._target_piece_pos[env_idx] = torch.tensor(
                src_pos, dtype=torch.float32, device=device
            )
            self._target_square_pos[env_idx] = torch.tensor(
                target_pos, dtype=torch.float32, device=device
            )
            self._is_grasped[env_idx] = False
            self._has_been_grasped[env_idx] = False
            self._phase[env_idx] = 0.0
            if hasattr(self, "_prev_approach_dist"):
                self._prev_approach_dist[env_idx] = 1.0

            # Track target piece index
            while len(self._target_piece_idx) <= env_idx:
                self._target_piece_idx.append(0)
            self._target_piece_idx[env_idx] = target_pool_idx
            self._target_piece_idx_tensor[env_idx] = target_pool_idx

    # ------------------------------------------------------------------ #
    # Physics step
    # ------------------------------------------------------------------ #

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process raw actions before physics simulation."""
        super()._pre_physics_step(actions)
        n_arm = self.cfg.num_arm_joints  # 5
        self._gripper_cmd = (self.actions[:, n_arm] >= 0.0).float()

        # Update grasped pieces to follow EE before physics step
        self._update_grasped_pieces()

    # _apply_action is inherited: sends self.actions as joint position targets

    # ------------------------------------------------------------------ #
    # Grasped piece following
    # ------------------------------------------------------------------ #

    def _update_grasped_pieces(self):
        """Move grasped pieces to follow the end-effector position.

        Sets grasped pieces to kinematic and updates their position via USD.
        """
        if self._is_grasped is None:
            return

        grasped_envs = self._is_grasped.nonzero(as_tuple=True)[0]
        if len(grasped_envs) == 0:
            return

        robot = self.scene["robot"]
        ee_pos = robot.data.body_pos_w[:, -1, :]  # (N, 3)
        stage = omni.usd.get_context().get_stage()

        for env_idx in grasped_envs.tolist():
            piece_idx = self._target_piece_idx[env_idx]
            prim_path = self._piece_pool_paths[env_idx][piece_idx]

            # Position piece at EE with Z offset
            pos = ee_pos[env_idx].cpu().numpy()
            pos[2] += self.cfg.grasp_offset_z

            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                xform = UsdGeom.Xformable(prim)
                xform.ClearXformOpOrder()
                xform.AddTranslateOp().Set(
                    Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
                )

            # Update tracked position
            self._target_piece_pos[env_idx] = ee_pos[env_idx].clone()
            self._target_piece_pos[env_idx, 2] += self.cfg.grasp_offset_z
            self._piece_pos[env_idx, piece_idx] = self._target_piece_pos[env_idx]

    # ------------------------------------------------------------------ #
    # Piece position readback
    # ------------------------------------------------------------------ #

    def _update_piece_positions(self):
        """Read piece positions from USD stage (for non-grasped pieces).

        For grasped pieces, position is already updated by _update_grasped_pieces.
        For non-grasped active pieces, read current USD position.
        """
        if self._piece_pos is None:
            return

        stage = omni.usd.get_context().get_stage()
        for env_idx in range(self.num_envs):
            for piece_idx in range(self._num_active_pieces[env_idx]):
                # Skip grasped target piece (already tracked)
                if (self._is_grasped[env_idx]
                        and piece_idx == self._target_piece_idx[env_idx]):
                    continue

                prim_path = self._piece_pool_paths[env_idx][piece_idx]
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue

                xform = UsdGeom.Xformable(prim)
                translate_ops = [
                    op for op in xform.GetOrderedXformOps()
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
                ]
                if translate_ops:
                    pos = translate_ops[0].Get()
                    if pos is not None:
                        self._piece_pos[env_idx, piece_idx] = torch.tensor(
                            [pos[0], pos[1], pos[2]],
                            dtype=torch.float32,
                            device=self.device,
                        )

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #

    def _get_observations(self) -> dict:
        """Compute observation dict."""
        # Update piece positions from physics
        self._update_piece_positions()

        obs = super()._get_observations()

        n_arm = self.cfg.num_arm_joints  # 5
        robot = self.scene["robot"]

        # Robot state
        joint_pos = robot.data.joint_pos[:, :n_arm]           # (N, 5)
        gripper_pos = robot.data.joint_pos[:, n_arm:n_arm+1]  # (N, 1)

        # End-effector pose from the last body (gripper link)
        ee_pos = robot.data.body_pos_w[:, -1, :]              # (N, 3)
        ee_quat = robot.data.body_quat_w[:, -1, :]            # (N, 4)

        # Relative positions to targets
        target_piece_rel = self._target_piece_pos - ee_pos     # (N, 3)
        target_square_rel = self._target_square_pos - ee_pos   # (N, 3)

        # Grasp and phase flags
        is_grasped = self._is_grasped.float().unsqueeze(-1)    # (N, 1)
        phase = self._phase.unsqueeze(-1)                       # (N, 1)

        rl_obs = torch.cat(
            [
                joint_pos,
                gripper_pos,
                ee_pos,
                ee_quat,
                target_piece_rel,
                target_square_rel,
                is_grasped,
                phase,
            ],
            dim=-1,
        )
        obs["policy"]["rl_obs"] = rl_obs
        return obs

    # ------------------------------------------------------------------ #
    # Rewards
    # ------------------------------------------------------------------ #

    def _get_rewards(self) -> torch.Tensor:
        """Compute per-step reward.

        Rewards:
        - Approach: exp(-dist/0.1) — dense approach to piece
        - Distance reduction bonus: rewards getting closer vs previous step
        - Grasp bonus: scaled by grasp quality (closer = more reward)
        - Transport: exp(-dist/0.1) when grasped
        - Success: +100 when piece placed within tolerance
        - Collision penalty: displacement of non-target pieces
        """
        robot = self.scene["robot"]
        ee_pos = robot.data.body_pos_w[:, -1, :]
        cfg = self.cfg

        # -- Grasp state update (with grasp quality) -------------------------
        gripper_closed = self._gripper_cmd >= 0.5
        ee_to_piece_dist = torch.norm(ee_pos - self._target_piece_pos, dim=-1)
        close_to_piece = ee_to_piece_dist < cfg.grasp_threshold

        # Sticky grasp: once grasped, stay grasped while gripper is closed
        was_grasped = self._is_grasped.clone()
        newly_grasped = gripper_closed & close_to_piece & ~self._is_grasped
        self._is_grasped = (gripper_closed & close_to_piece) | (gripper_closed & was_grasped)

        # On new grasp: set piece to kinematic
        newly_released = was_grasped & ~self._is_grasped

        for env_idx in newly_grasped.nonzero(as_tuple=True)[0].tolist():
            piece_idx = self._target_piece_idx[env_idx]
            prim_path = self._piece_pool_paths[env_idx][piece_idx]
            self._set_piece_kinematic(prim_path, True)

        # On release: set piece back to dynamic
        for env_idx in newly_released.nonzero(as_tuple=True)[0].tolist():
            piece_idx = self._target_piece_idx[env_idx]
            prim_path = self._piece_pool_paths[env_idx][piece_idx]
            self._set_piece_kinematic(prim_path, False)

        # Update phase
        self._phase = torch.where(
            self._is_grasped, torch.ones_like(self._phase), self._phase
        )

        # -- Approach reward (exponential shaping) ---------------------------
        approach_dist = torch.norm(ee_pos - self._target_piece_pos, dim=-1)
        approach_reward = torch.exp(-approach_dist / 0.1)

        # -- Distance reduction bonus ----------------------------------------
        if not hasattr(self, "_prev_approach_dist"):
            self._prev_approach_dist = approach_dist.clone()
        dist_reduction = self._prev_approach_dist - approach_dist
        dist_reduction_reward = 5.0 * dist_reduction
        self._prev_approach_dist = approach_dist.clone()

        # -- Grasp bonus with quality (one-time) -----------------------------
        first_grasp = newly_grasped & ~self._has_been_grasped
        grasp_quality = torch.exp(-ee_to_piece_dist / cfg.grasp_quality_sigma)
        grasp_reward = first_grasp.float() * cfg.grasp_bonus * grasp_quality
        self._has_been_grasped = self._has_been_grasped | first_grasp

        # -- Transport reward (only when grasped, exponential) ---------------
        piece_to_target = torch.norm(
            self._target_piece_pos - self._target_square_pos, dim=-1
        )
        transport_reward = (
            5.0 * torch.exp(-piece_to_target / 0.1) * self._is_grasped.float()
        )

        # -- Success bonus ---------------------------------------------------
        success = piece_to_target < cfg.placement_tolerance
        success_reward = success.float() * cfg.success_bonus

        # -- Collision penalty -----------------------------------------------
        collision_penalty = self._compute_collision_penalty()

        return (
            approach_reward
            + dist_reduction_reward
            + grasp_reward
            + transport_reward
            + success_reward
            + cfg.collision_weight * collision_penalty
        )

    def _compute_collision_penalty(self) -> torch.Tensor:
        """Compute penalty for displacing non-target pieces.

        Returns per-env penalty (positive values, multiplied by negative weight).
        """
        if self._piece_pos is None or self._piece_initial_pos is None:
            return torch.zeros(self.num_envs, device=self.device)

        # Displacement of each piece from initial position
        displacement = torch.norm(
            self._piece_pos - self._piece_initial_pos, dim=-1
        )  # (N, MAX_PIECES)

        # Build mask: active pieces that are NOT the target
        mask = self._piece_active_mask.clone()  # (N, MAX_PIECES)
        # Zero out target piece in mask
        target_idx = self._target_piece_idx_tensor
        mask.scatter_(1, target_idx.unsqueeze(-1), False)

        # Apply displacement threshold (5mm noise filter)
        significant = displacement > self.cfg.collision_displacement_threshold

        # Sum displacement of significantly moved non-target pieces
        penalty = (displacement * significant.float() * mask.float()).sum(dim=-1)
        return penalty

    # ------------------------------------------------------------------ #
    # Termination
    # ------------------------------------------------------------------ #

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination conditions.

        Returns (terminated, truncated):
        - terminated: piece placed successfully, dropped off board, or
          non-target piece knocked off table
        - truncated: episode length exceeded
        """
        piece_to_target = torch.norm(
            self._target_piece_pos - self._target_square_pos, dim=-1
        )
        success = piece_to_target < self.cfg.placement_tolerance

        # Failure: target piece dropped below board surface
        piece_dropped = self._target_piece_pos[:, 2] < (
            TABLE_HEIGHT + self.cfg.board_drop_z
        )

        terminated = success | piece_dropped

        # Failure: non-target piece knocked off the table
        if self.cfg.knocked_off_terminates and self._piece_pos is not None:
            piece_z = self._piece_pos[:, :, 2]  # (N, MAX_PIECES)
            # Build non-target active mask
            non_target_mask = self._piece_active_mask.clone()
            non_target_mask.scatter_(
                1, self._target_piece_idx_tensor.unsqueeze(-1), False
            )
            fallen = (piece_z < (TABLE_HEIGHT - 0.05)) & non_target_mask
            pieces_knocked_off = fallen.any(dim=-1)
            terminated = terminated | pieces_knocked_off

        # Truncated: episode timeout
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    def _check_success(self) -> torch.Tensor:
        """Check if piece is at target within tolerance (required by parent)."""
        piece_to_target = torch.norm(
            self._target_piece_pos - self._target_square_pos, dim=-1
        )
        return piece_to_target < self.cfg.placement_tolerance
