"""Chess pick-and-place RL environment with rigid body physics.

An SO-101 robot arm learns to pick a chess piece and place it at a random
target location within its workspace. Built on LeIsaac's SingleArmTaskDirectEnv
(which extends IsaacLab's DirectRLEnv via RecorderEnhanceDirectRLEnv).

Physics features:
- Pieces are dynamic rigid bodies with friction-based grasping
- Gripper physically wraps around pieces; friction holds them against gravity
- Grasp detection via gripper joint position + proximity (LeIsaac pattern)
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
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

from leisaac.tasks.template.direct.single_arm_env import SingleArmTaskDirectEnv

from .chess_env_cfg import ChessPickPlaceEnvCfg
from .fen_placement import (
    BOARD_SQUARE_SIZE,
    fen_to_board_state,
    get_square_position,
    validate_fen,
)
from .chess_scene_cfg import TABLE_HEIGHT, RL_SQUARE_SIZE, BOARD_CENTER_OFFSET_Y

# Maximum number of pieces on a chess board
MAX_PIECES = 32

# Typed pool layout: each slot has a fixed piece type matching the standard
# chess starting position.  At reset, FEN pieces are mapped to slots of the
# matching type.  Promotions overflow to any remaining slot.
_POOL_LAYOUT: list[tuple[str, str, int]] = [
    # White (slots 0-15)
    ("P", "pawn_w",   8),   # 0-7
    ("R", "rook_w",   2),   # 8-9
    ("N", "knight_w", 2),   # 10-11
    ("B", "bishop_w", 2),   # 12-13
    ("Q", "queen_w",  1),   # 14
    ("K", "king_w",   1),   # 15
    # Black (slots 16-31)
    ("p", "pawn_b",   8),   # 16-23
    ("r", "rook_b",   2),   # 24-25
    ("n", "knight_b", 2),   # 26-27
    ("b", "bishop_b", 2),   # 28-29
    ("q", "queen_b",  1),   # 30
    ("k", "king_b",   1),   # 31
]

# Derived lookup tables (built once at import time)
SLOT_PIECE_TYPES: list[str] = []   # len=32, USD piece type name per slot
SLOT_FEN_CHARS: list[str] = []     # len=32, FEN char per slot
FEN_CHAR_TO_SLOTS: dict[str, list[int]] = {}  # fen char -> list of slot indices

# Map piece type name to geometry child prim name inside the USD.
# Referencing this child directly (primPath="/root/<child>") is required
# for Fabric/RTX rendering — referencing the defaultPrim "/root" does
# not render in Fabric.
GEOM_CHILD_NAMES: dict[str, str] = {
    "pawn_w": "P0", "pawn_b": "p0",
    "rook_w": "R0", "rook_b": "r0",
    "knight_w": "N0", "knight_b": "n0",
    "bishop_w": "B0", "bishop_b": "b0",
    "queen_w": "Q0", "queen_b": "q0",
    "king_w": "K0", "king_b": "k0",
}

_slot_idx = 0
for _fc, _pt, _cnt in _POOL_LAYOUT:
    _slots: list[int] = []
    for _ in range(_cnt):
        SLOT_PIECE_TYPES.append(_pt)
        SLOT_FEN_CHARS.append(_fc)
        _slots.append(_slot_idx)
        _slot_idx += 1
    FEN_CHAR_TO_SLOTS[_fc] = _slots

# Robot workspace for random target sampling
_MAX_REACH = 0.30   # 30cm conservative (within 35cm SO-101 max)
_MIN_REACH = 0.05   # 5cm minimum from robot base


class ChessPickPlaceEnv(SingleArmTaskDirectEnv):
    """RL environment for chess piece pick-and-place with an SO-101 robot arm.

    Observation space (21-dim):
        - Arm joint positions (5)
        - Gripper state (1)
        - Jaw (grasp center) position (3)
        - Jaw quaternion (4)
        - Target piece position relative to jaw (3)
        - Target square position relative to jaw (3)
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

        # Disable camera to save VRAM when not needed (e.g., large env counts)
        if not cfg.enable_camera:
            from leisaac.utils.env_utils import delete_attribute
            delete_attribute(cfg.scene, "front")

        # Load FEN dataset before super().__init__ (which calls _setup_scene)
        self._fen_list = self._load_fen_dataset(
            self._project_root / cfg.fen_dataset_path
        )

        # Per-env episode state (initialized in _reset_idx)
        self._target_piece_pos: Optional[torch.Tensor] = None   # (num_envs, 3)
        self._target_square_pos: Optional[torch.Tensor] = None  # (num_envs, 3)
        self._target_piece_initial_z: Optional[torch.Tensor] = None  # (num_envs,) float
        self._is_grasped: Optional[torch.Tensor] = None         # (num_envs,) bool
        self._has_been_grasped: Optional[torch.Tensor] = None   # (num_envs,) bool
        self._phase: Optional[torch.Tensor] = None              # (num_envs,) float
        self._gripper_cmd: Optional[torch.Tensor] = None        # (num_envs,) float
        self._prev_actions: Optional[torch.Tensor] = None       # (num_envs, act_dim)

        # Milestone curriculum: grasp count drives penalty ramp
        self._total_grasp_count: int = 0

        # Configurable piece count (1 for fast training, 32 for full board eval)
        self._num_pieces = cfg.num_pieces

        # Piece physics state
        self._piece_pos: Optional[torch.Tensor] = None          # (num_envs, num_pieces, 3)
        self._piece_initial_pos: Optional[torch.Tensor] = None  # (num_envs, num_pieces, 3)
        self._piece_active_mask: Optional[torch.Tensor] = None  # (num_envs, num_pieces) bool
        self._num_active_pieces: List[int] = []                  # per-env count

        # Piece prim management
        self._piece_pool_paths: List[List[str]] = []  # per-env list of prim paths
        self._target_piece_idx: List[int] = []        # index into piece pool per env
        self._target_piece_idx_tensor: Optional[torch.Tensor] = None  # (num_envs,) long

        self._board_scale = RL_SQUARE_SIZE / BOARD_SQUARE_SIZE

        # Track per-piece fen chars for reward/penalty computation
        self._piece_fen_chars: List[List[str]] = []  # per-env, per-slot

        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # Physics tensors view for batch position read/write (sim is running).
        # Uses omni.physics.tensors with torch frontend for GPU-accelerated
        # get_transforms() / set_transforms() on all piece rigid bodies.
        from omni.physics.tensors import create_simulation_view
        self._physics_sim_view = create_simulation_view("torch")
        self._piece_rigid_view = self._physics_sim_view.create_rigid_body_view(
            "/World/envs/env_*/Pieces/piece_*"
        )

        # Piece rotation quaternion: +90° around X axis to stand pieces
        # upright (Blender-exported USDs are modeled lying flat).
        # PhysX format [qx, qy, qz, qw]: sin(45°), 0, 0, cos(45°)
        _s = math.sin(math.pi / 4)
        _c = math.cos(math.pi / 4)
        self._piece_quat_xyzw = torch.tensor(
            [_s, 0.0, 0.0, _c], dtype=torch.float32, device=self.device
        )

        # Pre-allocate full-size buffers for physics API calls.
        # set_transforms/set_velocities require data shaped (view_count, N)
        # even when updating a subset via indices.
        view_count = self._piece_rigid_view.count  # num_envs * num_pieces
        self._full_transforms = torch.zeros(
            (view_count, 7), dtype=torch.float32, device=self.device
        )
        # Default quat: +90° X rotation for upright pieces
        self._full_transforms[:, 3] = _s   # qx
        self._full_transforms[:, 6] = _c   # qw
        self._full_velocities = torch.zeros(
            (view_count, 6), dtype=torch.float32, device=self.device
        )

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
        piece prims with rigid body physics.

        Architecture: Each piece slot is a single prim:
          piece_N/           — Xform with RigidBodyAPI + MassAPI + USD reference

        USD reference must be directly on the positioned prim (not a
        child) for Fabric/RTX to render it. RigidBodyAPI on piece_N
        means Fabric controls its transform — the referenced mesh
        content renders at the physics position. Convex hull collision
        is applied to Mesh prims inside piece_N once at init. No USD
        reference swapping at reset — all pieces keep their typed mesh.
        """
        super()._setup_scene()

        stage = omni.usd.get_context().get_stage()
        board_usd = self._usd_dir / "board.usd"
        cfg = self.cfg
        board_scale = self._board_scale

        # -- Increase GPU broadphase capacity for 2048 rigid bodies ----------
        physx_scene_path = "/World/PhysicsScene"
        physx_scene_prim = stage.GetPrimAtPath(physx_scene_path)
        if not physx_scene_prim.IsValid():
            physx_scene_prim = stage.DefinePrim(physx_scene_path, "PhysicsScene")
            UsdPhysics.Scene.Define(stage, physx_scene_path)
        physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physx_scene_prim)
        physx_scene_api.CreateGpuFoundLostPairsCapacityAttr(4 * 1024 * 1024)
        physx_scene_api.CreateGpuMaxRigidPatchCountAttr(256 * 1024)

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
            # Remove DomeLight from board USD
            board_light = f"{board_path}/env_light"
            bl_prim = stage.GetPrimAtPath(board_light)
            if bl_prim.IsValid():
                stage.RemovePrim(board_light)

            # No collision on the board mesh — the Table cuboid already
            # provides a flat kinematic collision surface at TABLE_HEIGHT.
            # Adding triangle mesh collision to the board caused instability.

            # -- Pre-create typed piece pool with physics ---------------------
            # Each slot has a FIXED piece type from SLOT_PIECE_TYPES (matching
            # the standard starting position).  USD reference is set directly
            # on piece_N (required for Fabric/RTX rendering).  At reset, FEN
            # pieces are mapped to slots of matching type.  Convex hull
            # collision is applied to Mesh prims inside piece_N (accounts
            # for +90° X rotation in the piece USD).
            # Explicitly create the Pieces container as a typed Xform so
            # Fabric traverses into it and discovers piece children.
            pieces_container = f"{env_ns}/Pieces"
            stage.DefinePrim(pieces_container, "Xform")

            env_pieces: List[str] = []
            env_fen_chars: List[str] = []
            for piece_idx in range(self._num_pieces):
                piece_path = f"{env_ns}/Pieces/piece_{piece_idx}"
                piece_prim = stage.DefinePrim(piece_path, "Xform")

                # USD reference: reference the geometry child directly
                # (e.g., primPath="/root/P0") — Fabric/RTX only renders
                # when the geometry child is the referenced prim, not
                # the "/root" wrapper.
                piece_type = SLOT_PIECE_TYPES[piece_idx % MAX_PIECES]
                piece_usd = self._usd_dir / f"{piece_type}.usd"
                geom_child = GEOM_CHILD_NAMES.get(piece_type)
                if piece_usd.exists() and geom_child:
                    piece_prim.GetReferences().AddReference(
                        str(piece_usd), primPath=f"/root/{geom_child}"
                    )

                # Apply physics directly on piece prim — Fabric will
                # control its transform, rendering the USD mesh at the
                # physics position.
                rb_api = UsdPhysics.RigidBodyAPI.Apply(piece_prim)
                rb_api.CreateRigidBodyEnabledAttr(True)
                rb_api.CreateKinematicEnabledAttr(True)
                mass_api = UsdPhysics.MassAPI.Apply(piece_prim)
                mass_api.CreateMassAttr(cfg.piece_mass_kg)
                physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(piece_prim)
                physx_rb.CreateLinearDampingAttr(cfg.piece_linear_damping)
                physx_rb.CreateAngularDampingAttr(cfg.piece_angular_damping)

                # Apply convex hull collision to Mesh prims inside piece.
                # Done once here — no reference swapping at reset, so
                # collision shapes stay valid for the entire sim run.
                self._apply_mesh_collision(stage, piece_path)

                # Apply physics material for friction-based grasping.
                mat_path = f"{piece_path}/PhysicsMaterial"
                mat_prim = stage.DefinePrim(mat_path, "Material")
                phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
                phys_mat.CreateStaticFrictionAttr(cfg.piece_friction)
                phys_mat.CreateDynamicFrictionAttr(cfg.piece_friction)
                phys_mat.CreateRestitutionAttr(cfg.piece_restitution)
                # Bind material to collision mesh prims
                for child in piece_prim.GetChildren():
                    if child.GetTypeName() == "Mesh":
                        binding = UsdShade.MaterialBindingAPI.Apply(child)
                        binding.Bind(
                            UsdShade.Material(mat_prim), "physics"
                        )

                # Initial position off-board at table height (kinematic).
                # Kept at table height instead of Z=-100 to stay within
                # Fabric's spatial broadphase bounds.
                # +90° X rotation stands pieces upright (Blender USDs are flat).
                xform = UsdGeom.Xformable(piece_prim)
                xform.ClearXformOpOrder()
                xform.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.5, TABLE_HEIGHT))
                xform.AddRotateXOp().Set(90.0)
                xform.AddScaleOp().Set(
                    Gf.Vec3f(board_scale, board_scale, board_scale)
                )

                env_pieces.append(piece_path)
                env_fen_chars.append("")

            self._piece_pool_paths.append(env_pieces)
            self._piece_fen_chars.append(env_fen_chars)

    # ------------------------------------------------------------------ #
    # Piece management helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply_mesh_collision(stage, piece_path: str):
        """Apply CollisionAPI + MeshCollisionAPI(convexHull) to mesh prims.

        With geometry-child references (primPath="/root/P0"), the Mesh prim
        (Chessset_NNN) is a direct child of piece_N. We apply collision to
        the Mesh prim so PhysX builds the convex hull in the mesh's own
        coordinate frame.

        Called once per piece in _setup_scene() — no reference swapping at
        reset, so collision shapes stay valid for the entire simulation.
        """
        parent = stage.GetPrimAtPath(piece_path)
        if not parent.IsValid():
            return
        for child in parent.GetChildren():
            if child.GetTypeName() == "Mesh":
                UsdPhysics.CollisionAPI.Apply(child)
                mesh_col = UsdPhysics.MeshCollisionAPI.Apply(child)
                mesh_col.CreateApproximationAttr("convexHull")

    def _configure_piece(self, prim_path: str, visible: bool = True):
        """Toggle a piece between dynamic (visible) and kinematic (hidden).

        Position is NOT set here — it's set in batch via RigidBodyView after
        all pieces are configured (see _reset_idx).

        USD references are never swapped — each slot keeps its typed mesh
        from _setup_scene().  This preserves collision shapes and avoids
        invalidating the PhysX simulation view.
        """
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return

        # Set kinematic mode directly on piece prim
        rb_api = UsdPhysics.RigidBodyAPI(prim)
        kinematic_attr = rb_api.GetKinematicEnabledAttr()
        if kinematic_attr:
            kinematic_attr.Set(not visible)

    def _hide_all_pieces(self, env_idx: int):
        """Hide all pieces by moving off-board and setting kinematic.

        Pieces are moved to Y=-0.5 (behind robot) at table height rather
        than Z=-100, staying within Fabric's spatial broadphase bounds.

        Uses physics API (set_transforms) when available, falls back to
        USD xform ops during the first reset (before sim view is created).
        """
        env_origin = self.scene.env_origins[env_idx]
        stage = omni.usd.get_context().get_stage()

        if hasattr(self, "_piece_rigid_view"):
            # Batch move via physics API (full-size tensors required)
            start = env_idx * self._num_pieces
            end = (env_idx + 1) * self._num_pieces
            indices = torch.arange(start, end, device=self.device)
            self._full_transforms[start:end, 0] = env_origin[0]
            self._full_transforms[start:end, 1] = env_origin[1] - 0.5
            self._full_transforms[start:end, 2] = TABLE_HEIGHT
            self._full_transforms[start:end, 3] = self._piece_quat_xyzw[0]
            self._full_transforms[start:end, 4] = 0.0
            self._full_transforms[start:end, 5] = 0.0
            self._full_transforms[start:end, 6] = self._piece_quat_xyzw[3]
            self._full_velocities[start:end] = 0.0
            self._piece_rigid_view.set_transforms(self._full_transforms, indices)
            self._piece_rigid_view.set_velocities(self._full_velocities, indices)
        else:
            # Fallback: USD xform ops (first reset, before physics view exists)
            env_origin_cpu = env_origin.cpu().numpy()
            for prim_path in self._piece_pool_paths[env_idx]:
                prim = stage.GetPrimAtPath(prim_path)
                if prim.IsValid():
                    xform = UsdGeom.Xformable(prim)
                    xform.ClearXformOpOrder()
                    xform.AddTranslateOp().Set(
                        Gf.Vec3d(float(env_origin_cpu[0]), float(env_origin_cpu[1]) - 0.5, TABLE_HEIGHT)
                    )
                    xform.AddRotateXOp().Set(90.0)
                    s = self._board_scale
                    xform.AddScaleOp().Set(Gf.Vec3f(s, s, s))

        # Set all pieces kinematic
        for prim_path in self._piece_pool_paths[env_idx]:
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                rb_api = UsdPhysics.RigidBodyAPI(prim)
                kinematic_attr = rb_api.GetKinematicEnabledAttr()
                if kinematic_attr:
                    kinematic_attr.Set(True)

    def _make_transforms(self, positions: torch.Tensor) -> torch.Tensor:
        """Build (N, 7) PhysX transform tensor [x, y, z, qx, qy, qz, qw]."""
        n = positions.shape[0]
        quats = self._piece_quat_xyzw.unsqueeze(0).expand(n, -1)
        return torch.cat([positions, quats], dim=-1)

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
            self._target_piece_initial_z = torch.zeros(
                self.num_envs, dtype=torch.float32, device=device
            )
            self._is_grasped = torch.zeros(
                self.num_envs, dtype=torch.bool, device=device
            )
            self._was_grasped_prev = torch.zeros(
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
                (self.num_envs, self._num_pieces, 3), device=device
            )
            self._piece_initial_pos = torch.zeros(
                (self.num_envs, self._num_pieces, 3), device=device
            )
            self._piece_active_mask = torch.zeros(
                (self.num_envs, self._num_pieces), dtype=torch.bool, device=device
            )
            self._target_piece_idx_tensor = torch.zeros(
                self.num_envs, dtype=torch.long, device=device
            )
            self._num_active_pieces = [0] * self.num_envs

        for env_idx in env_ids.tolist():
            # Hide all existing pieces (sets them kinematic)
            self._hide_all_pieces(env_idx)

            # Compute board center and env origin in WORLD coordinates
            env_origin = self.scene.env_origins[env_idx].cpu().numpy()
            board_center = env_origin + np.array(
                [0.0, BOARD_CENTER_OFFSET_Y, TABLE_HEIGHT]
            )

            # Initialize all slots as hidden
            self._piece_active_mask[env_idx] = False
            far_pos = np.array([env_origin[0], env_origin[1] - 0.5, TABLE_HEIGHT])
            far_pos_t = torch.tensor(far_pos, dtype=torch.float32, device=device)
            self._piece_pos[env_idx] = far_pos_t.unsqueeze(0).expand(
                self._num_pieces, -1
            ).clone()
            self._piece_initial_pos[env_idx] = self._piece_pos[env_idx].clone()
            for i in range(self._num_pieces):
                self._piece_fen_chars[env_idx][i] = ""

            if self._num_pieces == 1:
                # Fast path: single piece at random board square
                col = random.randint(0, 7)
                row = random.randint(0, 7)
                square = chr(ord("A") + col) + str(row + 1)
                src_pos = get_square_position(
                    square, board_center=board_center, square_size=RL_SQUARE_SIZE
                )
                pos_t = torch.tensor(src_pos, dtype=torch.float32, device=device)
                self._piece_pos[env_idx, 0] = pos_t
                self._piece_initial_pos[env_idx, 0] = pos_t
                self._piece_active_mask[env_idx, 0] = True
                self._piece_fen_chars[env_idx][0] = "P"
                prim_path = self._piece_pool_paths[env_idx][0]
                self._configure_piece(prim_path, visible=True)
                self._num_active_pieces[env_idx] = 1
                target_pool_idx = 0
            else:
                # Full path: FEN-based placement with typed slots
                fen = random.choice(self._fen_list)
                board_state = fen_to_board_state(fen)

                # Map FEN pieces to typed slots (capped at num_pieces)
                available: dict[str, list[int]] = {}
                for fc, slots in FEN_CHAR_TO_SLOTS.items():
                    available[fc] = [s for s in slots if s < self._num_pieces]
                used_slots: set[int] = set()
                assignments: List[Tuple[int, str, str, np.ndarray]] = []
                overflow: List[Tuple[str, str, np.ndarray]] = []

                for square, fen_char in board_state.items():
                    pos = get_square_position(
                        square, board_center=board_center, square_size=RL_SQUARE_SIZE
                    )
                    if available.get(fen_char):
                        slot_idx = available[fen_char].pop(0)
                        assignments.append((slot_idx, square, fen_char, pos))
                        used_slots.add(slot_idx)
                    else:
                        overflow.append((square, fen_char, pos))

                remaining = [i for i in range(self._num_pieces) if i not in used_slots]
                for square, fen_char, pos in overflow:
                    if remaining:
                        slot_idx = remaining.pop(0)
                        assignments.append((slot_idx, square, fen_char, pos))
                        used_slots.add(slot_idx)

                # Place assigned pieces
                piece_squares: List[Tuple[int, str, str, np.ndarray]] = []
                for slot_idx, square, fen_char, pos in assignments:
                    prim_path = self._piece_pool_paths[env_idx][slot_idx]
                    self._configure_piece(prim_path, visible=True)

                    pos_t = torch.tensor(pos, dtype=torch.float32, device=device)
                    self._piece_pos[env_idx, slot_idx] = pos_t
                    self._piece_initial_pos[env_idx, slot_idx] = pos_t
                    self._piece_active_mask[env_idx, slot_idx] = True
                    self._piece_fen_chars[env_idx][slot_idx] = fen_char
                    piece_squares.append((slot_idx, square, fen_char, pos))

                self._num_active_pieces[env_idx] = len(assignments)

                if piece_squares:
                    src_list_idx = random.randrange(len(piece_squares))
                    slot_idx, src_sq, src_char, src_pos = piece_squares[src_list_idx]
                    target_pool_idx = slot_idx
                else:
                    src_pos = get_square_position(
                        "E2", board_center=board_center, square_size=RL_SQUARE_SIZE
                    )
                    target_pool_idx = 0

            # Batch set all piece positions via physics API (if available)
            if hasattr(self, "_piece_rigid_view"):
                start = env_idx * self._num_pieces
                end = (env_idx + 1) * self._num_pieces
                indices = torch.arange(start, end, device=device)
                self._full_transforms[start:end, :3] = self._piece_pos[env_idx]
                self._full_transforms[start:end, 3] = self._piece_quat_xyzw[0]
                self._full_transforms[start:end, 4] = 0.0
                self._full_transforms[start:end, 5] = 0.0
                self._full_transforms[start:end, 6] = self._piece_quat_xyzw[3]
                self._full_velocities[start:end] = 0.0
                self._piece_rigid_view.set_transforms(
                    self._full_transforms, indices
                )
                self._piece_rigid_view.set_velocities(
                    self._full_velocities, indices
                )
            else:
                # Fallback: USD xform ops (first reset, before physics view)
                stage = omni.usd.get_context().get_stage()
                for pool_idx in range(self._num_pieces):
                    pos = self._piece_pos[env_idx, pool_idx].cpu().numpy()
                    prim_path = self._piece_pool_paths[env_idx][pool_idx]
                    prim = stage.GetPrimAtPath(prim_path)
                    if prim.IsValid():
                        xform = UsdGeom.Xformable(prim)
                        xform.ClearXformOpOrder()
                        xform.AddTranslateOp().Set(
                            Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
                        )
                        xform.AddRotateXOp().Set(90.0)
                        s = self._board_scale
                        xform.AddScaleOp().Set(Gf.Vec3f(s, s, s))

            target_pos = self._pick_random_target(env_origin)

            # 6. Store episode state
            # Offset piece target Z to grasp center (physics pos is the base)
            src_pos_t = torch.tensor(
                src_pos, dtype=torch.float32, device=device
            )
            src_pos_t[2] += self.cfg.piece_grasp_z_offset
            self._target_piece_pos[env_idx] = src_pos_t
            self._target_square_pos[env_idx] = torch.tensor(
                target_pos, dtype=torch.float32, device=device
            )
            self._target_piece_initial_z[env_idx] = src_pos_t[2]
            self._is_grasped[env_idx] = False
            self._was_grasped_prev[env_idx] = False
            self._has_been_grasped[env_idx] = False
            self._phase[env_idx] = 0.0

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

    # _apply_action is inherited: sends self.actions as joint position targets

    # ------------------------------------------------------------------ #
    # Piece position readback
    # ------------------------------------------------------------------ #

    def _update_piece_positions(self):
        """Read piece positions from physics simulation (single GPU op).

        All active pieces are read from physics — with friction-based
        grasping, even grasped pieces are controlled by the physics engine.
        """
        if self._piece_pos is None or not hasattr(self, "_piece_rigid_view"):
            return

        # Single GPU read: get_transforms() returns (count, 7) [x,y,z,qx,qy,qz,qw]
        all_transforms = self._piece_rigid_view.get_transforms()
        all_pos = all_transforms[:, :3].reshape(self.num_envs, self._num_pieces, 3)

        # Update all active pieces from physics (no grasped-mask needed)
        self._piece_pos[self._piece_active_mask] = all_pos[self._piece_active_mask]

        # Update target piece position from physics (piece moves with gripper).
        # Offset Z upward to the piece's grasp center (physics pos is the base).
        grasp_z = self.cfg.piece_grasp_z_offset
        for e in range(self.num_envs):
            idx = self._target_piece_idx[e]
            self._target_piece_pos[e] = self._piece_pos[e, idx]
            self._target_piece_pos[e, 2] += grasp_z

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #

    def _get_observations(self) -> dict:
        """Compute observation dict."""
        # Update piece positions from physics
        self._update_piece_positions()

        # Skip parent's _get_observations when camera is disabled — it tries
        # to read from the "front" sensor which doesn't exist without camera.
        if self.cfg.enable_camera:
            obs = super()._get_observations()
        else:
            obs = {"policy": {}}

        n_arm = self.cfg.num_arm_joints  # 5
        robot = self.scene["robot"]

        # Robot state
        joint_pos = robot.data.joint_pos[:, :n_arm]           # (N, 5)
        gripper_pos = robot.data.joint_pos[:, n_arm:n_arm+1]  # (N, 1)

        # Jaw (grasp center) position — matches the frame used for rewards
        # and grasp detection.  ee_frame index 1 = jaw link + offset,
        # approximating the point between the gripper fingers.
        ee_frame = self.scene["ee_frame"]
        jaw_pos = ee_frame.data.target_pos_w[:, 1, :]         # (N, 3)
        jaw_quat = ee_frame.data.target_quat_w[:, 1, :]       # (N, 4)

        # Relative positions to targets (from jaw/grasp center, not gripper link)
        target_piece_rel = self._target_piece_pos - jaw_pos    # (N, 3)
        target_square_rel = self._target_square_pos - jaw_pos  # (N, 3)

        # Grasp and phase flags
        is_grasped = self._is_grasped.float().unsqueeze(-1)    # (N, 1)
        phase = self._phase.unsqueeze(-1)                       # (N, 1)

        rl_obs = torch.cat(
            [
                joint_pos,
                gripper_pos,
                jaw_pos,
                jaw_quat,
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
        """Compute per-step reward with phase-gated approach and grasp-gated transport.

        Reward terms:
        - Approach: tanh kernel (coarse + fine), zeroed after grasp
        - Close-on-piece: gripper closure × alignment — bridges approach → grasp
        - Grasp bonus: one-time when first grasped (exp quality)
        - Lift: binary per-step bonus when piece is above initial Z + threshold
        - Transport: tanh kernel (coarse + fine), gated on grasp (not lift)
        - Success: +100 when piece placed within tolerance
        - Collision penalty: curriculum ramp from near-zero to final weight
        - Action smoothness: ||a_t - a_{t-1}||² penalty, curriculum ramp
        """
        robot = self.scene["robot"]
        ee_pos = robot.data.body_pos_w[:, -1, :]
        cfg = self.cfg

        # -- Grasp state update (matches LeIsaac object_grasped exactly) ------
        # No latch — checks fresh each step like LeIsaac.
        # Uses _target_piece_pos (piece center with Z offset) everywhere,
        # matching LeIsaac's object.data.root_pos_w (center of mass).
        ee_frame = self.scene["ee_frame"]
        jaw_pos = ee_frame.data.target_pos_w[:, 1, :]  # jaw frame = grasp center
        n_arm = cfg.num_arm_joints
        gripper_joint_pos = robot.data.joint_pos[:, n_arm]  # same as [:, -1]

        jaw_to_piece = torch.norm(self._target_piece_pos - jaw_pos, dim=-1)
        self._is_grasped = (jaw_to_piece < cfg.grasp_threshold) & (
            gripper_joint_pos < cfg.gripper_closed_threshold
        )
        newly_grasped = self._is_grasped & ~self._was_grasped_prev
        self._was_grasped_prev = self._is_grasped.clone()

        # Phase tracks current grasp state
        self._phase = self._is_grasped.float()

        # -- 1. Approach reward (single tanh kernel, matching isaac_so_arm101) -
        # Always active (no mask).  Uses jaw-to-piece-center distance.
        approach_dist = jaw_to_piece
        approach_reward = cfg.approach_weight * (
            1.0 - torch.tanh(approach_dist / cfg.approach_std)
        )

        # -- 2. Grasp milestone tracking (for curriculum) --------------------
        # No close-on-piece or grasp bonus: like isaac_so_arm101, the policy
        # discovers grasping purely from the lift reward (object must rise).
        # We still track first-grasp events to ramp collision/action penalties.
        first_grasp = newly_grasped & ~self._has_been_grasped
        self._has_been_grasped = self._has_been_grasped | first_grasp
        n_new_grasps = first_grasp.sum().item()
        if n_new_grasps > 0:
            self._total_grasp_count += int(n_new_grasps)

        # -- 3. Lift reward (height-gated only, matching isaac_so_arm101) -----
        # Gate on piece HEIGHT only — no grasp detection needed.  If the
        # piece is above the threshold, gravity ensures it's being held.
        # This avoids false negatives from strict jaw-proximity checks.
        piece_z = self._target_piece_pos[:, 2]
        is_lifted = piece_z > (self._target_piece_initial_z + cfg.lift_threshold)
        lift_reward = cfg.lift_weight * is_lifted.float()

        # -- 4. Transport reward (tanh kernel, gated on LIFT) ----------------
        # Must be gated on lift, not just grasp: with friction-based grasping
        # the piece stays on the table until physically lifted.  Gating on
        # grasp alone gives free transport reward for a stationary piece.
        piece_to_target = torch.norm(
            self._target_piece_pos - self._target_square_pos, dim=-1
        )
        transport_reward = is_lifted.float() * (
            cfg.transport_weight
            * (1.0 - torch.tanh(piece_to_target / cfg.transport_std))
            + cfg.transport_fine_weight
            * (1.0 - torch.tanh(piece_to_target / cfg.transport_fine_std))
        )

        # -- 5. Success bonus (gated on lift — piece must be placed, not knocked)
        success = (piece_to_target < cfg.placement_tolerance) & is_lifted
        success_reward = success.float() * cfg.success_bonus

        # -- 6. Collision penalty (milestone curriculum: ramps with grasp count)
        t_col = min(1.0, self._total_grasp_count / max(1, cfg.collision_milestone_grasps))
        collision_w = (
            cfg.collision_weight_start
            + (cfg.collision_weight - cfg.collision_weight_start) * t_col
        )
        collision_penalty = self._compute_collision_penalty()

        # -- 7. Action rate penalty (milestone curriculum) --------------------
        t_reg = min(1.0, self._total_grasp_count / max(1, cfg.action_rate_milestone_grasps))
        action_rate_w = (
            cfg.action_rate_start
            + (cfg.action_rate_weight - cfg.action_rate_start) * t_reg
        )
        if self._prev_actions is None:
            self._prev_actions = self.actions.clone()
        action_diff_sq = (self.actions - self._prev_actions).pow(2).sum(dim=-1)
        self._prev_actions = self.actions.clone()

        return (
            approach_reward
            + lift_reward
            + transport_reward
            + success_reward
            + collision_w * collision_penalty
            + action_rate_w * action_diff_sq
        )

    def _compute_collision_penalty(self) -> torch.Tensor:
        """Compute penalty for displacing non-target pieces.

        Returns per-env penalty (positive values, multiplied by negative weight).
        """
        if self._piece_pos is None or self._piece_initial_pos is None:
            return torch.zeros(self.num_envs, device=self.device)
        if self._num_pieces == 1:
            return torch.zeros(self.num_envs, device=self.device)

        # Displacement of each piece from initial position
        displacement = torch.norm(
            self._piece_pos - self._piece_initial_pos, dim=-1
        )  # (N, num_pieces)

        # Build mask: active pieces that are NOT the target
        mask = self._piece_active_mask.clone()  # (N, num_pieces)
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
        if self.cfg.knocked_off_terminates and self._num_pieces > 1 and self._piece_pos is not None:
            piece_z = self._piece_pos[:, :, 2]  # (N, num_pieces)
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

    # ------------------------------------------------------------------ #
    # Camera access (for Reason2 critic)
    # ------------------------------------------------------------------ #

    def get_camera_rgb(self) -> Optional[torch.Tensor]:
        """Read RGB data from the front camera.

        Returns (num_envs, H, W, 3) float32 tensor in [0, 1], or None if
        camera data is unavailable.
        """
        try:
            camera = self.scene["front"]
            rgb = camera.data.output["rgb"]
            # TiledCamera returns (N, H, W, 4) RGBA float32 — take RGB
            if rgb is not None and rgb.numel() > 0:
                return rgb[:, :, :, :3]
        except (KeyError, AttributeError, RuntimeError):
            pass
        return None
