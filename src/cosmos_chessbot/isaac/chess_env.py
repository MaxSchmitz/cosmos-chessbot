"""Chess pick-and-place RL environment with rigid body physics.

An SO-101 robot arm learns to pick a chess piece and place it at a random
target location within its workspace. Built on LeIsaac's SingleArmTaskDirectEnv
(which extends IsaacLab's DirectRLEnv via RecorderEnhanceDirectRLEnv).

Physics features:
- Pieces are rigid bodies (collision via USD mesh geometry)
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
        self._target_piece_initial_z: Optional[torch.Tensor] = None  # (num_envs,) float
        self._is_grasped: Optional[torch.Tensor] = None         # (num_envs,) bool
        self._has_been_grasped: Optional[torch.Tensor] = None   # (num_envs,) bool
        self._phase: Optional[torch.Tensor] = None              # (num_envs,) float
        self._gripper_cmd: Optional[torch.Tensor] = None        # (num_envs,) float
        self._prev_actions: Optional[torch.Tensor] = None       # (num_envs, act_dim)

        # Milestone curriculum: grasp count drives penalty ramp
        self._total_grasp_count: int = 0

        # Piece physics state
        self._piece_pos: Optional[torch.Tensor] = None          # (num_envs, MAX_PIECES, 3)
        self._piece_initial_pos: Optional[torch.Tensor] = None  # (num_envs, MAX_PIECES, 3)
        self._piece_active_mask: Optional[torch.Tensor] = None  # (num_envs, MAX_PIECES) bool
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

        # Pre-compute identity quaternion in PhysX format [qx, qy, qz, qw]
        self._identity_quat_xyzw = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=self.device
        )

        # Pre-allocate full-size buffers for physics API calls.
        # set_transforms/set_velocities require data shaped (view_count, N)
        # even when updating a subset via indices.
        view_count = self._piece_rigid_view.count  # num_envs * MAX_PIECES
        self._full_transforms = torch.zeros(
            (view_count, 7), dtype=torch.float32, device=self.device
        )
        self._full_transforms[:, 6] = 1.0  # identity quat w component
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

        Architecture: Each piece slot has this hierarchy:
          piece_N/           — Xform with RigidBodyAPI + MassAPI (physics)
            Visual/          — Xform with USD reference (pawn_w, set once)

        RigidBodyAPI on piece_N means Fabric controls its transform —
        the Visual child inherits this, coupling physics and visual
        positions. Convex hull collision is applied to the actual Mesh
        prims inside Visual once at init. No USD reference swapping at
        reset — all pieces stay as pawn_w. The RL policy uses a 21-dim
        obs vector (not images), so visual type doesn't matter.
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
            # the standard starting position).  Visual child holds the USD
            # reference (set once, never swapped).  At reset, FEN pieces are
            # mapped to slots of matching type.  Convex hull collision is
            # applied to the Mesh prim inside Visual (accounts for +90° X
            # rotation in the piece USD).
            env_pieces: List[str] = []
            env_fen_chars: List[str] = []
            for piece_idx in range(MAX_PIECES):
                piece_path = f"{env_ns}/Pieces/piece_{piece_idx}"
                piece_prim = stage.DefinePrim(piece_path, "Xform")

                # Visual child — typed USD reference set once, never swapped.
                piece_type = SLOT_PIECE_TYPES[piece_idx]
                piece_usd = self._usd_dir / f"{piece_type}.usd"
                visual_path = f"{piece_path}/Visual"
                visual_prim = stage.DefinePrim(visual_path, "Xform")
                if piece_usd.exists():
                    visual_prim.GetReferences().AddReference(str(piece_usd))

                # Apply physics directly on piece prim — Fabric will
                # control its transform, and Visual inherits it.
                rb_api = UsdPhysics.RigidBodyAPI.Apply(piece_prim)
                rb_api.CreateRigidBodyEnabledAttr(True)
                rb_api.CreateKinematicEnabledAttr(True)
                mass_api = UsdPhysics.MassAPI.Apply(piece_prim)
                mass_api.CreateMassAttr(cfg.piece_mass_kg)
                physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(piece_prim)
                physx_rb.CreateLinearDampingAttr(cfg.piece_linear_damping)
                physx_rb.CreateAngularDampingAttr(cfg.piece_angular_damping)

                # Zero internal translate + remove DomeLight (on Visual child)
                self._zero_internal_piece_translate(stage, visual_path)
                light_path = f"{visual_path}/env_light"
                light_prim = stage.GetPrimAtPath(light_path)
                if light_prim.IsValid():
                    light_prim.SetActive(False)

                # Apply convex hull collision to the actual Mesh prims inside
                # Visual. Done once here — no reference swapping at reset, so
                # these collision shapes stay valid for the entire sim run.
                self._apply_mesh_collision(stage, visual_path)

                # Initial position far away (invisible + kinematic)
                xform = UsdGeom.Xformable(piece_prim)
                xform.ClearXformOpOrder()
                xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -100.0))
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
    def _zero_internal_piece_translate(stage, prim_path: str):
        """Zero out the internal translate offset on the USD reference's root child.

        Piece USDs (exported from Blender) have an internal Xform child
        (P0, Q0, etc.) with a translate that positions the piece at its
        original chess-set location. We zero this so the piece sits at
        the prim's origin, and let the prim's own translate handle
        world positioning.

        Skips non-USD children (env_light, _materials) that are local
        to our scene setup.
        """
        parent = stage.GetPrimAtPath(prim_path)
        if not parent.IsValid():
            return
        # USD reference children have names like P0, Q0, q0, etc.
        # (single uppercase/lowercase letter + digit). Skip our own
        # children (env_light, _materials).
        for child in parent.GetChildren():
            name = child.GetName()
            if name in ("env_light", "_materials"):
                continue
            child_xf = UsdGeom.Xformable(child)
            if not child_xf:
                continue
            for op in child_xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
                    break

    @staticmethod
    def _apply_mesh_collision(stage, visual_path: str):
        """Apply CollisionAPI + MeshCollisionAPI(convexHull) to mesh prims.

        Piece USDs have a geometry Xform child (P0, Q0, etc.) containing a Mesh
        prim (Chessset_NNN). We apply collision to the Mesh prim directly so
        PhysX builds the convex hull in the mesh's own coordinate frame, which
        includes the +90° X rotation that stands the piece upright.

        Called once per piece in _setup_scene() — no reference swapping at
        reset, so collision shapes stay valid for the entire simulation.
        """
        parent = stage.GetPrimAtPath(visual_path)
        if not parent.IsValid():
            return
        for child in parent.GetChildren():
            name = child.GetName()
            if name in ("env_light", "_materials"):
                continue
            # This is the geometry Xform (P0, Q0, K0, etc.) — find Mesh children
            for grandchild in child.GetChildren():
                if grandchild.GetTypeName() == "Mesh":
                    UsdPhysics.CollisionAPI.Apply(grandchild)
                    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(grandchild)
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
        """Hide all pieces by moving to Z=-100 and setting kinematic.

        Uses physics API (set_transforms) when available, falls back to
        USD xform ops during the first reset (before sim view is created).
        """
        env_origin = self.scene.env_origins[env_idx]
        stage = omni.usd.get_context().get_stage()

        if hasattr(self, "_piece_rigid_view"):
            # Batch move via physics API (full-size tensors required)
            start = env_idx * MAX_PIECES
            end = (env_idx + 1) * MAX_PIECES
            indices = torch.arange(start, end, device=self.device)
            self._full_transforms[start:end, 0] = env_origin[0]
            self._full_transforms[start:end, 1] = env_origin[1]
            self._full_transforms[start:end, 2] = -100.0
            self._full_transforms[start:end, 3:6] = 0.0
            self._full_transforms[start:end, 6] = 1.0
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
                        Gf.Vec3d(float(env_origin_cpu[0]), float(env_origin_cpu[1]), -100.0)
                    )
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

    def _set_piece_kinematic(self, prim_path: str, kinematic: bool):
        """Set a piece's kinematic mode directly on the piece prim."""
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            rb_api = UsdPhysics.RigidBodyAPI(prim)
            kinematic_attr = rb_api.GetKinematicEnabledAttr()
            if kinematic_attr:
                kinematic_attr.Set(kinematic)

    def _make_transforms(self, positions: torch.Tensor) -> torch.Tensor:
        """Build (N, 7) PhysX transform tensor [x, y, z, qx, qy, qz, qw]."""
        n = positions.shape[0]
        quats = self._identity_quat_xyzw.unsqueeze(0).expand(n, -1)
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

            # 4. Map FEN pieces to typed slots --------------------------------
            #    Each FEN piece is assigned to a slot whose fixed mesh type
            #    matches (e.g., 'R' → rook_w slot).  Overflow from promotions
            #    goes to any remaining slot (wrong visual, acceptable for RL).
            self._piece_active_mask[env_idx] = False
            far_pos = np.array([env_origin[0], env_origin[1], -100.0])
            far_pos_t = torch.tensor(far_pos, dtype=torch.float32, device=device)
            # Initialize all slots as hidden
            self._piece_pos[env_idx] = far_pos_t.unsqueeze(0).expand(
                MAX_PIECES, -1
            ).clone()
            self._piece_initial_pos[env_idx] = self._piece_pos[env_idx].clone()
            for i in range(MAX_PIECES):
                self._piece_fen_chars[env_idx][i] = ""

            # Build per-type availability (copy so we can pop)
            available: dict[str, list[int]] = {
                fc: list(slots) for fc, slots in FEN_CHAR_TO_SLOTS.items()
            }
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

            # Overflow (promotions): assign to any remaining slot
            remaining = [i for i in range(MAX_PIECES) if i not in used_slots]
            for square, fen_char, pos in overflow:
                if remaining:
                    slot_idx = remaining.pop(0)
                    assignments.append((slot_idx, square, fen_char, pos))
                    used_slots.add(slot_idx)

            # Place assigned pieces (set dynamic, store position)
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

            # Batch set all piece positions via physics API (if available)
            if hasattr(self, "_piece_rigid_view"):
                start = env_idx * MAX_PIECES
                end = (env_idx + 1) * MAX_PIECES
                indices = torch.arange(start, end, device=device)
                self._full_transforms[start:end, :3] = self._piece_pos[env_idx]
                self._full_transforms[start:end, 3:6] = 0.0
                self._full_transforms[start:end, 6] = 1.0
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
                for pool_idx in range(MAX_PIECES):
                    pos = self._piece_pos[env_idx, pool_idx].cpu().numpy()
                    prim_path = self._piece_pool_paths[env_idx][pool_idx]
                    prim = stage.GetPrimAtPath(prim_path)
                    if prim.IsValid():
                        xform = UsdGeom.Xformable(prim)
                        xform.ClearXformOpOrder()
                        xform.AddTranslateOp().Set(
                            Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
                        )
                        s = self._board_scale
                        xform.AddScaleOp().Set(Gf.Vec3f(s, s, s))

            # 5. Pick a random piece and random target location
            if piece_squares:
                src_list_idx = random.randrange(len(piece_squares))
                slot_idx, src_sq, src_char, src_pos = piece_squares[src_list_idx]
                target_pool_idx = slot_idx
            else:
                src_pos = get_square_position(
                    "E2", board_center=board_center, square_size=RL_SQUARE_SIZE
                )
                target_pool_idx = 0

            target_pos = self._pick_random_target(env_origin)

            # 6. Store episode state
            src_pos_t = torch.tensor(
                src_pos, dtype=torch.float32, device=device
            )
            self._target_piece_pos[env_idx] = src_pos_t
            self._target_square_pos[env_idx] = torch.tensor(
                target_pos, dtype=torch.float32, device=device
            )
            self._target_piece_initial_z[env_idx] = src_pos_t[2]
            self._is_grasped[env_idx] = False
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

        # Update grasped pieces to follow EE before physics step
        self._update_grasped_pieces()

    # _apply_action is inherited: sends self.actions as joint position targets

    # ------------------------------------------------------------------ #
    # Grasped piece following
    # ------------------------------------------------------------------ #

    def _update_grasped_pieces(self):
        """Move grasped pieces to follow the end-effector position.

        Sets grasped pieces to kinematic and updates their position via
        the physics simulation API (RigidPrimView).
        """
        if self._is_grasped is None:
            return

        grasped_envs = self._is_grasped.nonzero(as_tuple=True)[0]
        if len(grasped_envs) == 0:
            return

        robot = self.scene["robot"]
        ee_pos = robot.data.body_pos_w[:, -1, :]  # (N, 3)

        # Compute positions for all grasped pieces (EE + Z offset)
        positions = ee_pos[grasped_envs].clone()
        positions[:, 2] += self.cfg.grasp_offset_z

        # Compute global indices into the RigidPrimView
        piece_indices = self._target_piece_idx_tensor[grasped_envs]
        global_indices = grasped_envs * MAX_PIECES + piece_indices

        # Batch set via physics API (full-size tensors required)
        self._full_transforms[global_indices, :3] = positions
        self._full_transforms[global_indices, 3:6] = 0.0
        self._full_transforms[global_indices, 6] = 1.0
        self._full_velocities[global_indices] = 0.0
        self._piece_rigid_view.set_transforms(
            self._full_transforms, global_indices
        )
        self._piece_rigid_view.set_velocities(
            self._full_velocities, global_indices,
        )

        # Update tracked positions
        self._target_piece_pos[grasped_envs] = positions
        for i, env_idx in enumerate(grasped_envs.tolist()):
            piece_idx = self._target_piece_idx[env_idx]
            self._piece_pos[env_idx, piece_idx] = positions[i]

    # ------------------------------------------------------------------ #
    # Piece position readback
    # ------------------------------------------------------------------ #

    def _update_piece_positions(self):
        """Read piece positions from physics simulation (single GPU op).

        For grasped pieces, position is already updated by _update_grasped_pieces.
        For non-grasped active pieces, read from RigidPrimView.
        """
        if self._piece_pos is None or not hasattr(self, "_piece_rigid_view"):
            return

        # Single GPU read: get_transforms() returns (count, 7) [x,y,z,qx,qy,qz,qw]
        all_transforms = self._piece_rigid_view.get_transforms()
        all_pos = all_transforms[:, :3].reshape(self.num_envs, MAX_PIECES, 3)

        # Build mask: update active, non-grasped pieces only
        grasped_mask = torch.zeros(
            (self.num_envs, MAX_PIECES), dtype=torch.bool, device=self.device,
        )
        for e in range(self.num_envs):
            if self._is_grasped[e]:
                grasped_mask[e, self._target_piece_idx[e]] = True
        update_mask = self._piece_active_mask & ~grasped_mask

        self._piece_pos[update_mask] = all_pos[update_mask]

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
        """Compute per-step reward using tanh kernels, lift gating, and curriculum.

        Reward terms:
        - Approach: tanh kernel (coarse + fine) — dense signal to reach piece
        - Grasp bonus: one-time when first grasped (exp quality)
        - Lift: binary per-step reward when piece is above initial Z + threshold
        - Transport: tanh kernel (coarse + fine), gated on lift
        - Success: +100 when piece placed within tolerance
        - Collision penalty: curriculum ramp from near-zero to final weight
        - Action smoothness: ||a_t - a_{t-1}||² penalty, curriculum ramp
        """
        robot = self.scene["robot"]
        ee_pos = robot.data.body_pos_w[:, -1, :]
        cfg = self.cfg

        # -- Grasp state update ----------------------------------------------
        gripper_closed = self._gripper_cmd >= 0.5
        ee_to_piece_dist = torch.norm(ee_pos - self._target_piece_pos, dim=-1)
        close_to_piece = ee_to_piece_dist < cfg.grasp_threshold

        was_grasped = self._is_grasped.clone()
        newly_grasped = gripper_closed & close_to_piece & ~self._is_grasped
        self._is_grasped = (
            (gripper_closed & close_to_piece) | (gripper_closed & was_grasped)
        )
        newly_released = was_grasped & ~self._is_grasped

        for env_idx in newly_grasped.nonzero(as_tuple=True)[0].tolist():
            piece_idx = self._target_piece_idx[env_idx]
            prim_path = self._piece_pool_paths[env_idx][piece_idx]
            self._set_piece_kinematic(prim_path, True)

        for env_idx in newly_released.nonzero(as_tuple=True)[0].tolist():
            piece_idx = self._target_piece_idx[env_idx]
            prim_path = self._piece_pool_paths[env_idx][piece_idx]
            self._set_piece_kinematic(prim_path, False)

        # Phase tracks current grasp state (not latched)
        self._phase = self._is_grasped.float()

        # -- 1. Approach reward (tanh kernel, two scales) --------------------
        approach_dist = torch.norm(ee_pos - self._target_piece_pos, dim=-1)
        approach_reward = (
            cfg.approach_weight * (1.0 - torch.tanh(approach_dist / cfg.approach_std))
            + cfg.approach_fine_weight
            * (1.0 - torch.tanh(approach_dist / cfg.approach_fine_std))
        )

        # -- 2. Grasp bonus (one-time, widened sigma) ------------------------
        first_grasp = newly_grasped & ~self._has_been_grasped
        grasp_quality = torch.exp(-ee_to_piece_dist / cfg.grasp_quality_sigma)
        grasp_reward = first_grasp.float() * cfg.grasp_bonus * grasp_quality
        self._has_been_grasped = self._has_been_grasped | first_grasp

        # Update milestone curriculum: count total grasps across all envs
        n_new_grasps = first_grasp.sum().item()
        if n_new_grasps > 0:
            self._total_grasp_count += int(n_new_grasps)

        # -- 3. Lift reward (binary per-step, gated on grasp) ----------------
        piece_z = self._target_piece_pos[:, 2]
        is_lifted = (
            (piece_z > self._target_piece_initial_z + cfg.lift_threshold)
            & self._is_grasped
        )
        lift_reward = cfg.lift_weight * is_lifted.float()

        # -- 4. Transport reward (tanh kernel, gated on lift) ----------------
        piece_to_target = torch.norm(
            self._target_piece_pos - self._target_square_pos, dim=-1
        )
        transport_reward = is_lifted.float() * (
            cfg.transport_weight
            * (1.0 - torch.tanh(piece_to_target / cfg.transport_std))
            + cfg.transport_fine_weight
            * (1.0 - torch.tanh(piece_to_target / cfg.transport_fine_std))
        )

        # -- 5. Success bonus ------------------------------------------------
        success = piece_to_target < cfg.placement_tolerance
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
            + grasp_reward
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
