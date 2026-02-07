"""Chess board scene configuration for IsaacLab / LeIsaac.

Extends LeIsaac's SingleArmTaskSceneCfg (which provides SO-101 robot,
ee_frame, cameras, and lighting) with our chess board, table, and
piece placement geometry.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from leisaac.tasks.template.single_arm_env_cfg import SingleArmTaskSceneCfg
from leisaac.utils.env_utils import delete_attribute

from .fen_placement import BOARD_SQUARE_SIZE

# --------------------------------------------------------------------------- #
# Geometry constants
# --------------------------------------------------------------------------- #

# Table surface height (board sits on top)
TABLE_HEIGHT: float = 0.75

# RL board square size: scaled down to fit SO-101 workspace (~35cm reach).
# Original tournament board: 0.107m/square = 85cm total (too large for SO-101).
# Scaled: 0.035m/square = 28cm total board. All 64 squares are reachable.
RL_SQUARE_SIZE: float = 0.035

# Board center offset: shifted forward so robot can reach all squares.
# Robot shoulder is ~10cm above base; arm reaches ~33cm horizontally.
# Board center 15cm in front of robot → farthest rank at 15+14=29cm (within reach).
BOARD_CENTER_OFFSET_Y: float = 0.15

# Board center is at origin XY + offset, on the table surface
BOARD_CENTER = (0.0, BOARD_CENTER_OFFSET_Y, TABLE_HEIGHT)

# Robot base position: at origin XY, on the table surface.
ROBOT_POS = (0.0, 0.0, TABLE_HEIGHT)

# Overhead camera: looking straight down at the board
CAMERA_POS = (0.0, BOARD_CENTER_OFFSET_Y, TABLE_HEIGHT + 0.6)

# --------------------------------------------------------------------------- #
# Piece physics constants
# --------------------------------------------------------------------------- #
PIECE_MASS_KG: float = 0.010        # 10g (typical chess piece)
PIECE_FRICTION: float = 0.8
PIECE_RESTITUTION: float = 0.1      # low bounce
PIECE_LINEAR_DAMPING: float = 5.0   # prevent sliding
PIECE_ANGULAR_DAMPING: float = 5.0


# --------------------------------------------------------------------------- #
# Scene configuration
# --------------------------------------------------------------------------- #

@configclass
class ChessSceneCfg(SingleArmTaskSceneCfg):
    """Scene with a chess board, SO-101 robot arm, and overhead camera.

    Inherits from SingleArmTaskSceneCfg which provides:
    - robot (SO-101 follower arm with articulation config)
    - ee_frame (end-effector frame transformer)
    - wrist camera (attached to gripper)
    - front camera (attached to base)
    - dome light
    """

    # -- "scene" asset: table surface (required by parent) -------------------
    # SingleArmTaskSceneCfg expects a `scene` attribute with the main env USD.
    # We use a simple table cuboid since our board is loaded separately.
    scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 1.2, TABLE_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.3, 0.2),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, TABLE_HEIGHT / 2.0),
        ),
    )

    # -- Override front camera for overhead chess view -----------------------
    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=CAMERA_POS,
            rot=(0.707107, -0.707107, 0.0, 0.0),  # look straight down
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    # -- Brighter light for chess board visibility ---------------------------
    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=2000.0,
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        # Position the robot behind the board
        self.robot.init_state.pos = ROBOT_POS
        # Remove wrist camera (not needed for RL; saves compute)
        delete_attribute(self, "wrist")
