"""Waypoint-driven policy using Cosmos-Reason2 trajectory + empirical IK.

Converts Cosmos-Reason2 pixel waypoints to joint commands for the SO-101
arm.  Pan angle is computed analytically from the coordinate transform;
lift/elbow/wrist are interpolated from empirical calibration data.

The geometric 2-link IK model is kept for reference but NOT used for
real robot control -- its angle convention doesn't match lerobot's
calibrated degrees via simple constant offsets.
"""

import logging
import time
from enum import Enum
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from .base_policy import BasePolicy, PolicyAction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SO-101 geometry (from URDF joint origins)
# ---------------------------------------------------------------------------

# Base to shoulder pivot height (Z component of shoulder_pan joint origin)
DEFAULT_BASE_HEIGHT: float = 0.0624

# Upper arm: shoulder to elbow (effective, empirically calibrated)
# URDF value: 0.1126. Effective value absorbs calibration drift.
DEFAULT_L2: float = 0.0657

# Forearm: elbow to wrist (effective, empirically calibrated)
# URDF value: 0.1349. Effective value absorbs calibration drift.
DEFAULT_L3: float = 0.2166

# Wrist to gripper tip (magnitude of gripper_frame joint origin)
DEFAULT_L4: float = 0.098

# Joint limits in degrees (geometric-IK coordinate system).
# These are derived from the URDF limits but expressed in the planar-arm
# convention used by solve_ik():
#   q0 = shoulder_pan   (atan2 yaw, same as URDF)
#   q1 = shoulder_lift   (pitch from horizontal; URDF: +/-100 deg)
#   q2 = elbow_flex      (fold angle 0..pi; URDF: +/-97 deg)
#   q3 = wrist_flex      (computed to keep gripper vertical)
#   q4 = wrist_roll      (fixed at 0)
DEFAULT_JOINT_LIMITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (0.0, 150.0),
    "wrist_flex": (-180.0, 180.0),
    "wrist_roll": (-157.0, 163.0),
    "gripper": (-10.0, 100.0),
}

# Gripper positions
GRIPPER_OPEN: float = 60.0   # degrees -- fully open
GRIPPER_CLOSED: float = 2.0  # degrees -- grasping a piece


# ---------------------------------------------------------------------------
# URDF-based forward kinematics (SO-ARM100 / SO-101)
# ---------------------------------------------------------------------------
# Joint transforms extracted from TheRobotStudio/SO-ARM100 URDF.
# Each transform is the fixed origin+rpy of a joint in its parent frame.
# Joint angles are applied as Rz(q) after each fixed transform.
# Input: lerobot calibrated degrees (0 = calibration pose).

def _Rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def _Ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def _Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def _joint_transform(xyz: list[float], rpy: list[float]) -> np.ndarray:
    """Build a 4x4 homogeneous transform from xyz translation and rpy rotation."""
    m = _Rz(rpy[2]) @ _Ry(rpy[1]) @ _Rx(rpy[0])
    m[:3, 3] = xyz
    return m


# Fixed joint transforms from URDF (base_link -> gripper_frame)
_J_PAN = _joint_transform([0.0388353, 0, 0.0624], [3.14159, 0, -3.14159])
_J_LIFT = _joint_transform([-0.0303992, -0.0182778, -0.0542], [-1.5708, -1.5708, 0])
_J_ELBOW = _joint_transform([-0.11257, -0.028, 0], [0, 0, 1.5708])
_J_WRIST = _joint_transform([-0.1349, 0.0052, 0], [0, 0, -1.5708])
_J_ROLL = _joint_transform([0, -0.0611, 0.0181], [1.5708, 0.0487, 3.14159])
_J_GRIP = _joint_transform([-0.0079, -0.000218, -0.0981], [0, 3.14159, 0])

# Joint angle limits for numerical IK (lerobot calibrated degrees).
# Tighter than hardware limits to stay in well-behaved configurations.
IK_JOINT_BOUNDS = {
    "shoulder_pan": (-90.0, 90.0),
    "shoulder_lift": (-100.0, 30.0),
    "elbow_flex": (-30.0, 110.0),
    "wrist_flex": (-20.0, 90.0),
    "wrist_roll": (-90.0, 90.0),
}


def fk_urdf(
    pan_deg: float,
    lift_deg: float,
    elbow_deg: float,
    wrist_deg: float,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """URDF-based forward kinematics for SO-101.

    Computes the gripper tip position in the base_link frame from
    lerobot calibrated joint angles (degrees).

    Args:
        pan_deg: Shoulder pan angle.
        lift_deg: Shoulder lift angle.
        elbow_deg: Elbow flex angle.
        wrist_deg: Wrist flex angle.
        roll_deg: Wrist roll angle.

    Returns:
        (3,) gripper tip position [x, y, z] in base_link frame (meters).
    """
    q = np.radians([pan_deg, lift_deg, elbow_deg, wrist_deg, roll_deg])
    M = np.eye(4)
    M = M @ _J_PAN @ _Rz(q[0])
    M = M @ _J_LIFT @ _Rz(q[1])
    M = M @ _J_ELBOW @ _Rz(q[2])
    M = M @ _J_WRIST @ _Rz(q[3])
    M = M @ _J_ROLL @ _Rz(q[4])
    M = M @ _J_GRIP
    return M[:3, 3].copy()


def fk_urdf_full(
    pan_deg: float,
    lift_deg: float,
    elbow_deg: float,
    wrist_deg: float,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """URDF FK returning full 4x4 transform (position + orientation)."""
    q = np.radians([pan_deg, lift_deg, elbow_deg, wrist_deg, roll_deg])
    M = np.eye(4)
    M = M @ _J_PAN @ _Rz(q[0])
    M = M @ _J_LIFT @ _Rz(q[1])
    M = M @ _J_ELBOW @ _Rz(q[2])
    M = M @ _J_WRIST @ _Rz(q[3])
    M = M @ _J_ROLL @ _Rz(q[4])
    M = M @ _J_GRIP
    return M.copy()


def solve_ik_numerical(
    target_xyz: np.ndarray,
    initial_guess: Optional[np.ndarray] = None,
    roll_deg: float = 0.0,
    bounds: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """Numerical IK using URDF FK and scipy optimization.

    Finds joint angles [pan, lift, elbow, wrist_flex] that place the
    gripper tip at the target position, with wrist_roll fixed.

    Args:
        target_xyz: (3,) target position in base_link frame [x, y, z].
        initial_guess: (4,) initial joint angles [pan, lift, elbow, wrist]
            in degrees. If None, uses a reasonable default.
        roll_deg: Fixed wrist roll angle (degrees).
        bounds: Joint angle bounds dict. If None, uses IK_JOINT_BOUNDS.

    Returns:
        (5,) joint angles [pan, lift, elbow, wrist, roll] in lerobot
        degrees, or None if no solution found within tolerance.
    """
    if bounds is None:
        bounds = IK_JOINT_BOUNDS

    if initial_guess is None:
        # Start from a "reaching forward" configuration
        initial_guess = np.array([10.0, -80.0, 80.0, 50.0])

    target = np.asarray(target_xyz, dtype=np.float64)

    def objective(q4):
        pos = fk_urdf(q4[0], q4[1], q4[2], q4[3], roll_deg)
        return float(np.sum((pos - target) ** 2))

    scipy_bounds = [
        bounds["shoulder_pan"],
        bounds["shoulder_lift"],
        bounds["elbow_flex"],
        bounds["wrist_flex"],
    ]

    result = minimize(
        objective,
        initial_guess,
        method="L-BFGS-B",
        bounds=scipy_bounds,
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
    )

    pos_error = np.linalg.norm(fk_urdf(result.x[0], result.x[1], result.x[2], result.x[3], roll_deg) - target)

    if pos_error > 0.005:  # 5mm tolerance
        logger.warning(
            "IK solution inaccurate: error=%.4fm for target %s (result: %s)",
            pos_error, target, result.x,
        )
        # Try multiple initial guesses
        best_result = result
        best_error = pos_error
        for pan_init in [-20, 0, 20, 40, 60]:
            for lift_init in [-90, -60, -30]:
                guess = np.array([pan_init, lift_init, 80.0, 50.0])
                r = minimize(
                    objective, guess, method="L-BFGS-B",
                    bounds=scipy_bounds,
                    options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
                )
                err = np.linalg.norm(
                    fk_urdf(r.x[0], r.x[1], r.x[2], r.x[3], roll_deg) - target
                )
                if err < best_error:
                    best_error = err
                    best_result = r
        if best_error > 0.005:
            logger.error(
                "IK failed: best error=%.4fm for target %s", best_error, target,
            )
            return None
        result = best_result
        pos_error = best_error

    joints = np.array([result.x[0], result.x[1], result.x[2], result.x[3], roll_deg])
    logger.debug(
        "IK solved: target=%s -> joints=%s (error=%.4fm)",
        target, joints, pos_error,
    )
    return joints


# ---------------------------------------------------------------------------
# Robot pose calibration (empirically determined)
# ---------------------------------------------------------------------------

# Robot base position in world frame (meters).
# World frame: board center at (0, 0.20, 0), X = A->H, Y = rank 1->8.
# Empirically determined via scipy optimization on 4 data points
# (H1, G1, F1, E1) with corrected square assignments.
ROBOT_WORLD_X: float = 0.2198
ROBOT_WORLD_Y: float = 0.1799

# Robot heading: angle from world +X axis to robot forward direction (degrees).
ROBOT_HEADING_DEG: float = -101.2

# Z offset: world Z=0 is board surface, robot Z=0 is base_link.
# Board surface is below the base_link origin by this amount.
# Revised estimate based on piece-knock event: at home position
# (lift=-98, elbow=100, wrist=50-63), FK_Z ranges -0.001 to -0.019.
# The gripper knocked a pawn (3cm tall) near G1 during wrist sweep,
# so board_surface is between -0.031 and -0.049. Using -0.04.
ROBOT_Z_OFFSET: float = -0.04

# ---------------------------------------------------------------------------
# Empirical joint calibration data
# ---------------------------------------------------------------------------
# Measured lerobot joint angles at known board squares (z=0, board surface).
# Pan angle is the IK-computed pan for each square; lift/elbow/wrist are
# the physically measured lerobot angles at that position.
#
# Data collected with corrected square assignments (verified via overhead
# camera with visible file labels in good lighting).

CALIB_PAN = np.array([7.5, 20.0, 35.0, 50.0])      # H1, G1, F1, E1
CALIB_LIFT = np.array([-98.0, -93.0, -77.0, -50.0])
CALIB_ELBOW = np.array([100.0, 85.0, 57.0, 24.0])
CALIB_WRIST = np.array([63.0, 63.0, 75.0, 83.0])

# Wrist constraint: lift + elbow + wrist ≈ 55 (empirical, for z=0)
WRIST_CONSTRAINT_SUM: float = 55.0

# Build interpolators (linear, with extrapolation for edge cases)
_interp_lift = interp1d(CALIB_PAN, CALIB_LIFT, kind='linear', fill_value='extrapolate')
_interp_elbow = interp1d(CALIB_PAN, CALIB_ELBOW, kind='linear', fill_value='extrapolate')
_interp_wrist = interp1d(CALIB_PAN, CALIB_WRIST, kind='linear', fill_value='extrapolate')


# ---------------------------------------------------------------------------
# Waypoint phase classification
# ---------------------------------------------------------------------------

class WaypointPhase(Enum):
    """Phases of a pick-and-place trajectory."""
    APPROACH = "approach"
    LOWER_GRASP = "lower_grasp"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSPORT = "transport"
    LOWER_PLACE = "lower_place"
    PLACE = "place"
    RETREAT = "retreat"


def classify_waypoint_phases(labels: list[str]) -> list[WaypointPhase]:
    """Classify a sequence of Cosmos waypoint labels into manipulation phases.

    Uses a sequence-based approach rather than per-label keywords because
    Cosmos labels are verbose and keywords overlap heavily (e.g. every label
    mentions "gripper", approach labels say "prepare for grasping", etc.).

    Standard pick-and-place sequence for N waypoints:
      1. APPROACH      -- position above source (gripper open)
      2. LOWER_GRASP   -- lower to piece (gripper open, then close)
      3. LIFT          -- lift piece (gripper closed)
      4. TRANSPORT     -- move above target (gripper closed)
      5. LOWER_PLACE   -- lower to target (gripper closed, then open)

    For longer sequences we assign the middle waypoints as TRANSPORT.
    """
    n = len(labels)
    if n == 0:
        return []

    # Fixed mapping for standard 5-waypoint trajectory
    if n <= 2:
        # Too few waypoints -- just approach then place
        return [WaypointPhase.APPROACH] + [WaypointPhase.PLACE] * (n - 1)

    if n == 3:
        return [WaypointPhase.APPROACH, WaypointPhase.GRASP, WaypointPhase.PLACE]

    if n == 4:
        return [
            WaypointPhase.APPROACH,
            WaypointPhase.GRASP,
            WaypointPhase.LIFT,
            WaypointPhase.PLACE,
        ]

    # n >= 5: first 2 are pick, last 2 are place, middle are transport
    phases = []
    phases.append(WaypointPhase.APPROACH)       # WP 1: above source
    phases.append(WaypointPhase.GRASP)           # WP 2: lower + grasp
    for _ in range(n - 4):
        phases.append(WaypointPhase.LIFT)        # WP 3..n-2: lift/transport
    phases.append(WaypointPhase.TRANSPORT)       # WP n-1: above target
    phases.append(WaypointPhase.PLACE)           # WP n: lower + place

    return phases


def gripper_for_phase(phase: WaypointPhase, is_grasped: bool) -> tuple[float, bool]:
    """Return (gripper_deg, new_is_grasped) for a given phase.

    Args:
        phase: Current waypoint phase.
        is_grasped: Whether we currently hold a piece.

    Returns:
        (gripper_degrees, is_grasped_after)
    """
    if phase in (WaypointPhase.GRASP,):
        return GRIPPER_CLOSED, True
    if phase in (WaypointPhase.PLACE,):
        return GRIPPER_OPEN, False
    if phase in (WaypointPhase.APPROACH, WaypointPhase.LOWER_GRASP):
        return GRIPPER_OPEN, False
    # Maintain current state for lift, transport, lower_place, retreat
    return (GRIPPER_CLOSED if is_grasped else GRIPPER_OPEN), is_grasped


# ---------------------------------------------------------------------------
# Geometric IK for SO-101
# ---------------------------------------------------------------------------

def solve_ik(
    target_xyz: np.ndarray,
    base_height: float = DEFAULT_BASE_HEIGHT,
    L2: float = DEFAULT_L2,
    L3: float = DEFAULT_L3,
    L4: float = DEFAULT_L4,
    joint_limits: dict = DEFAULT_JOINT_LIMITS,
) -> Optional[np.ndarray]:
    """Solve inverse kinematics for SO-101 to reach a target XYZ position.

    The SO-101 is modelled as:
      - q0: shoulder_pan (yaw around Z)
      - q1: shoulder_lift (pitch in arm plane)
      - q2: elbow_flex (pitch in arm plane)
      - q3: wrist_flex (pitch -- keeps gripper vertical)
      - q4: wrist_roll (fixed at 0)

    The gripper should point straight down for pick/place, so we constrain
    q3 = -(q1 + q2) - pi/2 to keep the gripper vertical.

    Args:
        target_xyz: (3,) target position in robot base frame [x, y, z].
        base_height: Height from base to shoulder pivot.
        L2: Upper arm length (shoulder to elbow).
        L3: Forearm length (elbow to wrist).
        L4: Wrist to gripper tip length.
        joint_limits: Dict of (min_deg, max_deg) per joint.

    Returns:
        (5,) joint angles in degrees [pan, lift, elbow, wrist_flex, wrist_roll]
        or None if unreachable.
    """
    x, y, z = float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])

    # 1. Shoulder pan: point at target in XY plane
    q0 = np.arctan2(y, x)

    # 2. Project into arm plane
    r_xy = np.sqrt(x**2 + y**2)  # horizontal distance
    # Wrist point is L4 above the target (gripper points down)
    wrist_z = z + L4 - base_height
    wrist_r = r_xy

    # 3. Two-link IK for shoulder_lift + elbow_flex
    d_sq = wrist_r**2 + wrist_z**2
    d = np.sqrt(d_sq)

    if d > L2 + L3:
        logger.warning(
            "Target (%.3f, %.3f, %.3f) unreachable: d=%.3f > L2+L3=%.3f",
            x, y, z, d, L2 + L3,
        )
        # Scale wrist_r/wrist_z to max reach
        scale = (L2 + L3 - 0.001) / d
        wrist_r *= scale
        wrist_z *= scale
        d_sq = wrist_r**2 + wrist_z**2
        d = np.sqrt(d_sq)

    if d < abs(L2 - L3):
        logger.warning(
            "Target too close: d=%.3f < |L2-L3|=%.3f",
            d, abs(L2 - L3),
        )
        scale = (abs(L2 - L3) + 0.001) / d
        wrist_r *= scale
        wrist_z *= scale
        d_sq = wrist_r**2 + wrist_z**2

    # Elbow angle via law of cosines (positive = folded)
    cos_q2 = (d_sq - L2**2 - L3**2) / (2.0 * L2 * L3)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    q2 = np.arccos(cos_q2)  # elbow-up solution (positive q2)

    # Shoulder angle
    q1 = np.arctan2(wrist_z, wrist_r) - np.arctan2(
        L3 * np.sin(q2), L2 + L3 * np.cos(q2),
    )

    # 4. Wrist flex: keep gripper vertical (pointing straight down)
    q3 = -(q1 + q2) - np.pi / 2.0

    # 5. Wrist roll: fixed
    q4 = 0.0

    # Convert to degrees
    joints_deg = np.degrees(np.array([q0, q1, q2, q3, q4]))

    # Clamp to joint limits
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    for i, name in enumerate(names):
        lo, hi = joint_limits[name]
        if joints_deg[i] < lo or joints_deg[i] > hi:
            logger.debug(
                "Joint %s = %.1f deg outside [%.1f, %.1f], clamping",
                name, joints_deg[i], lo, hi,
            )
            joints_deg[i] = np.clip(joints_deg[i], lo, hi)

    return joints_deg


def world_to_robot_frame(
    target_xyz: np.ndarray,
    robot_x: float = ROBOT_WORLD_X,
    robot_y: float = ROBOT_WORLD_Y,
    heading_deg: float = ROBOT_HEADING_DEG,
    z_offset: float = ROBOT_Z_OFFSET,
) -> np.ndarray:
    """Transform a world-frame position to the robot's base frame.

    The robot base frame: +X forward (arm at pan=0), +Y left, +Z up.
    World Z=0 is board surface; robot Z=0 is base_link origin.
    The z_offset converts between them: robot_z = world_z + z_offset.
    """
    wx, wy, wz = float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])
    dx = wx - robot_x
    dy = wy - robot_y
    theta = np.radians(heading_deg)
    rx = dx * np.cos(theta) + dy * np.sin(theta)
    ry = -dx * np.sin(theta) + dy * np.cos(theta)
    rz = wz + z_offset
    return np.array([rx, ry, rz], dtype=np.float64)


def solve_ik_lerobot(
    target_world_xyz: np.ndarray,
    robot_x: float = ROBOT_WORLD_X,
    robot_y: float = ROBOT_WORLD_Y,
    heading_deg: float = ROBOT_HEADING_DEG,
    initial_guess: Optional[np.ndarray] = None,
    roll_deg: float = 0.0,
) -> Optional[np.ndarray]:
    """World coordinates -> lerobot joint angles via URDF-based numerical IK.

    Transforms the target from the board world frame to the robot base
    frame, then solves IK numerically using the URDF forward kinematics.

    Args:
        target_world_xyz: (3,) target in world frame [x, y, z].
        robot_x, robot_y: Robot base position in world frame (meters).
        heading_deg: Robot heading from world +X (degrees).
        initial_guess: (4,) initial [pan, lift, elbow, wrist] in degrees.
        roll_deg: Fixed wrist roll angle (degrees).

    Returns:
        (5,) lerobot joint angles [pan, lift, elbow, wrist, roll] in
        degrees, or None if unreachable.
    """
    robot_xyz = world_to_robot_frame(target_world_xyz, robot_x, robot_y, heading_deg)
    return solve_ik_numerical(
        robot_xyz,
        initial_guess=initial_guess,
        roll_deg=roll_deg,
    )


def forward_kinematics(
    joints_deg: np.ndarray,
    **_kwargs,
) -> np.ndarray:
    """Compute gripper tip position from joint angles.

    Uses URDF-based FK for accuracy. Accepts (5,) joint angles
    [pan, lift, elbow, wrist, roll] in lerobot calibrated degrees.

    Returns:
        (3,) gripper tip position [x, y, z] in base_link frame.
    """
    return fk_urdf(
        joints_deg[0], joints_deg[1], joints_deg[2],
        joints_deg[3], joints_deg[4] if len(joints_deg) > 4 else 0.0,
    )


# ---------------------------------------------------------------------------
# WaypointPolicy
# ---------------------------------------------------------------------------

class WaypointPolicy(BasePolicy):
    """Cosmos-Reason2 waypoint-driven manipulation policy.

    Converts Cosmos trajectory waypoints (pixel coordinates) to joint
    commands via geometric inverse kinematics.  Each waypoint is executed
    sequentially with linear interpolation in joint space.
    """

    def __init__(
        self,
        robot_x: float = ROBOT_WORLD_X,
        robot_y: float = ROBOT_WORLD_Y,
        heading_deg: float = ROBOT_HEADING_DEG,
        interp_steps: int = 20,
        step_period: float = 0.05,
        grasp_dwell: float = 0.5,
    ):
        """Initialize waypoint policy.

        Args:
            robot_x, robot_y: Robot base position in world frame (m).
            heading_deg: Robot heading from world +X (degrees).
            interp_steps: Number of interpolation steps between waypoints.
            step_period: Time between interpolation steps (seconds).
            grasp_dwell: Extra dwell time at grasp/place waypoints (seconds).
        """
        self.robot_x = robot_x
        self.robot_y = robot_y
        self.heading_deg = heading_deg
        self.interp_steps = interp_steps
        self.step_period = step_period
        self.grasp_dwell = grasp_dwell

        self._is_grasped = False

    # ----- BasePolicy interface -------------------------------------------

    def reset(self):
        self._is_grasped = False

    def select_action(self, images, robot_state, instruction=None):
        """Not used for waypoint policy (use run_waypoint_trajectory)."""
        return PolicyAction(
            actions=np.zeros(6),
            success_probability=0.0,
            metadata={"policy": "waypoint", "note": "use run_waypoint_trajectory"},
        )

    def plan_action(self, images, robot_state, instruction=None):
        return [self.select_action(images, robot_state, instruction)]

    # ----- Waypoint execution ---------------------------------------------

    def run_waypoint_trajectory(
        self,
        waypoints_3d: list[tuple[float, float, float]],
        labels: list[str],
        get_state_fn,
        send_action_fn,
    ) -> bool:
        """Execute a sequence of 3D waypoints on the real robot.

        For each waypoint:
          1. Classify the phase from the Cosmos label
          2. Solve IK for the target position
          3. Interpolate from current joints to target
          4. Send each interpolated step to the robot
          5. Dwell at grasp/place waypoints for gripper actuation

        Args:
            waypoints_3d: List of (x, y, z) world positions.
            labels: Cosmos waypoint labels (same length as waypoints_3d).
            get_state_fn: Returns (6,) joint angles in degrees.
            send_action_fn: Accepts (6,) joint targets in degrees.

        Returns:
            True if the full trajectory was executed.
        """
        self.reset()

        if len(waypoints_3d) == 0:
            logger.warning("No waypoints to execute")
            return False

        logger.info(
            "Executing waypoint trajectory: %d waypoints", len(waypoints_3d),
        )

        phases = classify_waypoint_phases(labels)

        for i, (xyz, label) in enumerate(zip(waypoints_3d, labels)):
            phase = phases[i]
            gripper_deg, self._is_grasped = gripper_for_phase(
                phase, self._is_grasped,
            )

            logger.info(
                "  WP %d/%d  %s  phase=%s  pos=(%.3f, %.3f, %.3f)  gripper=%.1f",
                i + 1, len(waypoints_3d), label, phase.value,
                xyz[0], xyz[1], xyz[2], gripper_deg,
            )

            # Solve IK (world coords -> lerobot joint angles)
            target = np.array(xyz, dtype=np.float64)
            arm_joints = solve_ik_lerobot(
                target,
                robot_x=self.robot_x,
                robot_y=self.robot_y,
                heading_deg=self.heading_deg,
            )

            if arm_joints is None:
                logger.error("IK failed for waypoint %d: %s", i, xyz)
                return False

            # Full 6-DOF target: 5 arm joints + gripper
            target_joints = np.concatenate([arm_joints, [gripper_deg]])

            # Get current state and interpolate
            current = get_state_fn()
            for step in range(self.interp_steps):
                alpha = (step + 1) / self.interp_steps
                interp = current + alpha * (target_joints - current)
                send_action_fn(interp)
                time.sleep(self.step_period)

            # Extra dwell at grasp/place for gripper actuation
            if phase in (WaypointPhase.GRASP, WaypointPhase.PLACE):
                logger.info("    Dwelling %.1fs for %s", self.grasp_dwell, phase.value)
                time.sleep(self.grasp_dwell)

        logger.info("Waypoint trajectory complete")
        return True


# ---------------------------------------------------------------------------
# Robot-to-world calibration
# ---------------------------------------------------------------------------

def calibrate_robot_pose(
    joint_angles_list: list[np.ndarray],
    world_positions: list[np.ndarray],
) -> tuple[float, float, float, float]:
    """Estimate robot base position and heading from observed data.

    Given pairs of (joint_angles, world_position), optimizes for the
    robot's position (robot_x, robot_y) and heading in the world frame
    by minimizing FK prediction error.

    Args:
        joint_angles_list: List of (5,) or (6,) joint angle arrays
            (lerobot degrees). Only first 5 are used.
        world_positions: List of (3,) world positions [x, y, z]
            corresponding to the gripper position at each joint config.

    Returns:
        (robot_x, robot_y, heading_deg, rms_error_mm) where rms_error_mm
        is the root-mean-square position error in millimeters.
    """
    # Compute FK positions in robot base frame
    fk_positions = []
    for joints in joint_angles_list:
        pos = fk_urdf(joints[0], joints[1], joints[2], joints[3],
                       joints[4] if len(joints) > 4 else 0.0)
        fk_positions.append(pos)

    world_pts = [np.asarray(w, dtype=np.float64) for w in world_positions]

    def cost(params):
        rx, ry, heading_rad = params
        total_sq = 0.0
        cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
        for fk_pos, world_pos in zip(fk_positions, world_pts):
            # FK position in robot frame -> world frame
            # world = robot_origin + R(heading) @ fk_robot
            wx = rx + cos_h * fk_pos[0] - sin_h * fk_pos[1]
            wy = ry + sin_h * fk_pos[0] + cos_h * fk_pos[1]
            wz = fk_pos[2]  # Z is shared (vertical)
            pred = np.array([wx, wy, wz])
            total_sq += np.sum((pred - world_pos) ** 2)
        return total_sq

    # Try multiple heading initializations
    best_result = None
    best_cost = float("inf")
    for heading_init in np.linspace(-np.pi, np.pi, 12, endpoint=False):
        r = minimize(
            cost,
            [0.0, -0.15, heading_init],
            method="L-BFGS-B",
            bounds=[(-0.5, 0.5), (-0.5, 0.5), (-np.pi, np.pi)],
        )
        if r.fun < best_cost:
            best_cost = r.fun
            best_result = r

    rx, ry, heading_rad = best_result.x
    heading_deg = float(np.degrees(heading_rad))
    n = len(joint_angles_list)
    rms_mm = float(np.sqrt(best_cost / n) * 1000)

    logger.info(
        "Robot pose calibration: x=%.4f y=%.4f heading=%.1f deg (RMS=%.1fmm, %d pts)",
        rx, ry, heading_deg, rms_mm, n,
    )
    return float(rx), float(ry), heading_deg, rms_mm


# ---------------------------------------------------------------------------
# Chess move waypoint generation
# ---------------------------------------------------------------------------

def chess_move_waypoints(
    from_sq: str,
    to_sq: str,
    approach_z: float = 0.05,
    grasp_z: float = 0.01,
    square_size: float = 0.05,
    center_y: float = 0.20,
    table_z: float = 0.0,
) -> tuple[list[dict], list[str]]:
    """Generate pick-and-place waypoints for a chess move.

    Produces a 5-waypoint trajectory:
      1. APPROACH  -- above source square (gripper open)
      2. GRASP     -- lower to piece (gripper closes)
      3. LIFT      -- lift piece above source
      4. TRANSPORT -- move above target square
      5. PLACE     -- lower to target (gripper opens)

    Args:
        from_sq: Source square in algebraic notation (e.g. 'e2').
        to_sq: Target square (e.g. 'e4').
        approach_z: Height above board for approach/lift/transport (meters).
        grasp_z: Height for grasp/place (meters, above table_z).
        square_size: Board square size (meters).
        center_y: Board center Y offset.
        table_z: Board surface height.

    Returns:
        (waypoints_3d, labels) ready for execute_trajectory.
        waypoints_3d is a list of {"x", "y", "z"} dicts.
    """
    from ..utils.pixel_to_board import square_to_world

    fx, fy, fz = square_to_world(from_sq, square_size, center_y, table_z)
    tx, ty, tz = square_to_world(to_sq, square_size, center_y, table_z)

    waypoints = [
        {"x": fx, "y": fy, "z": table_z + approach_z},
        {"x": fx, "y": fy, "z": table_z + grasp_z},
        {"x": fx, "y": fy, "z": table_z + approach_z},
        {"x": tx, "y": ty, "z": table_z + approach_z},
        {"x": tx, "y": ty, "z": table_z + grasp_z},
    ]
    labels = [
        f"approach above {from_sq}",
        f"lower to grasp piece on {from_sq}",
        f"lift piece from {from_sq}",
        f"transport piece above {to_sq}",
        f"lower to place piece on {to_sq}",
    ]
    return waypoints, labels
