"""Waypoint-driven policy using Cosmos-Reason2 trajectory + geometric IK.

Converts Cosmos-Reason2 pixel waypoints to joint commands via geometric
inverse kinematics for the SO-101 arm.  This makes Cosmos-Reason2 the
primary driver of robot manipulation -- every movement is explained by
its reasoning output.

Link lengths derived from the SO-101 URDF:
  /Users/max/Code/isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf
"""

import logging
import time
from enum import Enum
from typing import Optional

import numpy as np

from .base_policy import BasePolicy, PolicyAction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SO-101 geometry (from URDF joint origins)
# ---------------------------------------------------------------------------

# Base to shoulder pivot height (Z component of shoulder_pan joint origin)
DEFAULT_BASE_HEIGHT: float = 0.0624

# Upper arm: shoulder to elbow (magnitude of elbow_flex joint origin)
DEFAULT_L2: float = 0.1126

# Forearm: elbow to wrist (magnitude of wrist_flex joint origin)
DEFAULT_L3: float = 0.1349

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
GRIPPER_OPEN: float = 4.5   # degrees -- fully open
GRIPPER_CLOSED: float = 0.5  # degrees -- grasping a piece


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


def classify_waypoint_phase(label: str) -> WaypointPhase:
    """Classify a Cosmos waypoint label into a manipulation phase.

    Cosmos-Reason2 outputs labels like "above e2", "grasp e2", "lift",
    "above e4", "place e4". We map these to gripper open/close decisions.
    """
    label_lower = label.lower()

    if "grasp" in label_lower or "grip" in label_lower or "pick" in label_lower:
        return WaypointPhase.GRASP
    if "place" in label_lower or "release" in label_lower or "drop" in label_lower:
        return WaypointPhase.PLACE
    if "lift" in label_lower or "raise" in label_lower:
        return WaypointPhase.LIFT
    if "lower" in label_lower and ("target" in label_lower or "place" in label_lower):
        return WaypointPhase.LOWER_PLACE
    if "lower" in label_lower:
        return WaypointPhase.LOWER_GRASP
    if "above" in label_lower or "approach" in label_lower or "hover" in label_lower:
        # Distinguish approach vs transport by checking for source vs target keywords
        # In practice the first "above" is approach, subsequent ones are transport
        return WaypointPhase.APPROACH
    if "transport" in label_lower or "move" in label_lower or "travel" in label_lower:
        return WaypointPhase.TRANSPORT
    if "retreat" in label_lower or "retract" in label_lower:
        return WaypointPhase.RETREAT

    # Default: transport (safest -- keep current gripper state)
    return WaypointPhase.TRANSPORT


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


def forward_kinematics(
    joints_deg: np.ndarray,
    base_height: float = DEFAULT_BASE_HEIGHT,
    L2: float = DEFAULT_L2,
    L3: float = DEFAULT_L3,
    L4: float = DEFAULT_L4,
) -> np.ndarray:
    """Compute gripper tip position from joint angles (for verification).

    Args:
        joints_deg: (5,) joint angles in degrees [pan, lift, elbow, wrist, roll].
        base_height: Height from base to shoulder pivot.
        L2, L3, L4: Link lengths.

    Returns:
        (3,) gripper tip position [x, y, z] in base frame.
    """
    q = np.radians(joints_deg)
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

    # Arm plane (2D)
    r = L2 * np.cos(q1) + L3 * np.cos(q1 + q2) + L4 * np.cos(q1 + q2 + q3)
    z = base_height + L2 * np.sin(q1) + L3 * np.sin(q1 + q2) + L4 * np.sin(q1 + q2 + q3)

    # Rotate into 3D via shoulder pan
    x = r * np.cos(q0)
    y = r * np.sin(q0)

    return np.array([x, y, z], dtype=np.float64)


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
        base_height: float = DEFAULT_BASE_HEIGHT,
        L2: float = DEFAULT_L2,
        L3: float = DEFAULT_L3,
        L4: float = DEFAULT_L4,
        joint_limits: Optional[dict] = None,
        interp_steps: int = 20,
        step_period: float = 0.05,
        grasp_dwell: float = 0.5,
    ):
        """Initialize waypoint policy.

        Args:
            base_height: Height from base to shoulder pivot (m).
            L2: Upper arm length (m).
            L3: Forearm length (m).
            L4: Wrist-to-gripper length (m).
            joint_limits: Per-joint (min, max) in degrees.
            interp_steps: Number of interpolation steps between waypoints.
            step_period: Time between interpolation steps (seconds).
            grasp_dwell: Extra dwell time at grasp/place waypoints (seconds).
        """
        self.base_height = base_height
        self.L2 = L2
        self.L3 = L3
        self.L4 = L4
        self.joint_limits = joint_limits or DEFAULT_JOINT_LIMITS
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

        for i, (xyz, label) in enumerate(zip(waypoints_3d, labels)):
            phase = classify_waypoint_phase(label)
            gripper_deg, self._is_grasped = gripper_for_phase(
                phase, self._is_grasped,
            )

            logger.info(
                "  WP %d/%d  %s  phase=%s  pos=(%.3f, %.3f, %.3f)  gripper=%.1f",
                i + 1, len(waypoints_3d), label, phase.value,
                xyz[0], xyz[1], xyz[2], gripper_deg,
            )

            # Solve IK
            target = np.array(xyz, dtype=np.float64)
            arm_joints = solve_ik(
                target,
                base_height=self.base_height,
                L2=self.L2,
                L3=self.L3,
                L4=self.L4,
                joint_limits=self.joint_limits,
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
