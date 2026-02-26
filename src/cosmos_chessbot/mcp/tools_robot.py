"""Robot control tools for the SO-101 arm."""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from mcp.server.fastmcp import Context

from .server import mcp
from .state import ServerState


def _read_joints(state: ServerState) -> np.ndarray:
    """Read current joint positions as a numpy array."""
    obs = state.robot.get_observation()
    return np.array([
        float(obs[n].item() if hasattr(obs[n], "item") else obs[n])
        for n in state.joint_names
    ], dtype=np.float32)


def _send_joints(state: ServerState, targets: np.ndarray | list[float]):
    """Send joint angle targets to the robot."""
    action_dict = {}
    for i, name in enumerate(state.joint_names):
        if i < len(targets):
            action_dict[name] = torch.tensor([float(targets[i])], dtype=torch.float32)
    state.robot.send_action(action_dict)


@mcp.tool()
def get_robot_joints(ctx: Context) -> str:
    """Read current joint angles from the SO-101 robot arm.

    Returns 6 joint values in degrees: shoulder_pan, shoulder_lift,
    elbow_flex, wrist_flex, wrist_roll, gripper.
    """
    state: ServerState = ctx.request_context.lifespan_context
    if state.robot is None:
        return json.dumps({"error": "Robot not connected"})

    try:
        joints = _read_joints(state)
        return json.dumps({
            "joints": {name: float(joints[i]) for i, name in enumerate(state.joint_names)},
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def home_robot(ctx: Context) -> str:
    """Send the robot arm to its home (neutral) position.

    The home position points the arm forward over the board at a neutral height.
    """
    state: ServerState = ctx.request_context.lifespan_context
    if state.robot is None:
        return json.dumps({"error": "Robot not connected"})

    try:
        action_dict = {
            name: torch.tensor([val], dtype=torch.float32)
            for name, val in state.home_position.items()
        }
        state.robot.send_action(action_dict)
        return json.dumps({"success": True, "position": "home"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def park_robot(ctx: Context) -> str:
    """Park the arm out of the overhead camera's field of view.

    Rotates the shoulder to 90 degrees so the arm doesn't occlude the board.
    Waits 2 seconds for the arm to settle before returning.
    Call this before capture_overhead or detect_board_state.
    """
    state: ServerState = ctx.request_context.lifespan_context
    if state.robot is None:
        return json.dumps({"error": "Robot not connected"})

    try:
        action_dict = {
            name: torch.tensor([val], dtype=torch.float32)
            for name, val in state.park_position.items()
        }
        state.robot.send_action(action_dict)
        time.sleep(2.0)
        return json.dumps({"success": True, "position": "parked"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def execute_trajectory(
    ctx: Context,
    waypoints_3d: list[dict],
    labels: list[str],
) -> str:
    """Execute a 3D waypoint trajectory on the robot using geometric IK.

    Each waypoint is converted to joint angles via inverse kinematics,
    then interpolated smoothly from the current position. Gripper open/close
    is handled automatically based on the manipulation phase
    (approach -> grasp -> lift -> transport -> place).

    Get waypoints from plan_trajectory (which returns waypoints_3d when
    calibration is active).

    Args:
        waypoints_3d: List of {"x": float, "y": float, "z": float} in meters.
        labels: Cosmos waypoint labels, same length as waypoints_3d.
    """
    state: ServerState = ctx.request_context.lifespan_context
    if state.robot is None:
        return json.dumps({"error": "Robot not connected"})
    if state.waypoint_policy is None:
        return json.dumps({"error": "Waypoint policy not initialized"})

    try:
        xyz_list = [(wp["x"], wp["y"], wp["z"]) for wp in waypoints_3d]

        def get_state_fn():
            return _read_joints(state)

        def send_action_fn(targets):
            _send_joints(state, targets)

        success = state.waypoint_policy.run_waypoint_trajectory(
            waypoints_3d=xyz_list,
            labels=labels,
            get_state_fn=get_state_fn,
            send_action_fn=send_action_fn,
        )
        return json.dumps({"success": success, "waypoints_executed": len(xyz_list)})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def send_joint_angles(
    ctx: Context,
    shoulder_pan: float,
    shoulder_lift: float,
    elbow_flex: float,
    wrist_flex: float,
    wrist_roll: float,
    gripper: float,
) -> str:
    """Send raw joint angle targets to the robot in degrees.

    Low-level control -- prefer execute_trajectory for normal chess moves.

    Args:
        shoulder_pan: Shoulder pan angle in degrees.
        shoulder_lift: Shoulder lift angle in degrees.
        elbow_flex: Elbow flex angle in degrees.
        wrist_flex: Wrist flex angle in degrees.
        wrist_roll: Wrist roll angle in degrees.
        gripper: Gripper position in degrees (2=closed, 60=open).
    """
    state: ServerState = ctx.request_context.lifespan_context
    if state.robot is None:
        return json.dumps({"error": "Robot not connected"})

    try:
        targets = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        _send_joints(state, targets)
        return json.dumps({
            "success": True,
            "joints_sent": dict(zip(state.joint_names, targets)),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def move_to_position(
    ctx: Context,
    x: float,
    y: float,
    z: float,
    gripper: float = 60.0,
) -> str:
    """Move the robot end-effector to a 3D position using inverse kinematics.

    Solves IK for the target position and smoothly interpolates from the
    current joint angles (20 steps over ~1 second).

    Args:
        x: Target X in robot base frame (meters).
        y: Target Y in robot base frame (meters).
        z: Target Z in robot base frame (meters).
        gripper: Gripper position in degrees (60=open, 2=closed).
    """
    state: ServerState = ctx.request_context.lifespan_context
    if state.robot is None:
        return json.dumps({"error": "Robot not connected"})

    try:
        from ..policy.waypoint_policy import solve_ik_numerical, fk_urdf

        target = np.array([x, y, z], dtype=np.float64)

        # Use current joints as initial guess for IK
        current = _read_joints(state)
        initial_guess = current[:4].copy()

        arm_joints = solve_ik_numerical(target, initial_guess=initial_guess)
        if arm_joints is None:
            return json.dumps({"error": f"Position ({x}, {y}, {z}) unreachable"})

        # Verify FK matches target
        verify = fk_urdf(arm_joints[0], arm_joints[1], arm_joints[2], arm_joints[3], arm_joints[4])
        error_mm = float(np.linalg.norm(verify - target) * 1000)

        target_joints = np.concatenate([arm_joints, [gripper]])

        # Smooth interpolation: 20 steps at 50ms
        n_steps = 20
        for step in range(n_steps):
            alpha = (step + 1) / n_steps
            interp = current + alpha * (target_joints - current)
            _send_joints(state, interp)
            time.sleep(0.05)

        return json.dumps({
            "success": True,
            "target_xyz": [x, y, z],
            "joint_angles_deg": arm_joints.tolist(),
            "fk_error_mm": error_mm,
            "gripper": gripper,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})
