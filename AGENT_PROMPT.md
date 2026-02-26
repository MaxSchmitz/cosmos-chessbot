# Cosmos Chessbot -- Overnight Autonomous Loop

You are running in an autonomous loop. Each session is one iteration. You have MCP tools for camera, perception, chess engine, robot arm, and reasoning. You also have SSH access to the GPU server (`ssh ubuntu@isaacsim`, L40S GPU).

## Current state (updated after iteration 2)

### What's working
- **Board calibration**: YOLO corners detected. Board corners in pixels are known.
- **Robot-to-world transform**: Calibrated from 4 data points. Constants in waypoint_policy.py:
  - ROBOT_WORLD_X = 0.2198, ROBOT_WORLD_Y = 0.1799, ROBOT_HEADING_DEG = -101.2
  - ROBOT_Z_OFFSET = -0.04 (revised -- see below)
  - Calibration RMS = 19.7mm (roughly half a square)
- **URDF FK and numerical IK**: Working. `fk_urdf()`, `solve_ik_numerical()` in waypoint_policy.py.
- **Cosmos-Reason2 server**: Running on brev (port 8000). Both perception and reasoning models loaded.
  - Access via SSH tunnel: `ssh -f -N -L 8000:localhost:8000 ubuntu@isaacsim`
  - .mcp.json CHESS_COSMOS_URL = "http://localhost:8000" (needs tunnel running)
  - Tested successfully: `/reason/action` endpoint returns valid grasp strategies.
- **Camera reconnect tool**: Added `reconnect_cameras` MCP tool in tools_camera.py. Camera class has `reconnect()` method. Will be available after next MCP server restart.
- **Home position**: pan=7.8, lift=-98, elbow=99.4, wrist=63, roll=0.

### Critical hardware issues

**1. Elbow servo stuck at ~100 degrees**
- The elbow servo reads 99.6 and CANNOT increase past this value, regardless of what angle is commanded (tested 104, 110).
- It CAN decrease (commanded 90, read 94.4).
- This limits how far down the gripper can reach. With elbow stuck at 100, the minimum FK_Z is about -0.035 in robot frame.
- This is likely a mechanical or servo limit, not a software issue.

**2. Cameras failed (USB issue)**
- Both overhead and wrist cameras stopped returning frames mid-session.
- cv2.VideoCapture.read() returns ret=False.
- This happened after extended use. Likely USB bandwidth or stale file handles.
- The new `reconnect_cameras` tool should fix this on MCP restart.

**3. Revised Z offset: -0.04 (not -0.07)**
- Previous estimate of -0.07 was too low.
- Evidence: at home position (FK_Z=-0.019), the arm knocked over a piece near G1. If board were at Z=-0.07, the gripper would be 5cm above the board (impossible to knock a 3cm pawn).
- Revised estimate: board surface at robot Z ≈ -0.04. This means:
  - Pawn top (3cm tall): robot Z ≈ -0.01
  - Pawn midpoint (1.5cm): robot Z ≈ -0.025
  - At approach position (lift=-92, elbow=100, wrist=70): FK_Z=-0.028, which is AT pawn midpoint.

### Grasp attempt results
- Positioned at pan=18, lift=-92, elbow=100, wrist=70 (FK position: 0.119, -0.026, -0.028).
- Gripper opened to 55 degrees, then closed to 2 degrees.
- Gripper reading: 2.25 degrees -- no piece caught.
- Possible causes: XY calibration error (19.7mm RMS = almost half a square), or Z still not right.

## Safety rules

- ALWAYS park the robot before capturing overhead images.
- Use incremental joint movements (max 10 degrees per step for pan/lift/elbow, max 15 for wrist_flex).
- Before any trajectory: check overhead camera to confirm gripper position.
- Never command pan outside [-10, 70] degrees.
- Gripper: 2=closed, 60=open.
- A piece near G1 was knocked over previously -- board is not in pristine starting position.

## Objectives (in priority order)

### 1. Recover cameras and improve calibration (FIRST)

The camera reconnect tool has been added. On MCP server restart it should be available.

**Steps:**
1. Try `reconnect_cameras` tool first thing.
2. If cameras work: park robot, capture overhead, verify board is visible.
3. **Recalibrate with better method**: Instead of estimating gripper pixel position by eye, use a more precise approach:
   - Move gripper to a known position above the board.
   - Use wrist camera to see which square the gripper is directly above (the wrist cam looks down from the gripper).
   - Use overhead camera to get the gripper's pixel position precisely.
   - This gives a much more accurate (joint_angles, world_position) pair.
4. Collect 4-6 calibration points with this improved method.
5. Run `calibrate_robot_pose()` -- target RMS < 10mm.

### 2. Pick up a chess piece

With better calibration and working cameras:

**Strategy:**
1. Park robot, detect board state to find exact piece positions via YOLO.
2. Use `square_to_world()` for the target pawn.
3. Transform to robot frame with `world_to_robot_frame()`.
4. **Critical: elbow is stuck at 100.** Use joint configs that work within this constraint:
   - At pan=18, lift=-92, elbow=100, wrist=70: FK=(0.119, -0.026, -0.028). This is near H2 at ~pawn height.
   - Descending further requires increasing lift toward -80, but the arm retracts horizontally.
   - Try different pan angles to find the square where reach and Z align.
5. Open gripper to 55, position above piece, close to 2, check reading.
6. If gripper > 4 degrees: piece is grasped. Lift and transport.

**Backup approach if IK doesn't converge due to elbow limit:**
- Use direct `send_joint_angles` with manually computed configurations from the FK sweep tables.
- The FK tables in this prompt show exactly what Z is reachable at each joint configuration.

### 3. Set up SSH tunnel for Cosmos

The Cosmos server is running on brev but port 8000 is not exposed externally. Need an SSH tunnel:
```bash
ssh -f -N -L 8000:localhost:8000 ubuntu@isaacsim
```
This must be re-established each session. Once running, CHESS_COSMOS_URL=http://localhost:8000 works.

After tunnel is up, test with the MCP reasoning tools (reason_action, plan_trajectory).

### 4. Full game loop

If pick-and-place works:
1. detect_board_state -> FEN
2. get_best_move(fen) -> UCI move
3. Plan approach with Cosmos reason_action (grasp strategy, risks)
4. chess_move_waypoints(from, to) -> waypoints (or manual FK-based approach)
5. execute_trajectory or manual send_joint_angles
6. verify with detect_board_state

### 5. Pi0.5 (low priority)

- PI0Policy available in lerobot 0.3.2 but pi0.5 (pi05_base) needs lerobot >= 0.4.x.
- Upgrading could break existing code. Only attempt if main objectives are complete.
- There's a SO-100 wrist camera model: `salhotra/lerobot_pi0_so100wrist_test`.

## Reference: FK sweep data (elbow stuck at 100)

At pan=18, elbow=100 (actual ~99.6):

| lift  | wrist | FK_X   | FK_Y    | FK_Z    | above_board (Z=-0.04) |
|-------|-------|--------|---------|---------|----------------------|
| -98   | 63    | 0.1436 | -0.0340 | -0.0185 | 21.5mm               |
| -92   | 70    | 0.1117 | -0.0237 | -0.0303 | 9.7mm                |
| -88   | 72    | 0.0976 | -0.0191 | -0.0327 | 7.3mm                |
| -85   | 73    | 0.0875 | -0.0158 | -0.0342 | 5.8mm                |
| -82   | 74    | 0.0774 | -0.0125 | -0.0348 | 5.2mm                |

Note: FK_X decreases (arm retracts) as lift increases. H2 target X=0.112.
Best match for H2: lift=-92, wrist=70 (FK_X=0.112, 9.7mm above board).
