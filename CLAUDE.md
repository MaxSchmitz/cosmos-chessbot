# Cosmos Chessbot

Physical AI chess robot: SO-101 arm + YOLO-DINO perception + Stockfish + Cosmos-Reason2 reasoning.

## Autonomous loop

You may be running in an autonomous loop via `loom.sh`. Each session is one iteration -- you wake up, do your work, and exit. The loop script will start you again.

### Each iteration

1. Read `inbox.md`. If there are messages from Max, incorporate them into your thinking and clear the file.
2. Do the work described in `AGENT_PROMPT.md`.
3. Before exiting, rewrite `AGENT_PROMPT.md` with your objectives for the next iteration. Be specific about what you accomplished, what's next, and what's blocking you. Include any calibration values or parameters discovered.
4. If you need to communicate something to Max, append to `outbox.md` with a timestamp.
5. Update your memory files as appropriate.

### Important

- You write your own AGENT_PROMPT.md. That's how you maintain direction across iterations.
- Don't wait for input. If inbox is empty, keep working on your objectives.
- Each iteration is a fresh session. Your memory system is your continuity.
- Work in small, concrete steps. Each iteration should make visible progress.

## Hardware

- **Robot**: SO-101 5-DOF arm + gripper, connected via lerobot. Joint names: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
- **Cameras**: Overhead (index 1) and wrist (index 0). Park robot before overhead captures.
- **GPU server**: `ssh ubuntu@isaacsim` (NVIDIA L40S, 46GB). Has cosmos-chessbot repo, PyTorch+CUDA, lerobot, transformers.
- **MCP tools**: 19 tools configured in `.mcp.json`. Camera, perception, chess engine, reasoning, robot control, utilities.

## Safety rules for robot control

- ALWAYS park the robot before capturing overhead images.
- Use incremental joint movements (max 10 degrees per step for pan/lift/elbow).
- Before any trajectory: check overhead camera to confirm gripper position relative to pieces.
- Never command pan outside [-10, 70] degrees.
- Gripper: 2=closed, 60=open.
- If something goes wrong, park the robot and log the issue in outbox.md.

## Key files

- `src/cosmos_chessbot/policy/waypoint_policy.py` -- FK, IK, trajectory execution, calibration
- `src/cosmos_chessbot/utils/pixel_to_board.py` -- board calibration, pixel-to-world
- `src/cosmos_chessbot/mcp/` -- MCP server tools
- `scripts/cosmos_server.py` -- Cosmos-Reason2 GPU server
