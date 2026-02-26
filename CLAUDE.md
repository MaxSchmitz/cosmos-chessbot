# Cosmos Chessbot

Physical AI chess robot for NVIDIA Cosmos Cookoff (deadline **March 5, 2026 5pm PT**). SO-101 arm + YOLO-DINO perception + Stockfish + Cosmos-Reason2 reasoning. Prizes: $3k/$2k/$500.

## Running scripts

- Local: `uv run python` for project Python scripts
- Brev server: `/home/ubuntu/.local/bin/uv run python` (uv not on PATH in non-interactive SSH)
- macOS `._*` resource fork files appear in datasets copied to brev -- clean with `find data/ -name '._*' -delete`

## Current state (Feb 26, 2026)

### What's working
- **MCP server**: 19 tools for camera, perception, chess, reasoning, robot control. Entry: `scripts/run_mcp_server.py`, config: `.mcp.json`
- **URDF FK/IK**: `fk_urdf()` and `solve_ik_numerical()` in waypoint_policy.py. Sub-mm accuracy.
- **YOLO piece detection**: mAP50=0.984. Weights on brev: `runs/detect/runs/detect/yolo26_chess_combined/weights/best.pt`
- **DINO-MLP classifier**: vits8, 91.3% val acc. Weights: `models/dino_mlp/dino_mlp_best.pth` (84MB)
- **Board calibration**: YOLO pose detects 4 corners, homography maps pixel-to-world
- **Robot-to-world calibration**: fitted from 4 points, RMS=19.7mm (needs improvement)

### Hardware issues
- **Elbow presses against table** at ~100 degrees in home config. Fix: raise robot base 2-3cm.
- **Cameras need USB replug** -- both died during overnight loop.

### Brev server (ssh ubuntu@isaacsim)
- NVIDIA L40S, 46GB VRAM
- **Cosmos server currently stopped** (killed to free GPU). Restart: `cd ~/cosmos-chessbot && nohup ~/.local/bin/uv run python scripts/cosmos_server.py --port 8000 > /tmp/cosmos_server.log 2>&1 &`
- Uses ~34GB VRAM when running. SSH tunnel needed: `ssh -f -N -L 8000:localhost:8000 ubuntu@isaacsim`
- lerobot 0.3.2 installed. Pi0.5 needs 0.4.x.
- YOLO and DINO training complete. Weights in `runs/` and `models/` dirs.

### Next priorities
1. **Explore pi0.5** on brev -- VLA model that could bypass IK calibration entirely
2. **Fix elbow** -- raise robot base, replug cameras
3. **Pick up a chess piece** -- either via geometric IK or pi0.5
4. **Full game loop** with Cosmos integration for demo video

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
