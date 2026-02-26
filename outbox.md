# Outbox

## 2026-02-26 ~2:00 AM - Iteration 2 report

### Progress
- **Robot-to-world calibration done**: 4 data points, RMS=19.7mm. Constants updated in waypoint_policy.py.
- **Z offset revised to -0.04** (was -0.07). The arm knocked a piece at FK_Z=-0.019, so the board can't be at -0.07.
- **Cosmos server running on brev**: Both perception and reasoning models loaded. Tested action reasoning endpoint -- returns valid grasp strategies. Needs SSH tunnel to access from local machine (`ssh -f -N -L 8000:localhost:8000 ubuntu@isaacsim`).
- **Camera reconnect tool added**: `reconnect_cameras` in tools_camera.py. Will work after MCP server restart.
- **Pi0.5 explored**: Available on HF as `lerobot/pi05_base` but needs lerobot >= 0.4.x (current: 0.3.2).

### Blockers
1. **Elbow servo stuck at 100 degrees**. Cannot increase past ~99.6 regardless of commanded angle. This is the main mechanical constraint -- limits how low the gripper can reach. Can you check if the elbow servo is physically jammed or if there's a travel limit?
2. **Cameras died mid-session** (USB issue). Both overhead and wrist cameras return no frames. The reconnect tool should fix this on restart.
3. **Grasp attempt failed** -- positioned at H2 area, closed gripper, no piece caught (gripper closed freely to 2.25 degrees). Likely XY or Z positioning error given the 19.7mm calibration RMS.

### What I need from you
- Check elbow servo physically -- can it flex further than its current position?
- If possible, move a pawn to a very accessible square (like the one right in front of the gripper at home position) for an easier first grasp.
- The cameras should recover with a replug or on MCP restart.
