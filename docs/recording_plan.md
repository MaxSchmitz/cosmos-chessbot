# Recording Plan

## Setup
- Terminal 1: `lerobot-teleoperate --robot.type=so101_follower --robot.port=/dev/tty.usbmodem58FA0962531 --robot.id=my_follower_arm --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem5AB01829561 --teleop.id=my_leader_arm`
- Terminal 2: preview_hud_feed.py with recording flags (see below)
- Corner confidence threshold: 0.97

## Shots

### Shot 1: Static board (Act 1) -- DONE
```bash
uv run python scripts/hardware/preview_hud_feed.py --no-hud --record shots/act1_board.mp4
```
Board with pieces in starting position, arm parked. 5s.

### Shot 2: Pick e2 place e4 with HUD (Act 3b) -- IN PROGRESS
```bash
uv run python scripts/hardware/preview_hud_feed.py --source e2 --target e4 --record shots/act3b_e2e4.mp4
```
Teleoperate: pick e2 pawn, place on e4. Green circle on e2, magenta on e4.

### Shot 3: Wrist cam view of same move (Act 3b)
```bash
uv run python scripts/hardware/preview_hud_feed.py --no-hud --cam 0 --record shots/act3b_wrist.mp4
```
Wrist camera during pick-and-place. No HUD needed.

### Shot 4: Post-move board (Act 3c)
```bash
uv run python scripts/hardware/preview_hud_feed.py --no-hud --record shots/act3c_post_move.mp4
```
Board after e2->e4 completed. Arm parked. For Cosmos verification overlay (added in post).

### Shot 5: Staged failure (Act 3d)
```bash
uv run python scripts/hardware/preview_hud_feed.py --source e2 --target e4 --record shots/act3d_failure.mp4
```
Teleoperate: pick e2 pawn, place on WRONG square (e.g. d4). HUD still shows e4 target.

### Shot 6: Recovery (Act 3d)
```bash
uv run python scripts/hardware/preview_hud_feed.py --source d4 --target e4 --record shots/act3d_recovery.mp4
```
Teleoperate: pick from wrong square (d4), place on correct (e4). Green on d4, magenta on e4.

### Shot 7: Multi-move sequence (Act 4)
Record 3-4 moves in sequence. Reset HUD between moves. Will be sped up in post.
```bash
# Move 1: e2->e4
uv run python scripts/hardware/preview_hud_feed.py --source e2 --target e4 --record shots/act4_move1.mp4
# Move 2: d2->d4
uv run python scripts/hardware/preview_hud_feed.py --source d2 --target d4 --record shots/act4_move2.mp4
# Move 3: g1->f3
uv run python scripts/hardware/preview_hud_feed.py --source g1 --target f3 --record shots/act4_move3.mp4
```

## Post-training shots (need GPU)
- Cosmos reasoning text overlays on shots 4, 5, 6
- YOLO detection boxes screen recording
- Architecture diagram animation

## Notes
- Record clean feed (no UI indicators burned in)
- Cosmos text added in post-production on saved frames
- Speed up Act 4 footage 4-8x in edit
