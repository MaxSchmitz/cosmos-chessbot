# Demo Video Script -- Cosmos Chessbot

**Target length**: 2:30 - 3:00
**Submission**: NVIDIA Cosmos Cookoff (cosmos-cookbook GitHub issue)
**Key differentiator**: Only entry doing closed-loop robotic manipulation with Cosmos reasoning. All others are video-analysis-only.

---

## Act 1: The Problem (0:00 - 0:25)

**Visual**: Quick montage of chess computers (Deep Blue, AlphaZero), then cut to SO-101 arm sitting beside a physical chess board.

**Narration**:
> Computers mastered chess strategy decades ago. But no computer has ever physically played a chess game autonomously -- sensing the board, reasoning about the physical scene, and moving pieces without human help.
>
> The gap isn't intelligence. It's embodiment.

---

## Act 2: Architecture (0:25 - 0:55)

**Visual**: Animated pipeline diagram. Each component lights up as described:

```
Camera Feed
    |
    v
YOLO + DINO -----> Board State (FEN)
                        |
                        v
                    Stockfish -----> Best Move (e.g. e2e4)
                        |
                        v
                Cosmos Reason 2 -----> Physical Reasoning
                        |               (obstacles, trajectory,
                        |                grasp strategy, verification)
                        v
                    Pi0.5 VLA -----> Motor Commands
                        |
                        v
                    SO-101 Arm -----> Physical Execution
```

**Narration**:
> We decompose the problem into four specialized models. YOLO and DINO handle fast local perception -- detecting pieces and reading the board in under two seconds. Stockfish computes the optimal move. Cosmos Reason 2 bridges the gap between symbolic planning and physical execution -- it reasons about obstacles, plans trajectories, and verifies outcomes. A vision-language action model executes the motion.
>
> The key insight: Cosmos is the reasoning supervisor. It watches, thinks, and advises -- but never issues motor commands directly.

---

## Act 3: Cosmos Reasoning in Action (0:55 - 2:00)

*This is the core of the video. Show 3-4 scenarios with Cosmos chain-of-thought output overlaid on camera footage.*

### Scene 3a: Board Analysis + Pre-Move Reasoning (~20s)

**Visual**: Overhead camera view of chess board. Cosmos reasoning text slides in as a panel.

**Show**:
1. `analyze_board` output: game phase, position summary, physical observations
2. `reason_about_action` for the planned move: detected obstacles, grasp strategy, risk assessment

**Example overlay**:
```
COSMOS REASON 2 -- Scene Analysis
<think>
I see the board from my overhead camera. White pieces are on ranks 1-2,
black on ranks 7-8. The e2 pawn has clear space above it -- no adjacent
pieces that could be bumped during pickup. The target square e4 is empty
with pawns on d2 and f2 nearby but not blocking the approach path...
</think>

Position: Standard opening, e4 pawn push prepared
Grasp strategy: Top-pinch, clear approach from above
Risks: None -- no adjacent pieces in grasp path
Confidence: 0.92
```

### Scene 3b: Trajectory Planning + Execution (~20s)

**Visual**: Camera view with waypoint circles drawn at each planned position. HUD overlay (green source circle, magenta target circle) appears. Robot executes the move.

**Show**:
1. `plan_trajectory` waypoints overlaid on image: numbered circles with labels
2. HUD overlay activating
3. Robot arm picking up piece and placing it

**Example overlay**:
```
COSMOS REASON 2 -- Trajectory Plan
1. [420, 380] -- approach above e2
2. [420, 380] -- descend to piece
3. [420, 380] -- grasp
4. [420, 350] -- lift clear of board
5. [420, 260] -- transit to e4
6. [420, 260] -- descend to e4
7. [420, 260] -- release
```

### Scene 3c: Goal Verification (~15s)

**Visual**: Post-move board image. Cosmos verification result overlaid.

**Show**: `verify_goal` output confirming success.

**Example overlay**:
```
COSMOS REASON 2 -- Move Verification
<think>
Looking at the board after the move. The e2 square is now empty.
A white pawn is standing upright on e4. No other pieces appear
disturbed. The pawn is centered on the square and stable...
</think>

Result: SUCCESS
Piece correctly placed on e4, stable and upright
Adjacent pieces undisturbed
Confidence: 0.95
```

### Scene 3d: Failure Detection + Recovery (~15s)

**Visual**: Staged failure -- piece lands on wrong square or tips over. Cosmos detects and diagnoses.

**Show**:
1. `verify_goal` detecting failure
2. `plan_correction` diagnosing the physical cause and planning recovery
3. Robot correcting the placement

**Example overlay**:
```
COSMOS REASON 2 -- Move Verification
Result: FAILURE
Piece not detected on target square e4.
Physical issue: piece appears to have slipped during placement

COSMOS REASON 2 -- Recovery Plan
<think>
The pawn was meant for e4 but I see it on d4. It likely slipped
from the gripper during the lateral transit. The d4 square has
the pawn standing upright, so I can re-grasp cleanly...
</think>

Correction: Pick from d4, retry placement on e4
Confidence: 0.88
```

---

## Act 4: Full Game Loop + Closing (2:00 - 2:45)

**Visual**: Sped-up footage of 3-4 moves. Opponent plays, Cosmos detects, robot responds. Brief text annotations for each step.

**Narration**:
> The full game loop runs continuously. Cosmos detects when the opponent has moved and identifies the change. Stockfish computes the response. Cosmos plans the physical execution -- analyzing obstacles, choosing a grasp strategy, planning a safe trajectory. The robot acts. And when things go wrong -- a piece slips, a square is misread -- Cosmos diagnoses the failure and plans recovery.
>
> This is physical AI: not just computing the right answer, but reasoning about the messy, unpredictable real world.

**End card** (2:40 - 2:50):
- Project name: Cosmos Chessbot
- GitHub URL
- Built with: Cosmos Reason 2, Pi0.5, YOLO, DINO, Stockfish, LeRobot, SO-101

---

## Recording Checklist

### Robot footage (teleoperation OK)
- [ ] Board overview: clean overhead shot of starting position
- [ ] Successful pick-and-place: e2 -> e4 (pawn opening)
- [ ] Multiple angles: overhead + wrist camera views during move
- [ ] Staged failure: place piece on wrong square, then correct
- [ ] Opponent turn: human hand moves a black piece, pause, robot responds
- [ ] Multiple moves: 3-4 move sequence for Act 4 montage

### Cosmos reasoning captures
- [ ] `analyze_board` on starting position
- [ ] `reason_about_action` for e2e4
- [ ] `plan_trajectory` for e2e4 (waypoints)
- [ ] `verify_goal` -- success case
- [ ] `verify_goal` -- failure case (staged)
- [ ] `plan_correction` for the failure
- [ ] `detect_move` for opponent's turn

### Screen recordings
- [ ] Cosmos server terminal showing `<think>` output streaming
- [ ] FEN detection pipeline running (YOLO boxes + FEN string)

### Post-production assets
- [ ] Architecture diagram (can be drawn in Excalidraw/Figma)
- [ ] Chess history montage (stock footage or generated)

## Production Notes

- **Cosmos runs after recording**: Capture robot footage first, run Cosmos on saved frames later. No real-time requirement.
- **OOM**: Can't run Cosmos while pi0.5 trains on brev (both need ~30GB+ VRAM). Stop training or wait for it to finish.
- **Overlay**: Use cv2.putText or ffmpeg drawtext for text overlays. Or screen-record a web UI showing results alongside video.
- **Edit cuts hide latency**: Cosmos inference takes 5-15s per call. Cut between input and output in the video.
