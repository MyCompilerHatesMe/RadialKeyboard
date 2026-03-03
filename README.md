# Radial Keyboard

A gesture-driven text input system using hand tracking. No keyboard required — rotate your left hand to aim, close your fist to type.

## How it works

Two hands, two roles:

**Left hand** — navigation
- **Depth (distance to camera)** snaps between three letter rings
- **Rotation** sweeps a clock-hand pointer around the active ring to select a character
- **Fist close** types the highlighted character
- **Pinch** (thumb + index) inserts a space

**Right hand** — modifiers
- **Rotate left** (wrist pointing left) deletes the last character
- **Pinch** (thumb + index) toggles caps lock

## Letter layout

Letters distributed by English frequency — most common letters on the inner ring, least movement required.

| Ring  | Letters           |
|-------|-------------------|
| Inner | E T A O I N S R   |
| Mid   | H L D C U M F P G |
| Outer | W Y B V K X J Q Z |

## Setup

**Dependencies**
```bash
pip install opencv-python mediapipe numpy
```

**MediaPipe model**

Download `hand_landmarker.task` from the [MediaPipe releases page](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) and place it at:
```
models/hand_landmarker.task
```

**Run**
```bash
python radialKeyboard.py
```

Two windows open — camera feed and keyboard UI.

## Architecture

### Live stream mode
MediaPipe runs in `LIVE_STREAM` mode with a callback, meaning hand detection happens asynchronously on a separate thread. The main loop calls `tracker.processFrame()` each frame and reads results via `tracker.getLatestResult()` — so detection never blocks rendering.

### Pre-baked UI backgrounds
Static elements (rings, letters, legend, text box) are rendered once at startup into six cached frames — one per ring × one per caps state. Each frame tick just copies the right cached background and draws only the dynamic elements on top (highlighted letter, pointer, typed text). This avoids redrawing unchanged geometry every frame.

### Smoothing
Three layers of smoothing prevent jitter:

- **Angle smoothing** — exponential smoothing on the wrist→MCP angle with circular-aware wraparound so it never snaps across the ±π boundary
- **Depth smoothing** — exponential smoothing on the hand size value that drives ring selection
- **Index stabilisation** — a hysteresis threshold on top of the raw index so the pointer has to move meaningfully before snapping to the next letter

### Squared distances
`dist2d` returns squared Euclidean distance. All thresholds (`DEPTH_NEAR`, `DEPTH_FAR`, `PINCH_THRESHOLD`) are stored pre-squared to avoid a `sqrt` call on every landmark pair every frame.

### Angle-to-index mapping
Rather than mapping the full 360° of hand rotation to the letter ring, only a physical arc of `PHYSICAL_RANGE_LIMIT` radians (default ±60°) centred around `HAND_REST_OFFSET` is used. This means comfortable wrist movement covers the whole ring without needing to fully rotate your hand.

## Tuning

| Constant | Default | Effect |
|----------|---------|--------|
| `PHYSICAL_RANGE_LIMIT` | `π/7` | Arc of physical wrist rotation that covers the full ring |
| `HAND_REST_OFFSET` | `-1.5` | Angle (radians) treated as the centre/neutral position |
| `DEPTH_NEAR` | `0.24²` | Squared wrist→midMCP distance threshold for inner ring |
| `DEPTH_FAR` | `0.13²` | Threshold for mid vs outer ring |
| `PINCH_THRESHOLD` | `0.05²` | Squared thumb-index distance to register as pinch |
| `LEFT_ROTATE_THRESHOLD` | `1.5` | Right hand angle (radians) that triggers backspace |
| `SMOOTHING_FACTOR` | `0.18` | Angle smoothing weight — higher = more responsive, more jitter |
| `DEPTH_SMOOTH` | `0.15` | Depth smoothing weight — lower = more stable ring selection |
| `INDEX_CHANGE_THRESHOLD` | `0.15` | Hysteresis on index snapping — higher = harder to change letter |

To calibrate depth zones, print `dist2d(landmarks[0], landmarks[9])` (the raw squared value) while holding your hand at each position, then set `DEPTH_NEAR` and `DEPTH_FAR` to the squared versions of those numbers.

To calibrate the angle mapping, print `tracker.getHandOrientation(landmarks)` with your hand at its natural rest position and set `HAND_REST_OFFSET` to that value.

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `C` | Clear text |

## Files

```
├── radialKeyboard.py       # main app
├── handTracker.py          # MediaPipe wrapper (live stream mode)
└── models/
    └── hand_landmarker.task
```

## Planned
- Interactive calibration mode for depth and angle offsets
- Performance optimisations
