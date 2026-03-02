"""
Radial Keyboard — two-hand gestural text input
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GESTURES
  Left hand depth (size) → snaps active ring: inner / mid / outer
  Left hand rotation (wrist → MCP) → sweeps clock-hand around the active ring
  Left hand fist close → TYPE highlighted letter
  Left hand pinch (thumb + index) → SPACE
  Right hand rotate left → BACKSPACE
  Right hand pinch (thumb + index) → toggle CAPS LOCK

LETTER LAYOUT (ordered by frequency)
  Inner  (8) : E T A O I N S R
  Mid    (9) : H L D C U M F P G
  Outer  (9) : W Y B V K X J Q Z

KEYS
  Q  - quit
  C  - clear typed text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import cv2 as cv
import mediapipe as mp
import numpy as np
import math

from handTracker import HandTracker, options

# constants

RINGS = [
    list("ETAOINSR"), # inner ring
    list("HLDCUMFPG"),
    list("WYBVKXJQZ"), # outer ring
]

RING_RADII = [
    65, 
    110,
    155
]

# bgr
RING_COLORS = [
    (50,230,100), # green, inner
    (255,170,50), # blue, middle
    (40,130,255), # orange, outer
]

ROTATION_SENSITIVITY = 2

FONT = cv.FONT_HERSHEY_SIMPLEX

WIDTH = 1280
HEIGHT = 720

# these are actually sizes of the hand
# bigger number => closer 
DEPTH_NEAR = 0.16
DEPTH_FAR = 0.10


def letterPos(centerX, centerY, radius, index, totalLetters):
    angle = (index/totalLetters) * (2 * math.pi) - (math.pi/2)
    # minus pi/2 to start at 12

    x = centerX + radius * math.cos(angle)
    y = centerY + radius * math.sin(angle)

    return int(x), int(y)

def drawUIWindow(uiFrame, centerX, centerY, activeRing, activeIndex, left_angle, caps, text):
    uiFrame[:] = (30, 25, 25) # gray background

    for radius, color in zip(RING_RADII, RING_COLORS):
        cv.circle(uiFrame, (centerX, centerY), radius, color, 1, cv.LINE_AA)
    
    for ring, (letters, radius, color) in enumerate(zip(RINGS, RING_RADII, RING_COLORS)):
        length = len(letters)
        for j, character in enumerate(letters):
            pos = letterPos(centerX, centerY, radius, j, length)

            label = character.upper() if caps else character.lower()

            cv.putText(uiFrame, label, (pos[0] - 5, pos[1] + 5), FONT,
                       0.35, (100, 100, 100), 1, cv.LINE_AA)


def dist2d(a, b):
    return math.hypot(a.x-b.x, a.y-b.y)

def getRing(landmarks):
    d = dist2d(landmarks[0], landmarks[9])
    if d > DEPTH_NEAR: return 0 # inner circle
    if d > DEPTH_FAR: return 1 # middle
    return 2 # outer

def angleToIndex(angle, totalItems):
    angleFromTop = angle + math.pi/2
    normalized = angleFromTop % (2*math.pi)
    circleFraction = normalized/(2*math.pi)
    position = circleFraction * totalItems * ROTATION_SENSITIVITY
    return int(position) % totalItems

def seperateHands(landmarkerResult):
    hands = landmarkerResult.hand_landmarks
    if not hands: return None, None

    if len(hands) == 1:
        wristX = hands[0][0].x
        isLeft = wristX < 0.5
        return (
            (hands[0], None) if isLeft 
            else (None, hands[0])
        )
    
    handsSortedByX = sorted(
        hands,
        key=lambda hand: hand[0].x
    )
    leftHand = handsSortedByX[0]
    rightHand = handsSortedByX[1]
    
    return leftHand, rightHand


def main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    uiFrame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    centerX, centerY = WIDTH//2, HEIGHT//2

    caps = False

    typed = ""

    with HandTracker(options) as tracker:
        while True:
            _, frame= cap.read()
            frame = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_RGB2BGR) # flip and get rgb

            frame = tracker.drawLandmarksOnImage(frame)

            activeRing = 1
            activeIndex = 0
            leftAngle = None

            drawUIWindow(uiFrame, centerX, centerY, activeRing, activeIndex, leftAngle, caps, typed)

            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            cv.imshow("Camera", frame)
            cv.imshow("UI", uiFrame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    main()