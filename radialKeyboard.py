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

FONT = cv.FONT_HERSHEY_SIMPLEX

WIDTH = 1280
HEIGHT = 720

# these are actually sizes of the hand
# bigger number => closer 
DEPTH_NEAR = 0.12
DEPTH_FAR = 0.08

PINCH_THRESHOLD = 0.03

# radians, for right hand
LEFT_ROTATE_THRESHOLD = 2.2 

# sensitivity stuff
PHYSICAL_RANGE_LIMIT = math.pi / 3
HAND_REST_OFFSET = -1.3

# smoothing 
SMOOTHING_FACTOR = 0.2
DEPTH_SMOOTH = 0.15
INDEX_CHANGE_THRESHOLD = 0.2

# ui consts, colors are in bgr for cv2
COLOR_BACKGROUND = (30, 25, 25)

COLOR_BLACK = (0, 0, 0)
COLOR_GRAY = (120, 120, 120)
COLOR_LIGHT_GRAY = (200, 200, 200)
COLOR_LIGHTER_GRAY = (225, 225, 225)

COLOR_RECT_FILL = (55, 35, 35)
COLOR_RECT_OUTLINE = (90, 90, 150)

COLOR_OUTPUT_TEXT = (255, 210, 200)
COLOR_CAPS_INDICATOR = (60, 220, 255)
RING_LABELS = ("INNER", "MID", "OUTER")
GESTURE_LEGEND = (
    "Left Fist = type", "Left Pinch = space", "Right rotate left = back", "Right pinch = caps",
    "q = quit", "c = clear"
)

def isPinch(landmarkerResult):
    return dist2d(landmarkerResult[4], landmarkerResult[8]) < PINCH_THRESHOLD


def smoothValue(currentVal, lastVal, factor):
    return (currentVal * factor) + (lastVal * (1.0 - factor))

def smoothAngle(currentAngle, lastAngle, factor):
    diff = (currentAngle - lastAngle + math.pi) % (2 * math.pi) - math.pi
    return lastAngle + diff * factor

def getStableIndex(rawIndex, lastIndex, totalItems):
    diff = rawIndex - lastIndex

    if diff > totalItems/2: diff -= totalItems
    if diff < -totalItems/2: diff += totalItems

    if abs(diff) > (0.5 + INDEX_CHANGE_THRESHOLD):
        return math.floor(rawIndex + 0.5) % totalItems
    return lastIndex


def letterPos(centerX, centerY, radius, index, totalLetters):
    angle = (index/totalLetters) * (2 * math.pi) - (math.pi/2)
    # minus pi/2 to start at 12

    x = centerX + radius * math.cos(angle)
    y = centerY + radius * math.sin(angle)

    return int(x), int(y)

def drawUIWindow(uiFrame, centerX, centerY, activeRingIndex, activeIndex, leftHandAngle, caps, text):
    uiFrame[:] = COLOR_BACKGROUND

    # ------- circles 
    for i, (radius, color) in enumerate(zip(RING_RADII, RING_COLORS)):
        isActive = i == activeRingIndex
        
        drawColor = color if isActive else tuple(c//5 for c in color)
        thickness = 3 if isActive else 1

        cv.circle(uiFrame, (centerX, centerY), radius, drawColor, thickness, cv.LINE_AA)
    
    # ------- letters display
    for ring, (letters, radius, color) in enumerate(zip(RINGS, RING_RADII, RING_COLORS)):
        length = len(letters)

        isActive = ring == activeRingIndex

        for j, character in enumerate(letters):
            pos = letterPos(centerX, centerY, radius, j, length)
            isLit = isActive and (j == activeIndex)

            if isLit:
                cv.circle(uiFrame, pos, 12, color, -1, cv.LINE_AA)
                textColor = COLOR_BLACK
                fontScale = 0.5
                fontThickness = 2
            
            elif isActive:
                textColor = COLOR_LIGHT_GRAY
                fontScale = 0.45
                fontThickness = 1
            
            else:
                textColor = COLOR_LIGHT_GRAY
                fontScale = 0.35
                fontThickness = 1

            displayChar = character.upper() if caps else character.lower()

            (textWidth, textHeight), _ = cv.getTextSize(
                displayChar,
                FONT,
                fontScale,
                fontThickness
            )


            cv.putText(uiFrame, displayChar, (pos[0] - textWidth//2, pos[1] + textHeight//2), FONT,
                       fontScale, textColor, fontThickness, cv.LINE_AA)

    # ------- pointers 
    if leftHandAngle is not None:
        pointerRadius = RING_RADII[activeRingIndex] + 12
        totalLetters = len(RINGS[activeRingIndex])
        pointerEndX, pointerEndY = letterPos(centerX, centerY, pointerRadius, activeIndex, totalLetters)

        cv.line(uiFrame, (centerX, centerY), (pointerEndX, pointerEndY), COLOR_LIGHT_GRAY, 2, cv.LINE_AA)

    # ------- center
    cv.circle(uiFrame, (centerX, centerY), 4, COLOR_LIGHTER_GRAY, -1, cv.LINE_AA)

    # ------- text output 
    boxX, boxY = 40, HEIGHT - 90
    boxWidth, boxHeight = WIDTH - 80, 60
    # fill
    cv.rectangle(uiFrame, (boxX, boxY), (boxX + boxWidth, boxY + boxHeight), COLOR_RECT_FILL, -1)
    # outline
    cv.rectangle(uiFrame, (boxX, boxY), (boxX + boxWidth, boxY + boxHeight), COLOR_RECT_OUTLINE, 2)

    # clamp text
    visibleText = (text[-55:] if len(text) > 55 else text)
    visibleText += "|" # cursor
    cv.putText(uiFrame, visibleText, (boxX + 14, boxY + 40), FONT, 0.75, COLOR_OUTPUT_TEXT, 2, cv.LINE_AA)

    # ------- hud
    cv.putText(uiFrame, f"Ring: {RING_LABELS[activeRingIndex]}", (20, 36), FONT, 0.6, RING_COLORS[activeRingIndex], 2, cv.LINE_AA)

    if caps:
        cv.putText(uiFrame, "CAPS", (WIDTH-100, 36), FONT, 0.65, COLOR_CAPS_INDICATOR, 2, cv.LINE_AA)

    # ------- legend
    for index, legendText in enumerate(GESTURE_LEGEND):
        cv.putText(uiFrame, legendText, (WIDTH - 300, HEIGHT - 300 + index*26), 
                   FONT, 0.42, COLOR_GRAY, 1, cv.LINE_AA)



def dist2d(a, b):
    return math.hypot(a.x-b.x, a.y-b.y)

def getRing(depth):
    if depth > DEPTH_NEAR: return 0 # inner circle
    if depth > DEPTH_FAR: return 1 # middle
    return 2 # outer

def angleToIndex(angle, totalItems):
    relativeAngle = angle - HAND_REST_OFFSET
    relativeAngle = (relativeAngle + math.pi) % (2 * math.pi) - math.pi 
    scaledProgress = (relativeAngle / PHYSICAL_RANGE_LIMIT)
    rawIndex = scaledProgress * (totalItems / 2)
    return math.floor(rawIndex) % totalItems


def seperateHands(landmarkerResult):
    leftHand, rightHand = None, None
    if not landmarkerResult.hand_landmarks:
        return None, None
    
    for index, handLandmarks in enumerate(landmarkerResult.hand_landmarks):
        handedness = landmarkerResult.handedness[index][0].category_name
        if handedness == "Right":
            leftHand = handLandmarks
        elif handedness == "Left":
            rightHand = handLandmarks
    
    return leftHand, rightHand



def main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    uiFrame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    centerX, centerY = WIDTH//2, HEIGHT//2

    caps = False
    typed = ""

    # states
    prevLeftOpen, prevLeftPinch = True, False
    
    prevRightPinch = False
    prevRightRotatedLeft = False
    
    activeRingIndex, activeIndex = 1, 0

    prevSmoothedLeftAngle = -1.3
    prevSmoothedLeftDepth = 0.0
    lastStableIndex = 0
    

    with HandTracker(options) as tracker:
        while True:
            leftAngle = None
            _, frame= cap.read()
            frame = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB) # flip and get rgb

            results = tracker.getLandMarks(frame)

            leftHandLandmarks, rightHandLandmarks = seperateHands(results)

            if leftHandLandmarks:
                depth = dist2d(leftHandLandmarks[0], leftHandLandmarks[9])
                smoothedLeftDepth = smoothValue(depth, prevSmoothedLeftDepth, DEPTH_SMOOTH)
                
                activeRingIndex = getRing(smoothedLeftDepth)
                ringLetters = RINGS[activeRingIndex]

                leftAngle = tracker.getHandOrientation(leftHandLandmarks)
                smoothedLeftAngle = smoothAngle(leftAngle, prevSmoothedLeftAngle, SMOOTHING_FACTOR)

                rawIndex = angleToIndex(smoothedLeftAngle, len(ringLetters))
                activeIndex = getStableIndex(rawIndex, lastStableIndex, len(ringLetters))

                leftOpen = tracker.isHandOpen(leftHandLandmarks)
                leftPinch = isPinch(leftHandLandmarks)

                # fist close => type
                if prevLeftOpen and not leftOpen and not leftPinch:
                    character = ringLetters[activeIndex]
                    typed += character.upper() if caps else character.lower()
                
                # pinch => space
                if leftPinch and not prevLeftPinch:
                    typed += " "

                prevLeftOpen = leftOpen
                prevLeftPinch = leftPinch
                prevSmoothedLeftAngle = smoothedLeftAngle
                prevSmoothedLeftDepth = smoothedLeftDepth

            if rightHandLandmarks:
                rightPinch = isPinch(rightHandLandmarks)
                rightAngle = tracker.getHandOrientation(rightHandLandmarks)

                rightRotatedLeft = abs(rightAngle) > LEFT_ROTATE_THRESHOLD

                if rightPinch and not prevRightPinch:
                    caps = not caps
                
                if rightRotatedLeft and not prevRightRotatedLeft:
                    typed = typed[:-1]

                prevRightPinch = rightPinch
                prevRightRotatedLeft = rightRotatedLeft

            drawUIWindow(uiFrame, centerX, centerY, activeRingIndex, activeIndex, leftAngle, caps, typed)

            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            cv.imshow("Camera", frame)
            cv.imshow("UI", uiFrame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                typed = ""
    
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()