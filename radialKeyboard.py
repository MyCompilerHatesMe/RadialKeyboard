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
import numpy as np
import math

from handTracker import HandTracker

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
# distances are square cuz dist2d returns square distance
DEPTH_NEAR = 0.12 ** 2
DEPTH_FAR = 0.08 ** 2

PINCH_THRESHOLD = 0.03 ** 2

# radians, for right hand
LEFT_ROTATE_THRESHOLD = 2.2 

# sensitivity stuff
PHYSICAL_RANGE_LIMIT = math.pi / 3
HAND_REST_OFFSET = -1.5

# smoothing 
SMOOTHING_FACTOR = 0.2
DEPTH_SMOOTH = 0.15
INDEX_CHANGE_THRESHOLD = 0.2

# letter lookup
LETTER_POSITIONS = []

UI_STATES_UPPER = []
UI_STATES_LOWER = []

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

def populateLetterPositions():
    for ringIndex, letters in enumerate(RINGS):
        ringPoints = []
        radius = RING_RADII[ringIndex]
        for letterIndex in range(len(letters)):
            ringPoints.append(letterPos(WIDTH//2, HEIGHT//2, radius, letterIndex,len(letters)))
        LETTER_POSITIONS.append(ringPoints)


def createUIBackgrounds(centerX, centerY):
    backgroundsCaps = []
    backgroundsLower = []

    # contains all the static elements
    baseFrame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    baseFrame[:] = COLOR_BACKGROUND

    # ------- legend
    for i, text in enumerate(GESTURE_LEGEND):
        cv.putText(baseFrame, text, (WIDTH - 300, HEIGHT - 300 + i*26), 
               FONT, 0.42, COLOR_GRAY, 1, cv.LINE_AA)
    
    # ------- text output 
    boxX, boxY = 40, HEIGHT - 90
    boxWidth, boxHeight = WIDTH - 80, 60
    # fill
    cv.rectangle(baseFrame, (boxX, boxY), (boxX + boxWidth, boxY + boxHeight), COLOR_RECT_FILL, -1)
    # outline
    cv.rectangle(baseFrame, (boxX, boxY), (boxX + boxWidth, boxY + boxHeight), COLOR_RECT_OUTLINE, 2)

    # ------- center
    cv.circle(baseFrame, (centerX, centerY), 4, COLOR_LIGHTER_GRAY, -1, cv.LINE_AA)

    # create a different background for each of the active rings  
    for activeIndex in range(3):
        backgroundUpper = np.copy(baseFrame)
        backgroundLower = np.copy(baseFrame)

        for i, (radius, color) in enumerate(zip(RING_RADII, RING_COLORS)):
            isActive = i == activeIndex

            drawColor = color if isActive else tuple(c//5 for c in color)
            thickness = 3 if isActive else 1

            cv.circle(backgroundUpper, (centerX, centerY), radius, drawColor, thickness, cv.LINE_AA)
            cv.circle(backgroundLower, (centerX, centerY), radius, drawColor, thickness, cv.LINE_AA)

            letters = RINGS[i]
            for j, char in enumerate(letters):
                pos = LETTER_POSITIONS[i][j]
                # draw letters in idle, probably at some point add more for lower and caps states
                f_scale = 0.45 if isActive else 0.35
                f_thick = 1
                
                (width, height), _ = cv.getTextSize(char.upper(), FONT, f_scale, f_thick)
                cv.putText(backgroundUpper, char.upper(), (pos[0] - width//2, pos[1] + height//2), 
                           FONT, f_scale, COLOR_LIGHT_GRAY, f_thick, cv.LINE_AA)
                
                # no need to recalculate cuz its a simplex font
                cv.putText(backgroundLower, char.lower(), (pos[0] - width//2, pos[1] + height//2), 
                           FONT, f_scale, COLOR_LIGHT_GRAY, f_thick, cv.LINE_AA)

        backgroundsCaps.append(backgroundUpper)
        backgroundsLower.append(backgroundLower)
    return backgroundsCaps, backgroundsLower
        


def drawUIWindow(uiFrame, centerX, centerY, activeRingIndex, activeIndex, leftHandAngle, caps, text):
    uiFrame[:] = UI_STATES_UPPER[activeRingIndex] if caps else UI_STATES_LOWER[activeRingIndex]
    
    # highlighted letter
    radius = RING_RADII[activeRingIndex]
    letters = RINGS[activeRingIndex]
    pos = LETTER_POSITIONS[activeRingIndex][activeIndex]
    color = RING_COLORS[activeRingIndex]

    cv.circle(uiFrame, pos, 12, color, -1, cv.LINE_AA)
    char = letters[activeIndex].upper() if caps else letters[activeIndex].lower()
    (width, height), _ = cv.getTextSize(char, FONT, 0.5, 2)
    cv.putText(uiFrame, char, (pos[0] - width//2, pos[1] + height//2), 
               FONT, 0.5, COLOR_BLACK, 2, cv.LINE_AA)

    # ------- pointers 
    if leftHandAngle is not None:
        pointerEndX, pointerEndY = LETTER_POSITIONS[activeRingIndex][activeIndex]
        cv.line(uiFrame, (centerX, centerY), (pointerEndX, pointerEndY), COLOR_LIGHT_GRAY, 2, cv.LINE_AA)

    # clamp text
    visibleText = (text[-55:] if len(text) > 55 else text)
    visibleText += "|" # cursor
    cv.putText(uiFrame, visibleText, (54 + 14, HEIGHT-50), FONT, 0.75, COLOR_OUTPUT_TEXT, 2, cv.LINE_AA)

    # ------- hud
    cv.putText(uiFrame, f"Ring: {RING_LABELS[activeRingIndex]}", (20, 36), FONT, 0.6, RING_COLORS[activeRingIndex], 2, cv.LINE_AA)

    if caps:
        cv.putText(uiFrame, "CAPS", (WIDTH-100, 36), FONT, 0.65, COLOR_CAPS_INDICATOR, 2, cv.LINE_AA)



# returns square distance to save on square rooting
def dist2d(a, b):
    return ((a.x-b.x)**2 + (a.y-b.y)**2)

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
    if not landmarkerResult or not landmarkerResult.hand_landmarks:
        return leftHand, rightHand
    
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

    prevTimestamp_ms = 0

    populateLetterPositions()

    global UI_STATES_UPPER
    global UI_STATES_LOWER
    UI_STATES_UPPER, UI_STATES_LOWER = createUIBackgrounds(centerX, centerY)
    
    with HandTracker() as tracker:
        while True:
            leftAngle = None
            _, frame= cap.read()

            frame = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB) # flip and get rgb

            tracker.processFrame(frame)

            results = tracker.getLatestResult()

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