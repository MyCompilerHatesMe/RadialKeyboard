import cv2 as cv
import mediapipe as mp
import numpy as np

class HandTracker:
    
    mp_hands = mp.tasks.vision.HandLandmarksConnections
    mp_drawing = mp.tasks.vision.drawing_utils
    mp_drawing_styles = mp.tasks.vision.drawing_styles
    HandLandmarker = mp.tasks.vision.HandLandmarker

    def __init__(self, options):
        self.landmarker = self.HandLandmarker.create_from_options(options)

    def getLandMarks(self, rgbImage):
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbImage)
        return self.landmarker.detect(mpImage)

    def close(self):
        self.landmarker.close()
    
    # returns hands angle in rads
    def getHandOrientation(self, handLandmarks):
        wrist = handLandmarks[0]
        midMcp = handLandmarks[9]
        dx = midMcp.x - wrist.x
        dy = midMcp.y - wrist.y
        return np.atan2(dy, dx)

    # returns whether three or more fingers are open
    def isHandOpen(self, handLandmarks):
        fingers = [
            (8, 6),
            (12, 10),
            (16, 14),
            (20, 18)
        ]
        wrist = handLandmarks[0]
        extended = 0
        for tipId, pipId in fingers:
            tip = handLandmarks[tipId]
            pip = handLandmarks[pipId]
            tipDist = (tip.x - wrist.x)**2 + (tip.y - wrist.y)**2
            pipDist = (pip.x - wrist.x)**2 + (pip.y - wrist.y)**2
            if tipDist > pipDist:
                extended += 1
        return extended >= 3

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
    
    def drawLandmarksOnImage(self, rgbImage):
        landmarkerResult = self.getLandMarks(rgbImage)
        
        if not landmarkerResult.hand_landmarks:
            return rgbImage
        
        landmarkList = landmarkerResult.hand_landmarks
        returnImage = np.copy(rgbImage)
        for landmarks in landmarkList:
            self.mp_drawing.draw_landmarks(
                returnImage,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        return returnImage

# options set up

HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
modelPath = "models/hand_landmarker.task"
baseOptions = mp.tasks.BaseOptions(model_asset_path=modelPath)

options = HandLandmarkerOptions(
    base_options=baseOptions,
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)