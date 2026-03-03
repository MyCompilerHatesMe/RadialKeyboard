import mediapipe as mp
import numpy as np
import time

class HandTracker:
    
    mp_hands = mp.tasks.vision.HandLandmarksConnections
    mp_drawing = mp.tasks.vision.drawing_utils
    mp_drawing_styles = mp.tasks.vision.drawing_styles

    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    def __init__(self, modelPath = "models/hand_landmarker.task"):
        self.latestResult = None
        self._lastTimestamp_ms = 0
        
        def resultCallback(result, outputImage, timestamp_ms):
            self.latestResult = result
        
        baseOptions = mp.tasks.BaseOptions(model_asset_path=modelPath)
        options = self.HandLandmarkerOptions(
            base_options=baseOptions,
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            result_callback=resultCallback
        )

        self.landmarker = self.HandLandmarker.create_from_options(options)

    def processFrame(self, rgbImage):
        timestamp_ms = int(time.time() * 1000)
        # timestamp must be strictly increasing
        if timestamp_ms <= self._lastTimestamp_ms:
            timestamp_ms = self._lastTimestamp_ms + 1
        self._lastTimestamp_ms = timestamp_ms
    
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbImage)
        self.detectAsync(mpImage, timestamp_ms)

    def detectAsync(self, mpImage, timestamp_ms):
        self.landmarker.detect_async(mpImage, timestamp_ms)

    def getLatestResult(self):
        return self.latestResult

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
        if not self.latestResult or not self.latestResult.hand_landmarks:
            return rgbImage
        
        landmarkList = self.latestResult.hand_landmarks
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