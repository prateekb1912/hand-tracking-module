import cv2
import mediapipe as mp

class HandDetector():
    def __init__(self, mode=False, maxHands=2, min_detect_conf=0.5, min_track_conf=0.5):
        """
        Initializes the hand detector object
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detect_conf = min_detect_conf
        self.track_conf = min_track_conf

        # Create a mediapipe Hands object to detect hands in any image
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detect_conf, self.track_conf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img):
        """
        Finds hand landmarks in the given image and draws lines and points
        on the same 
        """

        # Convert BGR image to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Processing the image to look for hand features
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            # Draw all of the hand landmarks 
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) 
        
        return img