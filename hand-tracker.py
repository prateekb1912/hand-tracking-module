import cv2
import mediapipe as mp
import time
import numpy as np


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
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            # Draw all of the hand landmarks 
            for handLms in results.multi_hand_landmarks:
                for idx, lm in enumerate(handLms.landmark):
                    h,w,c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED) 

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    # Start the video stream and end only when user hits 'ESC'
    while(cv2.waitKey(1) != 27):
        _, img = cap.read()

        cTime = time.time()
        fps = 1/(cTime - pTime)     # Calculated frame rate for the stream
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 255, 0), 3)

        cv2.imshow("Video Stream", img)


if __name__ == "__main__":
    main()
