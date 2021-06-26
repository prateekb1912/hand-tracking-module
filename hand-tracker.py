import cv2
import mediapipe as mp
import time
import numpy as np

class HandDetector():
    def __init__(self, mode = False, maxHands = 2, min_detect_conf = 0.5, min_track_conf = 0.5):
        """
        Initializes the hand detector object
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detect_conf = min_detect_conf
        self.track_conf = min_track_conf


        # Create a mediapipe Hands object to detect hands in any image
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detect_conf, self.track_conf)
        self.mpDraw = mp.solutions.drawing_utils




def main():
    pTime  = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    # Start the video stream and end only when user hits 'ESC'
    while(cv2.waitKey(1) != 27):
        _, img = cap.read()

        cTime =     time.time()
        fps = 1/(cTime - pTime)     # Calculated frame rate for the stream
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 255, 0), 3)

        cv2.imshow("Video Stream", img)


if __name__ == "__main__":
    main()