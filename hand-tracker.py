import cv2
import mediapipe as mp
import time
import numpy as np
from HandDetectorModule import HandDetector

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector(min_detect_conf = 0.6, min_track_conf = 0.7)

    # Start the video stream and end only when user hits 'ESC'
    while(cv2.waitKey(1) != 27):
        _, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        cTime = time.time()
        fps = 1/(cTime - pTime)     # Calculated frame rate for the stream
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 255, 0), 3)

        cv2.imshow("Video Stream", img)


if __name__ == "__main__":
    main()
