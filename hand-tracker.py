import cv2
import mediapipe
import time
import numpy as np

def main():
    pTime  = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    while(True):
        _, img = cap.read()

        cTime =     time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 255, 0), 3)

        cv2.imshow("Video Stream", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()