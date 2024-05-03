import cv2
import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints


def difference(cap):
    # Load two initial images from the video to begin
    ret, img0 = cap.read()
    ret, img1 = cap.read()

    # while cv2.waitKey(100) < 0:
    while cv2.waitKey(0):
        diff = cv2.subtract(img0, img1)  # calculate the difference of the two images

        # move the data in img0 to img1.
        img1 = img0
        ret, img0 = cap.read()   # grab a new frame from the video for img0

        ret, diff = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
        diff = cv2.medianBlur(diff, 3)
        diff = cv2.medianBlur(diff, 3)        # repeating medianBlur gets rid of salt and pepper noise
        # diff = cv2.medianBlur(diff, 3)

        cv2.imshow('Difference', diff)    # display the difference on the screen
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = cv2.VideoCapture('videos/peter_putting_fifth_60fps.mp4')
    # cap = cv2.VideoCapture('videos/peter_putting_second2.mp4')
    difference(cap)