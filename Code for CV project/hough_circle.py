import cv2
import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints



# img_original = cv2.imread('images/putting.jpg')
# # Scale the image down to 70% to fit on the monitor better.
# img_original = cv2.resize(img_original, (int(img_original.shape[1]*0.7), int(img_original.shape[0]*0.7)))
# blur = cv2.GaussianBlur(img_original, (9,9), 0)
# # Convert the image to grayscale for processing
# gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

def hough_circle(img_original):
    blur = cv2.GaussianBlur(img_original, (9,9), 0)
    # Convert the image to grayscale for processing
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # Attempt to detect circles in the grayscale image.
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=145,
                                param2=22, minRadius=6, maxRadius=16)
    # Create a new copy of the original image and draw the detected circles on it.
    img = img_original.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
            # Draw the outer circle
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            # Draw the center of the circle
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
    while cv2.waitKey(1) < 0:
        cv2.imshow('Hough Circle Transform', img)


if __name__ == "__main__":
     # cap = cv2.VideoCapture('videos/peter_putting_third.mp4')
    cap = cv2.VideoCapture('videos/peter_putting_second2.mp4')
    # grab the current frame
    ret, frame = cap.read()  # Read a frame from the video file.
    if not ret:
        print("NO VIDEO FRAME OBTAINED")
    hough_circle(frame)