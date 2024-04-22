import cv2
import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints


if __name__ == "__main__":
    # track_ball()

    cap = cv2.VideoCapture('videos/peter_putting_second2.mp4')
    

    while cv2.waitKey(0):  # this steps through each frame when you press 'q'
        
        # grab the current frame
        ret, frame = cap.read()  # Read a frame from the video file.
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # convert image from BGR to HSV

        # Define the upper and lower colour thresholds for the ball colour.
        lower = np.array([70, 0, 95], dtype="uint8")  
        upper = np.array([180, 80, 255], dtype="uint8") 
        mask = cv2.inRange(frame_HSV, lower, upper)  # Find all pixels in the image within the colour range.

        # Find a series of points which outline the shape in the mask.
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and process contours
        i = 0
        for contour in contours:
            # Fit minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            # Calculate contour area
            contour_area = cv2.contourArea(contour)
            
            # Calculate area of enclosing circle
            circle_area = np.pi * (radius ** 2)
            
            # Calculate ratio of contour area to circle area
            circularity = contour_area / circle_area

            # Find the best circle
            # if 
            
            # Filter contours based on circularity threshold
            if (i == 0):
                best_circularity = circularity
                bestCircle = contour
            if circularity > best_circularity:
                best_circularity = circularity
                bestCircle = contour
                print(best_circularity)
            i += 1

        (x, y), radius = cv2.minEnclosingCircle(bestCircle)
        center = np.array([int(x),int(y)])
        radius = int(radius)

        cv2.circle(frame, tuple(center), radius,(0,255,0), 2)
        # Draw the center (not centroid!) of the ball.
        cv2.circle(frame, tuple(center), 1,(0,255,0), 2)
        cv2.imshow('frame', frame)  # Display the grayscale frame on the screen.





