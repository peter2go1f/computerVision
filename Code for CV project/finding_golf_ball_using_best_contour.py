import cv2
import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

def find_ball(frame_HSV, lower, upper):
    # Detect a colour ball with a colour range.
    mask = cv2.inRange(frame_HSV, lower, upper)  # Find all pixels in the image within the colour range.

    # Perfrom closing morphology (dilate then erosion) to fill gaps and holes in image
    kernel = np.ones((3,3), np.uint8)
    opening_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # closing_mask = cv2.morphologyEx(opening_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # cv2.imshow('mask', closing_mask)
    cv2.imshow('mask', opening_mask)

    # Find a series of points which outline the shape in the mask.
    contours, _hierarchy = cv2.findContours(opening_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Filter and process contours
        i = 0
        for contour in contours:
            # Fit minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour) # could also set a threshold range for radius here
            
            # Calculate contour area
            contour_area = cv2.contourArea(contour)
            
            # Calculate area of enclosing circle
            circle_area = np.pi * (radius ** 2)
            
            # Calculate ratio of contour area to circle area
            circularity = contour_area / circle_area
            
            # Find the best circle
            if (i == 0):
                best_circularity = circularity
                bestCircle = contour
            if circularity > best_circularity:
                best_circularity = circularity
                bestCircle = contour
            i += 1           

        (x, y), radius = cv2.minEnclosingCircle(bestCircle)
        center = np.array([int(x),int(y)])
        radius = int(radius)
        return center, radius
    else:
        return None, None
    

def track_ball():
    cap = cv2.VideoCapture('videos/peter_putting_second2.mp4')

    # Define the upper and lower colour thresholds for the ball colour.
    lower = np.array([70, 0, 95], dtype="uint8")  
    upper = np.array([180, 80, 255], dtype="uint8") 
    
    # while cv2.waitKey(100) < 0:   # the number determines how fast the video plays
    while cv2.waitKey(0):
        # grab the current frame
        ret, frame = cap.read()  # Read a frame from the video file.
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # convert image from BGR to HSV
        if not ret:              # If we cannot read any more frames from the video file, then exit.
            break

        center, radius = find_ball(frame_HSV, lower, upper)
        if center is not None:
            # Draw circle around the ball.
            cv2.circle(frame, tuple(center), radius,(0,0,255), 2)
            # Draw the center (not centroid!) of the ball.
            cv2.circle(frame, tuple(center), 1,(0,0,255), 2)
        cv2.imshow('frame', frame)  # Display the grayscale frame on the screen.


if __name__ == "__main__":
    track_ball()

    # while cv2.waitKey(0):  # this steps through each frame when you press 'q'
        


