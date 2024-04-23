import cv2
import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

def find_ball(frame_HSV, lower_ball, upper_ball):
    # Detect a colour ball with a colour range.
    mask = cv2.inRange(frame_HSV, lower_ball, upper_ball)  # Find all pixels in the image within the colour range.

    # Perfrom closing morphology (dilate then erosion) to fill gaps and holes in image
    kernel = np.ones((3,3), np.uint8)
    opening_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closing_mask = cv2.morphologyEx(opening_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    cv2.imshow('closing_mask', closing_mask)
    cv2.imshow('opening_mask', opening_mask)

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
    cap = cv2.VideoCapture('videos/peter_putting_third.mp4')

    # Define the upper and lower colour thresholds for the ball colour.
    # lower_ball = np.array([70, 0, 95], dtype="uint8")  
    # upper_ball = np.array([180, 80, 255], dtype="uint8")   # peter_putting_second
    lower_ball = np.array([43, 0, 92], dtype="uint8")        
    upper_ball = np.array([62, 76, 255], dtype="uint8")     # peter_putting_third

    # Define the upper and lower HSV colour thresholds for the green (grass) colour.
    # lower_grass = np.array([50, 120, 70], dtype="uint8")  
    # upper_grass = np.array([110, 255, 255], dtype="uint8")   # peter_putting_second 
    lower_grass = np.array([20, 100, 30], dtype="uint8")  
    upper_grass = np.array([90, 255, 220], dtype="uint8")   # peter_putting_third

    while cv2.waitKey(100) < 0:   # the number determines how fast the video plays
    # while cv2.waitKey(0):
        # grab the current frame
        ret, frame = cap.read()  # Read a frame from the video file.
        if not ret:              # If we cannot read any more frames from the video file, then exit.
            break
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # convert image from BGR to HSV

        # Mask the frame using bitwise_and() operation with green grass so we only focus on the area with grass
        green_mask = cv2.inRange(frame_HSV, lower_grass, upper_grass)
        # Perfrom closing morphology (dilate then erosion) to fill gaps and holes in image
        kernel = np.ones((3,3), np.uint8)
        closing_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
        # do some erosion after this to get rid of random white spots
        # erosion_mask = cv2.erode(closing_mask, kernel, iterations=2)
        opening_mask = cv2.morphologyEx(closing_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        # Find contours in the binary image
        contours, _ = cv2.findContours(opening_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=len)
        # Approximate the contour with a polygon
        epsilon = 0.03 * cv2.arcLength(contour, True) # adjust the epsilon value as needed
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # Draw the polygon (optional)
        # cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)  # Green color, thickness=2
        # Get corner points
        # corners = np.squeeze(approx)
        # Draw circles at corner points (optional)
        # for corner in corners:
        #     cv2.circle(frame, tuple(corner), 5, (0, 0, 255), -1)  # Red color, filled circle
        # Create a black mask with the same dimensions as the image
        black_mask = np.zeros(frame_HSV.shape[:2], dtype="uint8")   # this black_mask works
        # Draw the polygon on the mask
        cv2.fillPoly(black_mask, [approx], 255)
        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(frame_HSV, frame_HSV, mask=black_mask)


        # now find the ball using the masked_image
        center, radius = find_ball(masked_image, lower_ball, upper_ball)    # using the find_ball function
        if center is not None:
            # Draw circle around the ball.
            cv2.circle(frame, tuple(center), radius,(0,0,255), 2)
            # Draw the center (not centroid!) of the ball.
            cv2.circle(frame, tuple(center), 1,(0,0,255), 2)
        cv2.imshow('frame', frame)  # Display the grayscale frame on the screen.


if __name__ == "__main__":
    track_ball()

    # while cv2.waitKey(0):  # this steps through each frame when you press 'q'
        


