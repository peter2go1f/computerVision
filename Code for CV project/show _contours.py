import cv2
import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

# for ball tracking contrail
import argparse
from collections import deque
import imutils

# for ball tracking distance between previous frame
dist = lambda x1, y1, x2, y2: np.sqrt((x2-x1)**2 + (y2-y1)**2)

# Functions
def find_ball(frame, frame_HSV, lower_ball, upper_ball):

    # Detect a colour ball with a colour range.
    mask = cv2.inRange(frame_HSV, lower_ball, upper_ball)  # Find all pixels in the image within the colour range.

    # Perfrom closing morphology (dilate then erosion) to fill gaps and holes in image
    kernel = np.ones((3,3), np.uint8)
    
    # dilation = cv2.dilate(erosion, kernel, iterations=1)
    opening_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closing_mask = cv2.morphologyEx(opening_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    erosion = cv2.erode(closing_mask, kernel, iterations=1)
    

    final_mask = erosion
    
    cv2.imshow('mask', final_mask)
    # cv2.imshow('closing_mask', closing_mask)
    # cv2.imshow('opening_mask', opening_mask)

    # Find a series of points which outline the shape in the mask.
    contours, _hierarchy = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
            if (circularity > best_circularity):   # will always skip when i==0 as circularity=best_circularity
                best_circularity = circularity
                bestCircle = contour
            i += 1          
    
        # draw circle of the best circle that is also within a ball's circumference from last frame
        (x, y), radius = cv2.minEnclosingCircle(bestCircle)    
        center = np.array([int(x), int(y)])  
        radius = int(radius)  

        return contours, center, radius
    else:
        return None, None, None
    

def track_ball():
    cap = cv2.VideoCapture('videos/peter_putting_fifth_closeup_4k.mp4')

    # Define the upper and lower HSV colour thresholds for the green (grass) colour.
    # lower_grass = np.array([50, 120, 70], dtype="uint8")  
    # upper_grass = np.array([110, 255, 255], dtype="uint8")   # peter_putting_second 
    # lower_grass = np.array([20, 100, 30], dtype="uint8")  
    # upper_grass = np.array([90, 255, 220], dtype="uint8")   # peter_putting_third
    # lower_grass = np.array([33, 54, 21], dtype="uint8")  
    # upper_grass = np.array([86, 255, 255], dtype="uint8")      # peter_putting_fourth
    # lower_grass = np.array([30, 68, 75], dtype="uint8")  
    # upper_grass = np.array([91, 210, 220], dtype="uint8")      # peter_putting_fourth60fps
    lower_grass = np.array([25, 45, 0], dtype="uint8")  
    upper_grass = np.array([90, 255, 255], dtype="uint8")      # peter_putting_fifth_closeup

    # Define the upper and lower colour thresholds for the ball colour.
    # lower_ball = np.array([70, 0, 95], dtype="uint8")  
    # upper_ball = np.array([180, 80, 255], dtype="uint8")   # peter_putting_second
    # lower_ball = np.array([43, 0, 92], dtype="uint8")        
    # upper_ball = np.array([62, 76, 255], dtype="uint8")     # peter_putting_third
    # lower_ball = np.array([75, 0, 117], dtype="uint8")  
    # upper_ball = np.array([160, 114, 255], dtype="uint8")      # peter_putting_fourth
    # lower_ball = np.array([50, 0, 130], dtype="uint8")  
    # upper_ball = np.array([105, 100, 255], dtype="uint8")      # peter_putting_fourth60fps
    lower_ball = np.array([0, 0, 150], dtype="uint8")  
    upper_ball = np.array([120, 110, 255], dtype="uint8")      # peter_putting_fifth_closeup_4k

    # stats analysis
    ball_found = 0
    num_frames = 0

    # while cv2.waitKey(100) < 0:   # the number determines how fast the video plays
    while cv2.waitKey(0):
        
        # grab the current frame
        ret, frame = cap.read()  # Read a frame from the video file.
        if not ret:              # If we cannot read any more frames from the video file, then exit.
            break
        # blur image
        blur = cv2.GaussianBlur(frame, (3,3), 0)
        frame_HSV = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)   # convert image from BGR to HSV

        # Mask the frame using bitwise_and() operation with green grass so we only focus on the area with grass
        green_mask = cv2.inRange(frame_HSV, lower_grass, upper_grass)
        # Perfrom closing morphology (dilate then erosion) to fill gaps and holes in image
        kernel = np.ones((3,3), np.uint8)
        closing_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        # do some erosion after this to get rid of random white spots
        # erosion_mask = cv2.erode(closing_mask, kernel, iterations=2)
        opening_mask = cv2.morphologyEx(closing_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # Find contours in the binary image
        contours, _ = cv2.findContours(opening_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=len)
        # Approximate the contour with a polygon
        epsilon = 0.001 * cv2.arcLength(contour, True) # adjust the epsilon value as needed
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
        contours, center, radius = find_ball(frame, masked_image, lower_ball, upper_ball)    # using the find_ball function

        

        if contours is not None:
            cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)
            for contour in contours:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = np.array([int(x), int(y)])
                radius = int(radius)
                cv2.circle(frame, tuple(center), radius,(0,0,255), 2)
            # Draw circle around the ball.
            # if center is not None:
            #     cv2.circle(frame, tuple(center), radius,(0,0,255), 2)
            #     print(radius)
            #     if (7 < radius <= 13):
            #         ball_found += 1
        
        cv2.imshow('frame', frame)  # Display the annotated frame on the screen.
        num_frames += 1

    print(f"ball found = {ball_found}")
    print(f"number of frames = {num_frames}")
    print(f"success rate = {(ball_found/num_frames):.2f}%")

# Main
if __name__ == "__main__":
    track_ball()
    

    # while cv2.waitKey(0):  # this steps through each frame when you press 'q'
        


