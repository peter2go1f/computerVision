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
def find_ball(frame_HSV, lower_ball, upper_ball, frame_num2, prevCenter):
    # Detect a colour ball with a colour range.
    mask = cv2.inRange(frame_HSV, lower_ball, upper_ball)  # Find all pixels in the image within the colour range.

    # Perfrom morphology
    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=0)
    dilation = cv2.dilate(mask, kernel, iterations=0)
    opening_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closing_mask = cv2.morphologyEx(opening_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    final_mask = closing_mask

    cv2.imshow('mask', final_mask)
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

        # check if circle is still on golf ball
        if (frame_num2 == 0):
            prevCenter = center
            frame_num2 += 1
            return center, radius, frame_num2, prevCenter
        else:
            dist_diff = dist(center[0], center[1], prevCenter[0], prevCenter[1]) # check distance between prev_circle and current_circle
            if dist_diff < 150:    # ball jumps upto 120 between frames when moving
                frame_num2 += 1
                return center, radius, frame_num2, prevCenter
            frame_num2 += 1
            return None, None, frame_num2, prevCenter
        
        # return center, radius, frame_num2
    else:
        return None, None, frame_num2, prevCenter
    

def track_ball():
    cap = cv2.VideoCapture('videos/peter_putting_fifth_60fps.mp4')

    # Circle tracking based on previous circle location
    prevCircle = None
    dist = lambda x1, y1, x2, y2: (x2-x1)**2 + (y2-y1)**2

    # Define the upper and lower HSV colour thresholds for the green (grass) colour.
    lower_grass = np.array([25, 45, 0], dtype="uint8")  
    upper_grass = np.array([90, 255, 255], dtype="uint8")      # peter_putting_fifth_closeup_4k

    # Define the upper and lower colour thresholds for the ball colour.
    lower_ball = np.array([0, 0, 145], dtype="uint8")  
    upper_ball = np.array([120, 95, 255], dtype="uint8")      # peter_putting_fifth_closeup_4k

    # argparse setup
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
    args = vars(ap.parse_args())
    # Ball tracking contrail list of x,y points
    pts = deque(maxlen=args["buffer"])  # for ball tracking trail. Updated in find_ball()

    frame_num = 0    # for drawing ball contrail
    frame_num2 = 0   # for use in find_ball()
    prevCenter = None

    first_frame = True
    # while cv2.waitKey(100) < 0:   # the number determines how fast the video plays
    while cv2.waitKey(0):
        # grab the current frame
        ret, frame = cap.read()  # Read a frame from the video file.
        if not ret:              # If we cannot read any more frames from the video file, then exit.
            break
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # convert image from BGR to HSV

        if first_frame:   # since the green and camera are stationary, only need green detection boudnary for first frame
            first_frame = False
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
            epsilon = 0.003 * cv2.arcLength(contour, True) # adjust the epsilon value as needed
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # Draw the polygon (optional)
            # cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)  # Green color, thickness=2
            # # Get corner points
            # corners = np.squeeze(approx)
            # # Draw circles at corner points (optional)
            # for corner in corners:
            #     cv2.circle(frame, tuple(corner), 5, (0, 0, 255), -1)  # Red color, filled circle
            # Create a black mask with the same dimensions as the image
            black_mask = np.zeros(frame_HSV.shape[:2], dtype="uint8")   # this black_mask works
            # Draw the polygon on the mask
            cv2.fillPoly(black_mask, [approx], 255)
            # Apply the mask to the original image
            masked_image = cv2.bitwise_and(frame_HSV, frame_HSV, mask=black_mask)
            # masked_image = cv2.bitwise_and(frame, frame, mask=black_mask)
        else:
            # Create a black mask with the same dimensions as the image
            black_mask = np.zeros(frame_HSV.shape[:2], dtype="uint8")   # this black_mask works
            # Draw the polygon on the mask
            cv2.fillPoly(black_mask, [approx], 255)
            # Apply the mask to the original image
            masked_image = cv2.bitwise_and(frame_HSV, frame_HSV, mask=black_mask)
            # masked_image = cv2.bitwise_and(frame, frame, mask=black_mask)


        # now find the ball using the masked_image
        center, radius, frame_num2, prevCenter = find_ball(masked_image, lower_ball, upper_ball, frame_num2, prevCenter)    # using the find_ball function

        if center is not None:
            print(f"center: {center}")
            # ball tracking contrail
            if (frame_num == 0):
                prevCenter = center
            else:
                dist_diff = dist(center[0], center[1], prevCenter[0], prevCenter[1]) # check distance between prev_circle and current_circleq
                if dist_diff < 150:    # ball jumps upto 120 between frames when moving
                    pts.appendleft(center)
                prevCenter = center
            frame_num += 1

        if center is not None:
            # Draw circle around the ball.
            cv2.circle(frame, tuple(center), radius,(0,0,255), 2)
            # Draw the center (not centroid!) of the ball.
            cv2.circle(frame, tuple(center), 1,(0,0,255), 2)

            # draw contrail using pts deque
            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore them
                if pts[i-1] is None or pts[i] is None:
                    continue
                thickness = int(np.sqrt(args["buffer"] / float(i+1)) * 1.5)
                cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), thickness)

        cv2.imshow('mask', masked_image)
        cv2.imshow('frame', frame)  # Display the annotated frame on the screen.


# Main
if __name__ == "__main__":
    track_ball()

    # while cv2.waitKey(0):  # this steps through each frame when you press 'q'
        


