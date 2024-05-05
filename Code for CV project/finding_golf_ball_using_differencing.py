import cv2
import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints



# Functions

def difference(img0, img1):
    diff = cv2.subtract(img0, img1)  # calculate the difference of the two images
    
    # move the data in img0 to img1.
    img1 = img0
    ret, diff = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
    diff = cv2.medianBlur(diff, 3)
    diff = cv2.medianBlur(diff, 3)        # repeating medianBlur gets rid of salt and pepper noise
    gray_image = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Difference', diff)    # display the difference on the screen
    return gray_image, img1

def find_ball(diff, lower_ball, upper_ball):
    # Perform morphology
    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(diff, kernel, iterations=0)
    dilation = cv2.dilate(diff, kernel, iterations=0)
    opening_mask = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=0)
    closing_mask = cv2.morphologyEx(opening_mask, cv2.MORPH_CLOSE, kernel, iterations=0)
    final_mask = closing_mask

    cv2.imshow('mask', final_mask)

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
        
        # find coordinates of the best circle
        (x, y), radius = cv2.minEnclosingCircle(bestCircle)    
        center = np.array([int(x), int(y)])  
        radius = int(radius)  
        return center, radius
    else:
        return None, None

if __name__ == "__main__":
    cap = cv2.VideoCapture('videos/peter_putting_fifth_60fps.mp4')
    # cap = cv2.VideoCapture('videos/peter_putting_second2.mp4')

    lower_ball = np.array([50, 0, 130], dtype="uint8")  
    upper_ball = np.array([105, 100, 255], dtype="uint8") 

    # Load two initial images from the video to begin
    ret, img0 = cap.read()
    ret, img1 = cap.read()

    # while cv2.waitKey(100) < 0:
    while cv2.waitKey(0):
        ret, frame = cap.read()   # grab a new frame from the video 
        if not ret:               # If we cannot read any more frames from the video file, then exit.
            break
        img0 = frame              # grab a new frame from the video for img0
        # ret, img0 = cap.read()
        diff, img1 = difference(img0, img1)

        

        # now find the ball using the differenced image
        center, radius = find_ball(diff, lower_ball, upper_ball)    # using the find_ball function
        
        if center is not None:
            # Draw circle around the ball.
            cv2.circle(frame, tuple(center), radius,(0,0,255), 2)
            # Draw the center (not centroid!) of the ball.
            cv2.circle(frame, tuple(center), 1,(0,0,255), 2)
            print("Found")

        cv2.imshow('frame', frame)  # Display the annotated frame on the screen.

























