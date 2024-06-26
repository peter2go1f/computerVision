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

    cv2.imshow('Hough Circle Transform', img)

def find_ball(mask, lower, upper):
    # Detect a colour ball with a colour range.
    # mask = cv2.inRange(img, lower, upper)  # Find all pixels in the image within the colour range.

    # Find a series of points which outline the shape in the mask.
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_circle = contour = max(contours, key=len)
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

    (x, y), radius = cv2.minEnclosingCircle(bestCircle)
    center = np.array([int(x),int(y)])
    radius = int(radius)
    return center, radius

    #     if 0.8 <= circularity <= 1.2:
    #         # Draw contours and enclosing circles
    #         cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    #         cv2.circle(image, (int(x), int(y)), int(radius), (0, 0, 255), 2)

    # if contours:
    #     contour = max(contours, key=len)  # Assume the contour with the most points is the ball.
    #     # Fit a circle to the points of the contour enclosing the ball.
    #     (x,y),radius = cv2.minEnclosingCircle(contour)
    #     center = np.array([int(x),int(y)])
    #     radius = int(radius)
    #     return center, radius
    # else:
    #     return None, None

def track_ball():
    img_original = cv2.imread('images/putting.jpg')

    # Define the upper and lower colour thresholds for the ball colour.
    lower_ball = np.array([0, 0, 145], dtype="uint8")  
    upper_ball = np.array([120, 95, 255], dtype="uint8")

    while cv2.waitKey(80) < 0:
        center, radius = find_ball(img_original, lower_ball, upper_ball)
        if center is not None:
            # Draw circle around the ball.
            cv2.circle(img_original, tuple(center), radius,(0,255,0), 2)
            # Draw the center (not centroid!) of the ball.
            cv2.circle(img_original, tuple(center), 1,(0,255,0), 2)
        cv2.imshow('frame', img_original)  # Display the grayscale frame on the screen.


if __name__ == "__main__":
    # track_ball()
    
    cap = cv2.VideoCapture('videos/peter_putting_fifth_60fps.mp4')
    
    # mask = cv2.inRange(frame_HSV, lower, upper)
    while cv2.waitKey(0):    # cv2.waitKey(0) < 0:
        # grab the current frame
        ret, frame = cap.read()  # Read a frame from the video file.
        if not ret:
            break

        ret, img_original = cap.read()  # Read a frame from the video file.

        # img_original = cv2.imread('putting.jpg')
        # frame_HSV = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)   # convert image from BGR to HSV
        # Define the upper and lower HSV colour thresholds for the ball colour.
        # for peter_putting1-4
        # lower = np.array([57, 0, 192], dtype="uint8")  
        # upper = np.array([96, 35, 255], dtype="uint8") 
        # for peter_putting_second1-3
        lower = np.array([70, 0, 95], dtype="uint8")  
        upper = np.array([180, 80, 255], dtype="uint8") 

        # mask_inverted = cv2.bitwise_not(mask)
        # cv2.imshow('frame', mask)  # Display the grayscale frame on the screen
    # This shows a black/white mask of the golf ball and any other objects that are within the white-grey colour range

        # Now we want to identify round objects, which should be the golf ball
        # center, radius = find_ball(mask, lower, upper)  # uses max of contours
        hough_circle(img_original)
        # if center is not None:
        #     # Draw circle around the ball.
        #     cv2.circle(img_original, tuple(center), radius,(0,255,0), 2)
        #     # Draw the center (not centroid!) of the ball.
        #     cv2.circle(img_original, tuple(center), 1,(0,255,0), 2)
        # cv2.imshow('frame', img_original)  # Display the grayscale frame on the screen.






