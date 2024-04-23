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

def find_ball(mask, lower, frameupper):
    # Detect a colour ball with a colour range.
    # mask = cv2.inRange(img, lower, upper)  # Find all pixels in the image within the colour range.

    # Find a series of points which outline the shape in the mask.
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=len)  # Assume the contour with the most points is the ball.
        # Fit a circle to the points of the contour enclosing the ball.
        (x,y),radius = cv2.minEnclosingCircle(contour)
        center = np.array([int(x),int(y)])
        radius = int(radius)
        return center, radius
    else:
        return None, None

def track_ball():
    img_original = cv2.imread('images/putting.jpg')

    # Define the upper and lower colour thresholds for the ball colour.
    lower = np.array([30, 30, 90], dtype="uint8")
    upper = np.array([150, 87, 200], dtype="uint8")

    while cv2.waitKey(80) < 0:
        center, radius = find_ball(img_original, lower, upper)
        if center is not None:
            # Draw circle around the ball.
            cv2.circle(img_original, tuple(center), radius,(0,255,0), 2)
            # Draw the center (not centroid!) of the ball.
            cv2.circle(img_original, tuple(center), 1,(0,255,0), 2)
        cv2.imshow('frame', img_original)  # Display the grayscale frame on the screen.





#########################################################################################
""" Main """
#########################################################################################

if __name__ == "__main__":
    # track_ball()
    

    cap = cv2.VideoCapture('videos/peter_putting_second2.mp4')
    # grab the current frame
    ret, frame = cap.read()  # Read a frame from the video file.
    # if not ret:
    #     break     # this needs to go in a while loop
    # while cv2.waitKey(1) < 0:
        

    # mask = cv2.inRange(frame_HSV, lower, upper)

    # img_original = cv2.imread('putting.jpg')
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # convert image from BGR to HSV
    # Define the upper and lower HSV colour thresholds for the green (grass) colour.
    lower = np.array([50, 120, 70], dtype="uint8")  
    upper = np.array([110, 255, 255], dtype="uint8") 

    # Detect a colour ball with a colour range.
    mask = cv2.inRange(frame_HSV, lower, upper)  # Find all pixels in the image within the colour range.

    # Perfrom closing morphology (dilate then erosion) to fill gaps and holes in image
    kernel = np.ones((3,3), np.uint8)
    closing_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    # do some erosion after this to get rid of random white spots
    # erosion_mask = cv2.erode(closing_mask, kernel, iterations=2)
    opening_mask = cv2.morphologyEx(closing_mask, cv2.MORPH_OPEN, kernel, iterations=2)   # trying opening instead of just erosion

    # mask_inverted = cv2.bitwise_not(mask)
    # cv2.imshow('closing mask', closing_mask)  # Display the grayscale frame on the screen
    # This shows a black/white mask of the golf ball and any other objects that are within the white-grey colour range
    # cv2.imshow('opening mask', opening_mask)

    # Define regions of interest (for demonstration purposes, let's assume predefined ROIs)
    # roi for drawing small box around the golf ball
    # rois = [(200, 630, 100, 100)]  # Format: (start_x, start_y, width, height)
    rois = [(200, 630, 100, 100)]  # Format: (start_x, start_y, width, height)

    # Draw bounding boxes
    for roi in rois:
        start_x, start_y, width, height = roi
        end_x = start_x + width
        end_y = start_y + height
        # cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)  # Green color, thickness=2
    # cv2.imshow('frame', frame)   # show original img with bounding box

    # Find contours in the binary image
    contours, _ = cv2.findContours(opening_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through each contour
    for contour in contours:
        # Approximate the contour with a polygon
        epsilon = 0.03 * cv2.arcLength(contour, True) # adjust the epsilon value as needed
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Draw the polygon (optional)
        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)  # Green color, thickness=2
        
        # Get corner points
        corners = np.squeeze(approx)
        
        # Draw circles at corner points (optional)
        for corner in corners:
            cv2.circle(frame, tuple(corner), 5, (0, 0, 255), -1)  # Red color, filled circle

    
    # Create a black mask with the same dimensions as the image
    black_mask = np.zeros(frame.shape[:2], dtype="uint8")   # this code works
    black_mask1 = np.zeros_like(frame, dtype="uint8") # this code doesn't work - even tho its dtype=uint8 it thinks its int8
    print(frame.dtype)
    print(black_mask1.dtype)
    
    # Draw the polygon on the mask
    cv2.fillPoly(black_mask, [approx], 255)
    
    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(frame, frame, mask=black_mask)

    while cv2.waitKey(1) < 0:
        # Display the image with contours and corner points
        cv2.imshow('Contours and Corner Points', frame)
        cv2.imshow('masked_image', masked_image)

    cv2.destroyAllWindows()




