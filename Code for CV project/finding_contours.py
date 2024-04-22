import cv2
import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints



# img_original = cv2.imread('images/putting.jpg')
# # Scale the image down to 70% to fit on the monitor better.
# img_original = cv2.resize(img_original, (int(img_original.shape[1]*0.7), int(img_original.shape[0]*0.7)))
# blur = cv2.GaussianBlur(img_original, (9,9), 0)
# # Convert the image to grayscale for processing
# gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)



def find_ball(mask):
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


if __name__ == "__main__":

    img_original = cv2.imread('putting.jpg')
    blur = cv2.GaussianBlur(img_original, (9,9), 0)
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)          # turn it gray
    # edges = cv2.Canny(gray, 100, 200)                       # get canny edges
    # cv2.imshow('Test', edges)                               # display the result

    # blob detection
    # create the params and deactivate the 3 default filters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = False
    params.filterByInertia = False
    params.filterByConvexity = False
    detector = cv2.SimpleBlobDetector_create(params)              # Create a SimpleBlobDetector object
    keypoints = detector.detect(gray)                       # Detect blobs in the image
    # Draw circles on the original image
    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        r = int(keypoint.size / 2)
        cv2.circle(img_original, (x, y), r, (0, 255, 0), 2)

    cv2.waitKey(0)
    # # Now we want to identify round objects, which should be the golf ball
    # center, radius = find_ball(edges)
    # if center is not None:
    #     # Draw circle around the ball.
    #     cv2.circle(img_original, tuple(center), radius,(0,255,0), 2)
    #     # Draw the center (not centroid!) of the ball.
    #     cv2.circle(img_original, tuple(center), 1,(0,255,0), 2)
    # # cv2.imshow('frame', img_original)  # Display the grayscale frame on the screen.
    cv2.imshow('detected circles', img_original)
    cv2.destroyAllWindows()

    # frame_HSV = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)   # convert image from BGR to HSV
    # # Define the upper and lower HSV colour thresholds for the ball colour.
    # lower = np.array([57, 0, 192], dtype="uint8")   
    # upper = np.array([96, 35, 255], dtype="uint8") 

    # mask = cv2.inRange(frame_HSV, lower, upper)
    # while cv2.waitKey(80) < 0:
        # mask_inverted = cv2.bitwise_not(mask)
    #     cv2.imshow('frame', mask)  # Display the grayscale frame on the screen
    # # This shows a black/white mask of the golf ball and any other objects that are within the white-grey colour range








