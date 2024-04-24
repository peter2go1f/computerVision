from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np

max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
def on_low_H_thresh_trackbar(val):
 global low_H
 global high_H
 low_H = val
 low_H = min(high_H-1, low_H)
 cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
 global low_H
 global high_H
 high_H = val
 high_H = max(high_H, low_H+1)
 cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
 global low_S
 global high_S
 low_S = val
 low_S = min(high_S-1, low_S)
 cv.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
 global low_S
 global high_S
 high_S = val
 high_S = max(high_S, low_S+1)
 cv.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
 global low_V
 global high_V
 low_V = val
 low_V = min(high_V-1, low_V)
 cv.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
 global low_V
 global high_V
 high_V = val
 high_V = max(high_V, low_V+1)
 cv.setTrackbarPos(high_V_name, window_detection_name, high_V)
parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
cap = cv.VideoCapture(args.camera)
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)

while True:
    cap = cv.VideoCapture('videos/peter_putting_fourth.mp4')
    ret, frame = cap.read()  # Read a frame from the video file.
    # frame = cv.imread('putting_face_on.png')
    if frame is None:
        break
    
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # green_masking to then test threshold for golf ball only
    # lower_grass = np.array([20, 100, 30], dtype="uint8")  
    # upper_grass = np.array([90, 255, 220], dtype="uint8")   # peter_putting_third
    lower_grass = np.array([33, 54, 21], dtype="uint8")  
    upper_grass = np.array([86, 255, 255], dtype="uint8")      # peter_putting_fourth
    # Mask the frame using bitwise_and() operation with green grass so we only focus on the area with grass
    green_mask = cv.inRange(frame_HSV, lower_grass, upper_grass)
    # Perfrom closing morphology (dilate then erosion) to fill gaps and holes in image
    kernel = np.ones((3,3), np.uint8)
    closing_mask = cv.morphologyEx(green_mask, cv.MORPH_CLOSE, kernel, iterations=5)
    # do some erosion after this to get rid of random white spots
    # erosion_mask = cv2.erode(closing_mask, kernel, iterations=2)
    opening_mask = cv.morphologyEx(closing_mask, cv.MORPH_OPEN, kernel, iterations=2)
    # Find contours in the binary image
    contours, _ = cv.findContours(opening_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=len)
    # Approximate the contour with a polygon
    epsilon = 0.03 * cv.arcLength(contour, True) # adjust the epsilon value as needed
    approx = cv.approxPolyDP(contour, epsilon, True)
    # Draw the polygon (optional)
    cv.drawContours(frame, [approx], 0, (0, 255, 0), 2)  # Green color, thickness=2
    # Get corner points
    corners = np.squeeze(approx)
    # Draw circles at corner points (optional)
    for corner in corners:
        cv.circle(frame, tuple(corner), 5, (0, 0, 255), -1)  # Red color, filled circle
    # Create a black mask with the same dimensions as the image
    black_mask = np.zeros(frame_HSV.shape[:2], dtype="uint8")   # this black_mask works
    # Draw the polygon on the mask
    cv.fillPoly(black_mask, [approx], 255)
    # Apply the mask to the original image
    masked_image = cv.bitwise_and(frame_HSV, frame_HSV, mask=black_mask)


    # thresholding for the green grass
    # frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    # thresholding for white ball after green_mask
    frame_threshold = cv.inRange(masked_image, (low_H, low_S, low_V), (high_H, high_S, high_V))       
            
    cv.imshow(window_capture_name, frame_HSV)
    cv.imshow(window_detection_name, frame_threshold)
    
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break