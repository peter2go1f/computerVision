import numpy as np
import cv2 


# videoCap = cv2.VideoCapture(2)
prevCircle = None
dist = lambda x1, y1, x2, y2: (x2-x1)**2 + (y2-y1)**2

cap = cv2.VideoCapture('videos/peter_putting_fourth60fps.mp4')


# while True:
while cv2.waitKey(0):
    # ret, frame = videoCap.read()
    ret, frame = cap.read()
    if not ret:
        break

    lower_grass = np.array([30, 68, 75], dtype="uint8")  
    upper_grass = np.array([91, 210, 220], dtype="uint8")
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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
    # Create a black mask with the same dimensions as the image
    black_mask = np.zeros(frame_HSV.shape[:2], dtype="uint8")   # this black_mask works
    # Draw the polygon on the mask
    cv2.fillPoly(black_mask, [approx], 255)
    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(frame_HSV, frame_HSV, mask=black_mask)

    # grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayFrame = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    # blurFrame = cv2.GaussianBlur(grayFrame, (15,15), 0)
    
    circles = cv2.HoughCircles(grayFrame, cv2.HOUGH_GRADIENT, 1, 20, 
                              param1=235, param2=9, minRadius=5, maxRadius=9)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None:
                chosen = i
            if prevCircle is not None:
                if dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) <= dist(i[0], i[1], prevCircle[0], prevCircle[1]):
                    chosen = i
        cv2.circle(frame, (chosen[0], chosen[1]), 1, (0,100,100), 3)
        cv2.circle(frame, (chosen[0], chosen[1]), chosen[2], (255,0,255), 3)
        prevCircle = chosen
    
    cv2.imshow("circles", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    
videoCap.release()
cv2.destroyAllWindows()
    
    
    
    
    
    
    
    