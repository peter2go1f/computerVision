










# Define regions of interest (for demonstration purposes, let's assume predefined ROIs)
rois = [(start_x, start_y, width, height), ...]  # Format: (start_x, start_y, width, height)

# Draw bounding boxes
for roi in rois:
    start_x, start_y, width, height = roi
    end_x = start_x + width
    end_y = start_y + height
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)  # Green color, thickness=2