import cv2
import numpy as np

def detect_objects(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge map
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the tuple to a list and sort it by area
    contours_list = list(contours)
    contours_list.sort(key=lambda x: cv2.contourArea(x), reverse=True)

    # Keep only the top 3 contours
    top_contours = contours_list[:3]

    # Extract the coordinates of the bounding boxes
    boxes = []
    for contour in top_contours:
        # Get the bounding box of the current contour
        box = cv2.boundingRect(contour)

        # Add the bounding box coordinates to the list
        boxes.append(box)

    return boxes

# Load the image
image = cv2.imread('C:\Diplom_po_ii\ggd.png')

# Detect and extract the coordinates of the top 3 objects in the image
boxes = detect_objects(image)

# Load the image on which to draw the bounding boxes
image_to_draw = cv2.imread('C:\Diplom_po_ii\gg.png')

# Draw the bounding boxes on the image
for box in boxes:
    x, y, w, h = box
    cv2.rectangle(image_to_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Save the output image with bounding boxes
cv2.imwrite('C:\Diplom_po_ii\ggdb.png', image_to_draw)