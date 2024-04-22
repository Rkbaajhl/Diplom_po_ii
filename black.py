import cv2

# Load the image
image = cv2.imread('C:\Diplom_po_ii\gg.png', cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection to detect edges
edges = cv2.Canny(image, 50, 150)

# Invert the edges to get the lines
lines = cv2.bitwise_not(edges)

# Save the output image with black lines
cv2.imwrite('C:\Diplom_po_ii\ggd.png', lines)