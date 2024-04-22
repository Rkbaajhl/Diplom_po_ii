import cv2

def convert_to_grayscale(input_image_path, output_image_path):
    # Загрузка изображения
    image = cv2.imread(input_image_path)
    
    # Преобразование в черно-белое изображение
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to detect edges
    edges = cv2.Canny(grayscale_image, 50, 150)

    # Invert the edges to get the lines
    lines = cv2.bitwise_not(edges)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(lines, (5, 5), 0)

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

    # Load the image on which to draw the bounding boxes
    image_to_draw = image.copy()

    # Draw the bounding boxes on the image
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(image_to_draw, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Save the output image with bounding boxes
    cv2.imwrite(output_image_path, image_to_draw)

input_image_path = 'C:\Diplom_po_ii\g.png'
output_image_path = 'C:\Diplom_po_ii\gg.png'

convert_to_grayscale(input_image_path, output_image_path)