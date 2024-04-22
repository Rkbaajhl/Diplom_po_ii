import cv2
import numpy as np

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

    centers = []
    for box in boxes:
        x, y, w, h = box
        center_x = x + w // 2
        center_y = y + h // 2
        centers.append((center_x, center_y))

    # Определение самых левых верхних, нижних и правых bounding box
    left_top_boxes = [box for box in boxes if box[0] == min(x[0] for x in boxes) and box[1] == min(x[1] for x in boxes)]
    left_bottom_boxes = [box for box in boxes if box[0] == min(x[0] for x in boxes) and box[1] == max(x[1] for x in boxes)]
    right_top_boxes = [box for box in boxes if box[0] == max(x[0] for x in boxes) and box[1] == min(x[1] for x in boxes)]

    # Обрезка и сохранение изображений
    for box, name in zip([left_top_boxes[0], left_bottom_boxes[0], right_top_boxes[0]], ['0.png', '1.png', '2.png']):
        x, y, w, h = box
        cropped_image = image[y:y+h, x:x+w]
        cv2.imwrite(f'C:\\Diplom_po_ii\\{name}', cropped_image)

# Ваш код для вызова функции
input_image_path = 'C:\\Diplom_po_ii\\e1.png'
output_image_path = 'C:\\Diplom_po_ii\\e1g.png'

convert_to_grayscale(input_image_path, output_image_path)

