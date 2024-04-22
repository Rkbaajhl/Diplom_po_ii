import cv2
import numpy as np
import os

def convert_to_grayscale(input_image_path, output_image_path):
    # Check if the input image file exists
    if not os.path.isfile(input_image_path):
        print(f"Error: Input image file '{input_image_path}' does not exist.")
        return

    # Load the image
    image = cv2.imread(input_image_path)
    
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(grayscale_image, (7, 7), 0)

    # Apply Canny edge detection to detect edges
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge map
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the tuple to a list and sort it by area
    contours_list = list(contours)
    contours_list.sort(key=lambda x: cv2.contourArea(x), reverse=True)

    # Keep only the top 3 contours
    top_contours = contours_list[:3]

    # Extract the coordinates of the bounding boxes with an additional 7 mm margin
    boxes = []
    for contour in top_contours:
        # Get the bounding box of the current contour
        x, y, w, h = cv2.boundingRect(contour)

        # Expand the bounding box by 7 mm in each direction
        x -= 10
        y -= 10
        w += 20
        h += 20

        # Add the adjusted bounding box coordinates to the list
        boxes.append((x, y, w, h))

    # Load the image on which to draw the adjusted bounding boxes
    image_to_draw = image.copy()

    # Draw the adjusted bounding boxes on the image
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(image_to_draw, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Save the output image with adjusted bounding boxes
    cv2.imwrite(output_image_path, image_to_draw)

    # Определение центров bounding box
    centers = [(x + w // 2, y + h // 2) for x, y, w, h in boxes]

    # Вычисление длин отрезков от углов изображения до центров bounding box
    distances = []
    for center in centers:
        distances.append(np.sqrt((0 - center[0])**2 + (0 - center[1])**2)) # От левого верхнего угла
        distances.append(np.sqrt((0 - center[0])**2 + (image.shape[0] - center[1])**2)) # От левого нижнего угла
        distances.append(np.sqrt((image.shape[1] - center[0])**2 + (0 - center[1])**2)) # От правого верхнего угла

    # Находим индексы трех минимальных расстояний
    min_distances_indices = np.argsort(distances)[:3]

    # Обрезка и сохранение изображений
    for i, name in enumerate(['0.png', '1.png', '2.png']):
        if i < len(min_distances_indices) and min_distances_indices[i] < len(boxes):
            index = min_distances_indices[i]
            x, y, w, h = boxes[index]
            cropped_image = image[y:y+h, x:x+w]
            cv2.imwrite(f'C:\\Diplom_po_ii\\{name}', cropped_image)
        else:
            print("Not enough boxes to crop")

input_image_path = 'C:\\Diplom_po_ii\\d.png'
output_image_path = 'C:\\Diplom_po_ii\\dg.png'

convert_to_grayscale(input_image_path, output_image_path)