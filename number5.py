from PIL import Image
import numpy as np
import cv2
import pytesseract

image = cv2.imread('C:\\Diplom_po_ii\\123.png')
imagen = cv2.resize(image, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)

# Отображение изображения с обведенными числами
cv2.imwrite("C:\\Diplom_po_ii\\trash1.png", imagen)

def process_image(image_path):
    # Открытие изображения
    image = Image.open(image_path).convert('L') # Преобразование в черно-белое
    image_array = np.array(image)
    
    # Применение порогового значения для создания черно-белого изображения
    # Значения выше порога становятся белыми, ниже - черными
    threshold = 200 # Пороговое значение можно настроить
    processed_image_array = np.where(image_array > threshold, 255, 0)
    
    # Создание нового изображения из массива
    processed_image = Image.fromarray(processed_image_array.astype(np.uint8))
    
    # Сохранение обработанного изображения
    processed_image.save('C:\\Diplom_po_ii\\trash11.png')

def recognize_number(image_path):
    # Чтение изображения
    img = cv2.imread(image_path)
    
    text = pytesseract.image_to_string(img, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    
    return text


image_path = 'C:\\Diplom_po_ii\\trash1.png'
process_image(image_path)

recognized_number = recognize_number('C:\\Diplom_po_ii\\trash11.png')
print(f"Распознанное число: {recognized_number}")
