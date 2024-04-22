import cv2
import pytesseract

def recognize_number(image_path):
    # Чтение изображения
    img = cv2.imread(image_path)
    
    text = pytesseract.image_to_string(img, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    
    return text

# Пример использования функции
image_path = 'C:\\Diplom_po_ii\\trash11.png'
recognized_number = recognize_number(image_path)
print(f"Распознанное число: {recognized_number}")