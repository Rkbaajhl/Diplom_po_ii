import cv2

def convert_to_grayscale(input_image_path, output_image_path):
    # Загрузка изображения
    image = cv2.imread(input_image_path)
    
    # Преобразование в черно-белое изображение
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Сохранение черно-белого изображения
    cv2.imwrite(output_image_path, grayscale_image)

# Пример использования
input_image_path = 'C:\Diplom_po_ii\g.png'
output_image_path = 'C:\Diplom_po_ii\gg.png'

convert_to_grayscale(input_image_path, output_image_path)