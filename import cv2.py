import cv2

# Загрузка изображения
image = cv2.imread('C:\\Diplom_po_ii\\123.png')
imagen = cv2.resize(image, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)

# Отображение изображения с обведенными числами
cv2.imwrite("C:\\Diplom_po_ii\\trash1.png", imagen)

