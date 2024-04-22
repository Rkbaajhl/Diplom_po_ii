import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from PIL import Image, ImageOps
import cv2
import numpy as np
from matplotlib import pyplot as plt

def sharpen_image(image_path):
    
    # Создание фильтра увеличения зерцалости
    sharpen_filter = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    
    # Применение фильтра к изображению
    sharpened_image = cv2.filter2D(image_path, -1, sharpen_filter)
    
    return sharpened_image

def preprocess_image(image):
    # Увеличение контрастности
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    # Применение сглаживания
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

    
    
    

# Загрузка и обучение классификатора
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
clf = svm.SVC(gamma=0.001)
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)
clf.fit(X_train, y_train)

# Загрузка пользовательского изображения
image_file = 'C:\\Diplom_po_ii\\k4.png' # Замените на путь к вашему изображению
img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE) # Загрузка изображения в оттенках серого
image = cv2.resize(img, None, fx=100, fy=100, interpolation=cv2.INTER_CUBIC)
_, binary_img = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)
# Уменьшение размера изображения до 8x8
resized_img = cv2.resize(binary_img, (8, 8), interpolation=cv2.INTER_AREA)


inverted_image = cv2.bitwise_not(resized_img)
# Преобразование изображения в формат, подходящий для классификатора
image_for_prediction = inverted_image.reshape(1, -1)

# Предсказание числа на изображении
predicted = clf.predict(image_for_prediction)

# Вывод результата
print(f"Предсказанное число: {predicted[0]}")

# Визуализация изображения
plt.imshow(inverted_image, cmap=plt.cm.gray_r, interpolation="nearest")
plt.title(f"Предсказанное число: {predicted[0]}")
plt.axis('off')
plt.show()