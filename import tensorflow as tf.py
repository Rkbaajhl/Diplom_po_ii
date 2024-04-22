import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Загрузка датасета MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализация изображений
x_train, x_test = x_train / 255.0, x_test / 255.0

# Добавление дополнительного измерения для каналов (Grayscale)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Создание модели
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
model.fit(x_train, y_train, epochs=5)

import numpy as np

def predict_number(image_path):
    # Загрузка и обработка изображения
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(28, 28), color_mode="grayscale")
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    # Предсказание числа
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Получение индекса максимального значения в предсказании
    predicted_number = np.argmax(score)

    return predicted_number

# Пример использования функции
image_path = 'C:\\Diplom_po_ii\\trash11.png'
predicted_number = predict_number(image_path)
print(f"Предсказанное число: {predicted_number}")