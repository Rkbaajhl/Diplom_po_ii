import cv2
import keras

CATEGORIES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]  # will use this to convert prediction num to string value


def prepare(filepath):
    IMG_SIZE = 28  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    cv2.imwrite("C:\\Diplom_po_ii\\trash111.png", new_array)  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.

model = keras.models.load_model("digits2.keras")
prediction = model.predict([prepare('C:\\Diplom_po_ii\\k4.png')]) 

print(prediction)