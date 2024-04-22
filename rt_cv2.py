import cv2
import keras
import numpy as np


model = keras.models.load_model("digits3.keras")

img = cv2.imread("C:\\Diplom_po_ii\\trash111.png", cv2.IMREAD_GRAYSCALE)
gray = 255 - img
  
gray = cv2.resize(gray, (28, 28))
cv2.imwrite('gray'+ "C:\\Diplom_po_ii\\trash111.png", gray)
img = gray / 255.0
img = np.array(img).reshape(-1, 28, 28, 1)
out = str(np.argmax(model.predict(img)))
print (out)