import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract

image_file = r"C:\\Diplom_po_ii\\gg_1.png"
img = cv2.imread(image_file)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thr = cv2.threshold(gray, 254, 255, cv2.THRESH_TOZERO_INV)
ret, thresh = cv2.threshold(thr, 90, 255, cv2.THRESH_BINARY_INV)
img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

# Get contours
contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

output = img.copy()

for idx, contour in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(contour)
    # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
    # hierarchy[i][0]: the index of the next contour of the same level
    # hierarchy[i][1]: the index of the previous contour of the same level
    # hierarchy[i][2]: the index of the first child
    # hierarchy[i][3]: the index of the parent
    if hierarchy[0][idx][3] == 0:
        cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)

image = thresh
#image = cv2.resize(new_image, None, fx=4, fy=2.5, interpolation=cv2.INTER_CUBIC)
string = pytesseract.image_to_string(image, lang='eng')
print(string)
# cv2.imshow("Input", img)
# cv2.imshow("Enlarged", img_erode)
#cv2.imshow("Output", output)
cv2.imwrite("C:\\Diplom_po_ii\\trash1.png", image)
#cv2.imshow("thresh", image)
#cv2.imwrite("C:\\Diplom_po_ii\\trash2.png", output)
# cv2.imshow("thr", thr)
