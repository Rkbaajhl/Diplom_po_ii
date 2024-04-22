import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the digits dataset from sklearn
digits = datasets.load_digits()

# Preprocess the images in the dataset
images = digits.images.reshape((len(digits.images), -1))
images = images / 16.0

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, digits.target, test_size=0.2, random_state=42)

# Train a linear support vector machine (SVM) classifier on the training set
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X_train, y_train)

# Define a function to recognize digits in an image
def recognize_digits(image_path):
    # Load the image from file
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Preprocess the image
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (16, 16))
    img = img.flatten()
    img = img / 255.0

    # Recognize the digits using the trained SVM classifier
    y_pred = clf.predict(np.array([img]))

    # Return the recognized digits
    return y_pred[0]

# Example usage
image_path = 'C:\\Diplom_po_ii\\k4.png'
digits = recognize_digits(image_path)
print('Recognized digits:', digits)



