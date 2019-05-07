import numpy as np
import cv2
from sklearn.externals import joblib
import argparse
from B_Vision.B_Simple_Projects.Handwriting.hog import HOG
from B_Vision.B_Simple_Projects.Handwriting import data_loader
import mahotas

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="Path where the model will be stored")
ap.add_argument("-i", "--image", required=True, help="Path to the image file")

args = vars(ap.parse_args())

model = joblib.load(args["model"])
hog = HOG(orientations=18, pixelPerCell=(10, 10), cellsPerBlock=(1, 1), transform=True)

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)

(_, contours, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key=lambda x: x[1])

for (c, _) in contours:
    (x, y, w, h) = cv2.boundingRect(c)

    if w >= 7 and h >= 20:
        roi = gray[y:y + h, x:x + w]

        threshold = roi.copy()

        T = mahotas.thresholding.otsu(roi)
        threshold[threshold > T] = 255

        threshold = cv2.bitwise_not(threshold)

        threshold = data_loader.deskew(threshold, 20)
        threshold = data_loader.center_extent(threshold, (20, 20))

        cv2.imshow("Threshold", threshold)

        hist = hog.describe(threshold)

        digit = model.predict([hist])[0]
        print("I think that number is: {}".format(digit))

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(image, str(digit), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imshow("image", image)
        cv2.waitKey(0)
