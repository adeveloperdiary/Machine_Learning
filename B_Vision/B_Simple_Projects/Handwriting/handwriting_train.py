import numpy as np
import cv2
import argparse
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from B_Vision.B_Simple_Projects.Handwriting.hog import HOG
from B_Vision.B_Simple_Projects.Handwriting import data_loader

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the dataset file")
ap.add_argument("-m", "--model", required=True, help="Path where the model will be stored")

args = vars(ap.parse_args())

(digits, target) = data_loader.load_digits(args["dataset"])
data = []

hog = HOG(orientations=18, pixelPerCell=(10, 10), cellsPerBlock=(1, 1), transform=True)

for image in digits:
    image = data_loader.deskew(image, 20)
    image = data_loader.center_extent(image, (20, 20))

    hist = hog.describe(image)

    data.append(hist)

model = LinearSVC(random_state=42,max_iter=2000)
model.fit(data, target)

joblib.dump(model, args["model"])
