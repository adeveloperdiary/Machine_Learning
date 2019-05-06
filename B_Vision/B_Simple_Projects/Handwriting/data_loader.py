import numpy as np
import cv2
from B_Vision.utils import imutils
import mahotas


def load_digits(datasetPath):
    data = np.genfromtxt(datasetPath, delimiter=",", dtype="uint8")
    target = data[:, 0]
    data = data[:, 1:].reshape(data.shape[0], 28, 28)

    return data, target
