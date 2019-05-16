import numpy as np
import cv2
import argparse

if __name__ == '__main__':
    filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
