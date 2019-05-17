import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from B_Vision.C_Theory_Implementations.convolution import convolution


def sobel_edge_detection(image, filter, verbose=False):
    new_image1 = convolution(image, filter, verbose)
    new_image2 = convolution(image, filter.T, verbose)

    gradient_magnitude = np.sqrt(np.square(new_image1) + np.square(new_image2))

    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    if verbose:
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("Output Image")
        plt.show()

    gradient_direction = np.rad2deg(np.arctan(new_image2 / (new_image1 + 1e-8)))

    if verbose:
        plt.imshow(gradient_direction)
        plt.title("Output Image")
        plt.show()


if __name__ == '__main__':
    filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sobel_edge_detection(image, filter, False)
