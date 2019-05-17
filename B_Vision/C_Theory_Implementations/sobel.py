import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from B_Vision.C_Theory_Implementations.convolution import convolution
from Logging.Logging import info_log, error_log, warning_log, output_log


def sobel_edge_detection(image, filter, convert_to_degree=False, verbose=False):
    info_log("sobel_edge_detection()")

    new_image1 = convolution(image, filter, verbose)
    new_image2 = convolution(image, filter.T, verbose)

    gradient_magnitude = np.sqrt(np.square(new_image1) + np.square(new_image2))

    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    if verbose:
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("Output Image")
        plt.show()

    gradient_direction = np.arctan2(new_image2, new_image1)

    if convert_to_degree:
        gradient_direction = np.rad2deg(gradient_direction)

    # print(np.min(gradient_direction), np.max(gradient_direction))

    '''
    if verbose:
        plt.imshow(gradient_direction)
        plt.title("Output Image")
        plt.show()
    '''

    return gradient_magnitude, gradient_direction


if __name__ == '__main__':
    filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sobel_edge_detection(image, filter, True)
