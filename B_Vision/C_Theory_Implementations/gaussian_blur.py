import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import math
from B_Vision.C_Theory_Implementations.sobel import convolution
from Logging.Logging import info_log, error_log, warning_log, output_log


def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_kernel(size, sigma=1, verbose=False):
    info_log("gaussian_kernel()")

    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.max()

    if verbose:
        plt.imshow(kernel_2D, interpolation='none')
        plt.title("Image")
        plt.show()

    return kernel_2D


def gaussian_blur(image, kernel_size, verbose=False):
    info_log("gaussian_blur()")
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size), verbose=verbose)
    return convolution(image, kernel, average=True, verbose=verbose)


if __name__ == '__main__':
    filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gaussian_blur(image, 5, verbose=True)
