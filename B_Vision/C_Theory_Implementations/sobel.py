import numpy as np
import cv2
import argparse
from Logging.Logging import info_log, error_log, warning_log, output_log
import matplotlib.pyplot as plt


def convolution(image, kernel, verbose=False):
    if len(image.shape) == 3:
        warning_log("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        info_log("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        info_log("Image Shape : {}".format(image.shape))

    info_log("Kernel Shape : {}".format(kernel.shape))

    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + 2 * pad_height, image_col + 2 * pad_width))

    padded_image[pad_height:image_row + 1, pad_width:image_col + 1] = image

    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])

    output_log("Output Image size : {}".format(output.shape))

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image")
        plt.show()

    return output


h_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

new_image = convolution(image, h_filter,verbose=True)
