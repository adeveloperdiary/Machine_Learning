import numpy as np
import cv2
import argparse

from B_Vision.C_Theory_Implementations.sobel import sobel_edge_detection
from B_Vision.C_Theory_Implementations.gaussian_blur import gaussian_blur
from Logging.Logging import info_log
import matplotlib.pyplot as plt


def non_max_suppression(gradient_magnitude, gradient_direction, verbose):
    info_log("non_max_suppression()")

    image_row, image_col = gradient_magnitude.shape

    gradient_direction += 180

    output = np.zeros(gradient_magnitude.shape)

    PI = 180

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]

            # (0 - PI/8 and 15PI/8 - 2PI)
            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]

            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]

            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]

            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]

            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Non Max Suppression")
        plt.show()

    return output


def threshold(image, low, high, weak, verbose=False):
    info_log("threshold()")

    output = np.zeros(image.shape)

    strong = 255

    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))

    output[strong_row, strong_col] = strong
    output[weak_row, weak_col] = weak

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("threshold")
        plt.show()

    return output


def hysteresis(image, weak):
    info_log("hysteresis")
    image_row, image_col = image.shape

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            if image[row, col] == weak:
                if image[row, col + 1] == 255 or image[row, col - 1] == 255 or image[row - 1, col] == 255 or image[row + 1, col] == 255 or image[
                    row - 1, col - 1] == 255 or image[row + 1, col - 1] == 255 or image[row - 1, col + 1] == 255 or image[row + 1, col + 1] == 255:
                    image[row, col] = 255
                else:
                    image[row, col] = 0

    return image


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-v", "--verbose", type=bool, default=False, help="Path to the image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])

    blurred_image = gaussian_blur(image, kernel_size=5, verbose=True)

    edge_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).T

    gradient_magnitude, gradient_direction = sobel_edge_detection(blurred_image, edge_filter, convert_to_degree=True, verbose=args["verbose"])

    new_image = non_max_suppression(gradient_magnitude, gradient_direction, verbose=args["verbose"])

    weak = 50

    new_image = threshold(new_image, 2, 20, weak=weak, verbose=args["verbose"])

    new_image = hysteresis(new_image, weak)

    plt.imshow(new_image, cmap='gray')
    plt.title("Canny Edge Detector")
    plt.show()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    new_image = cv2.Canny(image, 100, 220)
    plt.imshow(new_image, cmap='gray')
    plt.title("Canny Edge Detector - cv2")
    plt.show()
