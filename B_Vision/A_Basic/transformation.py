import numpy as np
import cv2
import argparse

from B_Vision.utils.imutils import translate, rotate, resize

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args['image'])

cv2.imshow("Original", image)

shifted = translate(image, 0, 100)
cv2.imshow("Shifted Down", shifted)

shifted = rotate(image, 180)
cv2.imshow("Rotated by 180 Degrees", shifted)

shifted = resize(image, width=100)
cv2.imshow("Resized", shifted)

fliped = cv2.flip(image, 1)
cv2.imshow("Flipped", fliped)

# adding to all the channels
M = np.ones(image.shape, dtype="uint8") * 100
added = cv2.add(image, M)
cv2.imshow("Added", added)

sub = cv2.subtract(image, M)
cv2.imshow("Sub", sub)

cv2.waitKey(0)
