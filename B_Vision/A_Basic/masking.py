import numpy as np
import cv2
import argparse

canvas1 = np.zeros((300, 300), dtype="uint8")
cv2.rectangle(canvas1, (25, 25), (275, 275), 255, -1)
cv2.imshow("Rect", canvas1)

canvas2 = np.zeros((300, 300), dtype="uint8")
cv2.circle(canvas2, (150, 150), 150, 255, -1)
cv2.imshow("circle", canvas2)

b_and = cv2.bitwise_and(canvas1, canvas2)
cv2.imshow("AND", b_and)

cv2.waitKey(0)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args['image'])

cv2.imshow("Original", image)

mask = np.zeros(image.shape[:2], dtype="uint8")
(cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
cv2.rectangle(mask, (cX - 25, cY - 25), (cX + 25, cY + 25), 255, -1)
cv2.imshow("Mask", mask)

masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Masked Image", masked)
cv2.waitKey(0)
