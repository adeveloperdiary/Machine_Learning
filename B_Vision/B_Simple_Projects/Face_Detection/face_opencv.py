import numpy as np
import cv2


class FaceDetector:
    def __init__(self, face_cascade_path):
        self.faceCascade = cv2.CascadeClassifier(face_cascade_path)

    def detect(self, image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        # Haar cascade classifiers
        rects = self.faceCascade.detectMultiScale(image, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size,
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
        return rects


class EyeTracker:
    def __init__(self, face_cascade_path, eye_cascade_path):
        self.faceCascade = cv2.CascadeClassifier(face_cascade_path)
        self.eyeCascade = cv2.CascadeClassifier(eye_cascade_path)

    def track(self, image):
        faceRects = self.faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                      flags=cv2.CASCADE_SCALE_IMAGE)
        rects = []

        for (fX, fY, fW, fH) in faceRects:
            faceROI = image[fY:fY + fH, fX:fX + fW]
            rects.append((fX, fY, fX + fW, fY + fH))

            eyeRects = self.eyeCascade.detectMultiScale(faceROI, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)

            for (eX, eY, eW, eH) in eyeRects:
                rects.append((fX + eX, fY + eY, fX + eX + eW, fY + eY + eH))

        return rects
