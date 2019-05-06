import numpy as np
import cv2
import argparse
import cv2
from B_Vision.B_Simple_Projects.Face_Detection.face_opencv import FaceDetector, EyeTracker
from B_Vision.utils.imutils import resize
import time

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True, help="Path to the face cascade")
ap.add_argument("-i", "--image", required=False, help="Path to the image")
ap.add_argument("-v", "--video", required=False, help="Path to the Video")
ap.add_argument("-t", "--tracking", type=bool, required=False, help="Tracking Object by Color")
ap.add_argument("-e", "--eye", required=False, help="Path to the eye cascade")

# Detect face in image
# Detect face in video
# Detect face in live video
# Track object by color
# Detect Eye


args = vars(ap.parse_args())
# Needed only for tracking
blueLower = np.array([100, 67, 0], dtype="uint8")
blueUpper = np.array([255, 128, 50], dtype="uint8")

if args['image'] is not None:
    image = cv2.imread(args['image'])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd = FaceDetector(args["face"])
    faceRects = fd.detect(gray, scale_factor=1.2, min_neighbors=3, min_size=(10, 10))
    print("I found {} faces(s)".format(len(faceRects)))

    for (x, y, w, h) in faceRects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Faces", image)
else:
    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args["video"])

    while True:
        (grabbed, frame) = camera.read()

        if args.get("video") and not grabbed:
            break

        if args['tracking'] is None or args['tracking'] is False:
            frame = resize(frame, width=300)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if args['eye'] is None:
                detector = FaceDetector(args["face"])
                faceRects = detector.detect(gray, scale_factor=1.1, min_neighbors=3, min_size=(30, 30))
                frameclone = frame.copy()
                for (x, y, w, h) in faceRects:
                    cv2.rectangle(frameclone, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("Face", frameclone)
            else:
                detector = EyeTracker(args["face"], args["eye"])
                rects = detector.track(gray)
                frameclone = frame.copy()
                for rect in rects:
                    cv2.rectangle(frameclone, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
                cv2.imshow("Eye Tracking", frameclone)
        else:
            blue = cv2.inRange(frame, blueLower, blueUpper)
            blue = cv2.GaussianBlur(blue, (3, 3), 0)

            (_, contours, _) = cv2.findContours(blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
                rect = np.int32(cv2.boxPoints(cv2.minAreaRect(contour)))
                cv2.drawContours(frame, [rect], -1, (0, 255, 0), 2)

            cv2.imshow("Tracking", frame)
            cv2.imshow("Binary", blue)

            # time.sleep(0.025)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()
