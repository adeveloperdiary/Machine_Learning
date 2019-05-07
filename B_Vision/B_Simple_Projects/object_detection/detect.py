from B_Vision.B_Simple_Projects.object_detection.object_descriptor import ObjectDescriptor
from B_Vision.B_Simple_Projects.object_detection.similarity import Similarity
import argparse
import glob
import csv
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True, help="Path to the book database")
ap.add_argument("-c", "--covers", required=True, help="Path to the book covers")
ap.add_argument("-q", "--query", required=True, help="Path to the query book cover")
ap.add_argument("-s", "--sift", type=int, default=0, help="whether or not SIFT should be used")

args = vars(ap.parse_args())

db = {}

for l in csv.reader(open(args["db"])):
    db[l[0]] = l[1:]

useSIFT = args["sift"] > 0
useHamming = args["sift"] == 0
ratio = 0.7
minMatches = 40

if useSIFT:
    minMatches = 50

cd = ObjectDescriptor(useSIFT=useSIFT)
cv = Similarity(cd, glob.glob(args["covers"] + "/*.png"), ratio=ratio, miniMatches=minMatches, useHamming=useHamming)

queryPaths = sorted(glob.glob(args["query"] + "/*.png"))

for queryPath in queryPaths:
    queryImage = cv2.imread(queryPath)
    gray = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)

    (queryKps, queryDesc) = cd.describe(gray)

    results = cv.find_match(queryKps, queryDesc)

    cv2.imshow("Query", queryImage)

    if len(results) == 0:
        print("No match found !")
        cv2.waitKey(0)

    else:
        for (i, (score, path)) in enumerate(results):
            (author, title) = db[path[path.rfind("/") + 1:]]
            print("{}. {:.2f}% : {} - {}".format(i + 1, score * 100, author, title))

            result = cv2.imread(path)
            cv2.imshow("Result", result)
            cv2.waitKey(0)
