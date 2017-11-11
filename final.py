import numpy as np
import argparse
import cv2
import imutils
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
args = vars(ap.parse_args())

if not args.get("video", False):
    camera = cv2.VideoCapture(0)

else:
    camera = cv2.VideoCapture(args["video"])


def nothing(x):
    pass

cv2.namedWindow('image')

cv2.createTrackbar('lowH', 'image', 0, 255, nothing)
cv2.createTrackbar('uppH', 'image', 255, 255, nothing)
cv2.createTrackbar('lowS', 'image', 0, 255, nothing)
cv2.createTrackbar('uppS', 'image', 255, 255, nothing)
cv2.createTrackbar('lowV', 'image', 0, 255, nothing)
cv2.createTrackbar('uppV', 'image', 255, 255, nothing)
cv2.createButton('apply', nothing)

while True:
    (grabbed, frame) = camera.read()

    if args.get("video") and not grabbed:
        camera = cv2.VideoCapture(args["video"])
        continue

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv,
                       (
                           cv2.getTrackbarPos('lowH', 'image'),
                           cv2.getTrackbarPos('lowS', 'image'),
                           cv2.getTrackbarPos('lowV', 'image')
                       ),
                       (
                           cv2.getTrackbarPos('uppH', 'image'),
                           cv2.getTrackbarPos('uppS', 'image'),
                           cv2.getTrackbarPos('uppV', 'image')
                       )
                       )
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]

    for c in cnts:
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)


    frame = imutils.resize(frame, width=600)
    blurred = imutils.resize(blurred, width=600)
    mask = imutils.resize(mask, width=600)
    cv2.imshow("image", np.vstack([
        np.hstack([blurred, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), frame]),
        # np.hstack([blurred, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), frame])
    ]))

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
