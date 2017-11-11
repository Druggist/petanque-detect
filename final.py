import argparse
import numpy as np
import cv2
import imutils


def draw_marker(frame, x, y, radius):
    cv2.circle(frame, (int(x), int(y)), int(radius),
               (0, 255, 255), 2)
    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)


def get_jack(frame):
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (1, 220, 83), (10, 255, 170))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(cnts) > 0:
        c = cnts[0]
        return cv2.minEnclosingCircle(c)


def get_boules(frame):
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (1, 0, 30), (80, 255, 100))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]

    mask = imutils.resize(mask, width=600)
    cv2.imshow("mask", mask)
    if len(cnts) < 10:
        boules = []
        for c in cnts:
            boules.append(cv2.minEnclosingCircle(c))
        return boules


def main(args):
    if not args.get("video", False):
        camera = cv2.VideoCapture(0)

    else:
        camera = cv2.VideoCapture(args["video"])

    while True:
        (grabbed, frame) = camera.read()
        if args.get("video") and not grabbed:
            camera = cv2.VideoCapture(args["video"])
            continue

        ((x, y), radius) = get_jack(frame)
        draw_marker(frame, x, y, radius)
        boules = get_boules(frame)
        if boules is not None:
            for b in boules:
                if b[1] > radius * 2:
                    draw_marker(frame, b[0][0], b[0][1], b[1])

        frame = imutils.resize(frame, width=600)
        cv2.imshow("frame", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    args = vars(ap.parse_args())

    #
    # def nothing(x):
    #     pass
    #

    # cv2.namedWindow('image')
    #
    # cv2.createTrackbar('lowH', 'image', 0, 255, nothing)
    # cv2.createTrackbar('uppH', 'image', 255, 255, nothing)
    # cv2.createTrackbar('lowS', 'image', 0, 255, nothing)
    # cv2.createTrackbar('uppS', 'image', 255, 255, nothing)
    # cv2.createTrackbar('lowV', 'image', 0, 255, nothing)
    # cv2.createTrackbar('uppV', 'image', 255, 255, nothing)

    main(args)
