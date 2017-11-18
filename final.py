import argparse
import math
import numpy as np
import cv2
import imutils

JACK_COLOR = [np.array([1, 220, 83]), np.array([10, 255, 170])]
BOULE_COLOR = [np.array([1, 0, 30]), np.array([80, 255, 100])]
DEVIATION = np.array([0, 0, 0])
REF_PT = []
CROPPING = False


def select_area(event, x, y, flags, param):
    global REF_PT, CROPPING

    if event == cv2.EVENT_LBUTTONDOWN:
        REF_PT = [(x, y), (x, y)]
        CROPPING = True

    elif CROPPING and event == cv2.EVENT_MOUSEMOVE:
        REF_PT[1] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        REF_PT[1] = (x, y)
        CROPPING = False


def get_colors(data):
    data = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)
    min_c = np.array([np.percentile(data[:, :, 0], 1), np.percentile(data[:, :, 1], 1), np.percentile(data[:, :, 2], 1)])
    max_c = np.array([np.percentile(data[:, :, 0], 99), np.percentile(data[:, :, 1], 99), np.percentile(data[:, :, 2], 99)])
    print([min_c, max_c])
    return [min_c, max_c]


def draw_marker(frame, x, y, radius):
    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), 2)


def draw_dist(frame, x, y, x2, y2):
    cv2.line(frame, (int(x), int(y)), (int(x2), int(y2)), (255, 0, 0), thickness=2)
    dist = math.hypot(x2 - x, y2 - y)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, str(round(dist, 1)), (int((x + x2) / 2), int((y + y2) / 2)), font, 0.8, (255, 255, 255), 1,
                cv2.LINE_AA)


def get_jack(frame):
    global JACK_COLOR, DEVIATION

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.clip(JACK_COLOR[0] - DEVIATION, 0, 255), np.clip(JACK_COLOR[1] + DEVIATION, 0, 255))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]

    mask = imutils.resize(mask, width=400)
    cv2.imshow("jack", mask)

    score = 0
    jack = None
    for c in cnts:
        if c.size > 10:
            ellipse = cv2.fitEllipse(c)
            if min(ellipse[1]) / max(ellipse[1]) > score:
                score = min(ellipse[1]) / max(ellipse[1])
                jack = cv2.minEnclosingCircle(c)
    return jack


def get_boules(frame):
    global BOULE_COLOR, DEVIATION

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.clip(BOULE_COLOR[0] - DEVIATION, 0, 255), np.clip(BOULE_COLOR[1] + DEVIATION, 0, 255))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]

    mask = imutils.resize(mask, width=400)
    cv2.imshow("boules", mask)
    boules = []
    for c in cnts:
        if c.size > 10:
            ellipse = cv2.fitEllipse(c)
            if min(ellipse[1]) / max(ellipse[1]) > 0.7:
                boules.append(cv2.minEnclosingCircle(c))
    return boules


def main(args):
    global REF_PT, JACK_COLOR, BOULE_COLOR, CROPPING

    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args["video"])

    if args["config"]:
        state = 0  # 0 - config, 1 - work
    else:
        state = 1

    (grabbed, frame) = camera.read()
    while True:
        frame = imutils.resize(frame, width=800)
        frame_tmp = frame.copy()
        if state == 0:
            cv2.setMouseCallback("frame", select_area)
            if (not CROPPING) and (len(REF_PT) == 2):
                blurred = cv2.GaussianBlur(frame, (3, 3), 0)
                if len(JACK_COLOR) == 0:
                    JACK_COLOR = get_colors(blurred[REF_PT[0][1]:REF_PT[1][1], REF_PT[0][0]:REF_PT[1][0]])
                    REF_PT = []
                elif len(BOULE_COLOR) == 0:
                    BOULE_COLOR = get_colors(blurred[REF_PT[0][1]:REF_PT[1][1], REF_PT[0][0]:REF_PT[1][0]])
                    REF_PT = []
                    state = 1
            if len(REF_PT) == 2:
                cv2.rectangle(frame_tmp, REF_PT[0], REF_PT[1], (0, 255, 0), 1)

        else:
            (grabbed, frame) = camera.read()
            if args.get("video") and not grabbed:
                camera = cv2.VideoCapture(args["video"])
                (grabbed, frame) = camera.read()
            frame = imutils.resize(frame, width=800)

            jack = get_jack(frame)
            if jack is not None:
                ((x, y), radius) = jack
                if radius > 10:
                    draw_marker(frame_tmp, x, y, radius)
                    boules = get_boules(frame_tmp)
                    if boules is not None:
                        for b in boules:
                            if radius * 3 > b[1] > radius * 2:
                                draw_marker(frame_tmp, b[0][0], b[0][1], b[1])
                                draw_dist(frame_tmp, b[0][0], b[0][1], x, y)

        cv2.imshow("frame", frame_tmp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            state = 0
            JACK_COLOR = []
            BOULE_COLOR = []

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("-v", "--video", help="path to the (optional) video file")
    ap.add_argument("-c", "--config", action='store_true', help="enter color configuration on startup")
    args = vars(ap.parse_args())
    main(args)
