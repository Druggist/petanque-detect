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


# def get_colors(event, x, y, flags, frame):
# 	global JACK_COLOR, BOULE_COLOR
# 	if event == cv2.EVENT_LBUTTONDBLCLK:
# 		JACK_COLOR = np.array(cv2.cvtColor(np.uint8([[frame[x,y]]]), cv2.COLOR_BGR2HSV)[0][0])
# 		print("j: {0}".format(JACK_COLOR))
# 	elif event == cv2.EVENT_RBUTTONDBLCLK: 
# 		BOULE_COLOR = np.array(cv2.cvtColor(np.uint8([[frame[x,y]]]), cv2.COLOR_BGR2HSV)[0][0])
# 		print("b: {0}".format(BOULE_COLOR))
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
	min_c = np.array([data[:,:,0].min(), data[:,:,1].min(), data[:,:,2].min()])
	max_c = np.array([data[:,:,0].max(), data[:,:,1].max(), data[:,:,2].max()])
	return [min_c, max_c]



def draw_marker(frame, x, y, radius):
	cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
	cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)


def draw_dist(frame, x, y, x2, y2):
	cv2.line(frame, (int(x), int(y)), (int(x2), int(y2)), (255, 0, 0), thickness=2)
	dist = math.hypot(x2 - x, y2 - y)
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(frame, str(round(dist,2)), (int((x + x2) / 2), int((y + y2) / 2)), font, 2, (255, 255, 255), 3, cv2.LINE_AA)
	cv2.text


def get_jack(frame):
	global JACK_COLOR, DEVIATION

	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, np.clip(JACK_COLOR[0] - DEVIATION, 0, 255), np.clip(JACK_COLOR[1] + DEVIATION, 0, 255))
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
							cv2.CHAIN_APPROX_SIMPLE)[-2]

	if len(cnts) > 0:
		c = cnts[0]
		return cv2.minEnclosingCircle(c)


def get_boules(frame):
	global BOULE_COLOR, DEVIATION

	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, np.clip(BOULE_COLOR[0] - DEVIATION, 0, 255), np.clip(BOULE_COLOR[1] + DEVIATION, 0, 255))
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
	global REF_PT, JACK_COLOR, BOULE_COLOR, CROPPING

	if not args.get("video", False):
		camera = cv2.VideoCapture(0)

	else:
		camera = cv2.VideoCapture(args["video"])

		(grabbed, frame) = (False, [])
		Config = False
	while True:
		if args.get("video") and not grabbed:
			camera = cv2.VideoCapture(args["video"])
			(grabbed, frame) = camera.read()
			continue
		
		frame_tmp = frame.copy()

		if not Config:
			((x, y), radius) = get_jack(frame_tmp)
			draw_marker(frame_tmp, x, y, radius)
			boules = get_boules(frame_tmp)
			if boules is not None:
				for b in boules:
					if b[1] > radius * 2:
						draw_marker(frame_tmp, b[0][0], b[0][1], b[1])
						draw_dist(frame_tmp, b[0][0], b[0][1], x, y)
			(grabbed, frame) = camera.read()
		elif (not CROPPING) and (len(REF_PT) == 2):
			frame = imutils.resize(frame, width=800)
			if len(JACK_COLOR) == 0:
				JACK_COLOR = get_colors(frame[REF_PT[0][1]:REF_PT[1][1], REF_PT[0][0]:REF_PT[1][0]])
				REF_PT = []
			else:
				BOULE_COLOR = get_colors(frame[REF_PT[0][1]:REF_PT[1][1], REF_PT[0][0]:REF_PT[1][0]])
				Config = False
				REF_PT = []

		frame_tmp = imutils.resize(frame_tmp, width=800)
		cv2.setMouseCallback("frame", select_area)
		if len(REF_PT) == 2:
			cv2.rectangle(frame_tmp, REF_PT[0], REF_PT[1], (0, 255, 0), 2)
		cv2.imshow("frame", frame_tmp)

	
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
		elif key == ord("c") or args["config"]:
			args["config"] = False
			Config = True
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
