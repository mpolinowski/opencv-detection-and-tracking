import cv2
import sys
from random import randint

# select tracking algorithm
# tracker = cv2.legacy.TrackerCSRT_create()
# you can start with the faster KCF algorithm
# if track fails automatically switch to CSRT (see below)
tracker = cv2.legacy.TrackerKCF_create()
# tracker = cv2.legacy.TrackerMOSSE_create()
# load video from file
video = cv2.VideoCapture('resources/group_of_people_06.mp4')
if not video.isOpened():
    print('[ERROR] loading video')
    sys.exit()
# get first frame of video
ok, frame = video.read()
if not ok:
    print('[ERROR] loading frame')
    sys.exit()
# select object detection cascade
detector = cv2.CascadeClassifier('cascades/haarcascade_fullbody.xml')


# grab frame from video and perform object detection
def detect():
    while True:
        ok, frame = video.read()
        if not ok:
            print('[ERROR] frame could not be read OR end of file')
            sys.exit()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Adjust minSize according to your video resolution
        detections = detector.detectMultiScale(frame_gray, minSize=(300, 300))
        print('[INFO] people detected: ' + str(len(detections)))

        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # # display frame + detection for debugging
            # cv2.imshow('Detections', frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # if object is detected return coordinates
            if x > 0:
                print('[INFO] bounding box calculated')
                return x, y, w, h


# call detection function
# if you want OpenCV to detect the initial object automatically
# bbox = detect()
# print(bbox)
# or use the ROI selector to select initial object yourself
bbox = cv2.selectROI(frame)
# initialize tracker with detected coordinates
ok = tracker.init(frame, bbox)
if not ok:
    print('[ERROR] loading tracker')
# generate random colours for bounding box
colours = (randint(0, 255), randint(0, 255), randint(0, 255))

# loop through all frames and apply tracker on detected object
while True:
    ok, frame = video.read()
    if not ok:
        print('[INFO] reached end of video')
        break
    # update bounding box in new frame using tracker
    ok, bbox = tracker.update(frame)
    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), colours, 5)
        # show tracking / resize frame if necessary
        resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Tracking', resized)
        # press 'q' to break loop and close window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print('[WARNING] No Track')
        # re-run detection if track was lost
        bbox = detect()
        # restart tracker if person detected
        tracker = cv2.legacy.TrackerCSRT_create()
        tracker.init(frame, bbox)