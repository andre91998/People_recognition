# -*- coding: utf-8 -*-
"""
@author: Andre Barros de Medeiros
@Date:05/09/2020
@Copyright: Free to use, copy and modify
"""

# import the necessary packages
from __future__ import print_function
from collections import deque
from imutils.object_detection import non_max_suppression
from imutils.video import VideoStream
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help = "path to the (optional) video file")
ap.add_argument("-b", "--buffer", type = int, default = 32, 
            help = "max buffer size")
args = vars(ap.parse_args())

#initialize frame counter
counter = 0
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# keep looping
while True:

    # grab the current frame
    frame = vs.read()
    
    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame
    
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break
    
    # resize image it to (1) reduce detection time and (2) improve detection accuracy
    frame = imutils.resize(frame, width=min(400, frame.shape[1]))
    orig = frame.copy()
    
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
     padding=(8, 8), scale=1.05)
    
    # draw the original bounding boxes
    for (x, y, w, h) in rects: 
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
    		cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    # show the frame
    #cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", frame)
    
    # increment counter
    counter += 1
    
    # if the 'q' key is pressed, stop the loop
    key = cv2.waitKey(1) & 0xFF 
    # (& 0xFF) keeps last 8 bits of  waitKey output
    if key == ord("q"): break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False): vs.stop()

# otherwise, release the camera
else: vs.release()

# close all windows
cv2.destroyAllWindows()