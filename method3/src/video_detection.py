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
from skimage.filters import sobel, unsharp_mask
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
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

old_frame = None
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
    if old_frame is None:
        
        # resize image it to (1) reduce detection time and (2) improve detection accuracy
        frame = imutils.resize(frame, width=min(400, frame.shape[1]))
        orig = frame.copy()
        old_frame = orig
        #vor = Voronoi(orig[:,:,0])
        #voronoi_plot_2d(vor)
        #plt.show()
        #cv2.imshow("sobel",sobel(frame[:,:,0]))
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("gray", gray)
        #cv2.imshow("sharpen",unsharp_mask(gray))
        #frame = unsharp_mask(gImg)
        
        # detect people in the image
        (rects, weights) = hog.detectMultiScale(frame, winStride=(7,7),
         padding=(4,4), scale=1.3)
        
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
        
        # increment counter and update last frame
        counter += 1
        
        # if the 'q' key is pressed, stop the loop
        key = cv2.waitKey(1) & 0xFF 
        # (& 0xFF) keeps last 8 bits of  waitKey output
        if key == ord("q"): break

    if old_frame is not None:
        
        # resize image it to (1) reduce detection time and (2) improve detection accuracy
        frame = imutils.resize(frame, width=min(400, frame.shape[1]))
        if max(directed_hausdorff(frame[:,:,0], old_frame[:,:,0])[0], directed_hausdorff(old_frame[:,:,0], frame[:,:,0])[0]) > 0:
            orig = frame.copy()
            #vor = Voronoi(orig[:,:,0])
            #voronoi_plot_2d(vor)
            #plt.show()
            #cv2.imshow("sobel",sobel(frame[:,:,0]))
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #cv2.imshow("gray", gray)
            #cv2.imshow("sharpen",unsharp_mask(gray))
            #frame = unsharp_mask(gImg)
            
            # detect people in the image
            (rects, weights) = hog.detectMultiScale(frame, winStride=(7,7),
             padding=(4,4), scale=1.3)
            
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
            
            # increment counter and update last frame
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