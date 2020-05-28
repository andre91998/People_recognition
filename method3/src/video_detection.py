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
from scipy.stats import pearsonr
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

coef = (0,0)

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
        old_frame = orig.flatten() #update old frame

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
        
        # increment counter
        counter += 1
        
        # if the 'q' key is pressed, stop the loop
        key = cv2.waitKey(1) & 0xFF 
        # (& 0xFF) keeps last 8 bits of  waitKey output
        if key == ord("q"): break

    if old_frame is not None:
        
        # resize image it to (1) reduce detection time and (2) improve detection accuracy
        frame = imutils.resize(frame, width=min(400, frame.shape[1]))
        
        #flatten current frame for running the Pearson's Correlation 
        flat_frame = frame.flatten()
        #calculate the pearson's correlation coeficient
        coef = pearsonr(flat_frame, old_frame)
        
        #if on second frame, create threshold_arr for holding the PCCs
        if (counter == 1): 
            threshold_arr = np.array(coef[0])
            
        #if on any other frame, append to the array
        else:
            threshold_arr = np.append(threshold_arr, coef[0])
            
        #dynamical threshold calcuation: mean of all previous PCCs    
        threshold = np.mean(threshold_arr)
        print(coef,threshold)
        
        #if PCC below the threshold, re-classify
        if (((coef[0] < threshold)and(coef[0]>0))or((coef[0]>-1*threshold)and(coef[0]<0))):
            orig = frame.copy()
            old_frame = orig.flatten() #update old_frame
            
            # detect people in the image
            (rects, weights) = hog.detectMultiScale(frame, winStride=(4,4),
             padding=(8,8), scale=1.05)
            
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
        
        # if PCC is above threshold, update frame, but keep same rectangles
        else:
            # draw the same rectangle as before on the current frame (which wasn't proccessed by HoG)
            for (xA, yA, xB, yB) in pick:
                cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            
            cv2.imshow("After NMS", frame)
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