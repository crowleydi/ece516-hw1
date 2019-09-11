# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 17:25:15 2019

@author: wenjing
"""

import cv2
import sys
import os
import numpy as np

print("label: " + sys.argv[1])
print("file: " + sys.argv[2])
	
# Set labels of objects
object_label_1 = sys.argv[1]
# Create a VideoCapture object and read from input file
videoName = sys.argv[2]
cap = cv2.VideoCapture(videoName)


# Get information of the video
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
fp = cap.get(cv2.CAP_PROP_FPS)

# Read the first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
ret, frame_1 = cap.read()

# Convert color image into grayscale image
gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)

# Save the first frame
baseName = os.path.basename(os.path.splitext(videoName)[0])
cv2.imwrite('Output/FirstFrame/' + baseName +'.jpg', gray_1)

# make baseName uniq for video/object
baseName = baseName + '_' + object_label_1

# Select ROI
r = cv2.selectROI(frame_1)
print('r=', r)
# Save the coordinate and size of the rectangle
np.save('Output/Data/'+ baseName + '.npy', r)    

# Define the codec and create VideoWriter objects
outRect = cv2.VideoWriter('Output/Videos/' + baseName + '_rect.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fp, (width, height))
outCropped = cv2.VideoWriter('Output/Videos/' + baseName + '_cropped.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fp, (r[2], r[3])) 

cv2.destroyAllWindows()  

while(cap.isOpened()):
        
    ret, frame = cap.read()
    
    if ret == True:
        
        # Convert color image into grayscale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Crop the frame
        imCrop = frame.copy()[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        cv2.rectangle(frame,(int(r[0]), int(r[1])),(int(r[0]+r[2]), int(r[1]+r[3])), (0,0,255), 3)
        
        # write frames
        outRect.write(frame)
        outCropped.write(imCrop)
        
        cv2.imshow('Grayscale Video', gray)
        cv2.imshow('With Rectangle', frame)
        cv2.imshow('Cropped Image', imCrop)
        cv2.waitKey(1)
    else:
        break
   
# Release everything if job is finished            
cap.release()
outRect.release()
outCropped.release()
cv2.destroyAllWindows()
