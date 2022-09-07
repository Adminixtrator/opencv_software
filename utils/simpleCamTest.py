'''
Simple Cam Test - BGR and Gray
    Create by pythonprogramming.net ==> See the tutorial here:
    https://pythonprogramming.net/loading-video-python-opencv-tutorial
Adapted by Marcelo Rovai - MJRoBot.org @8Feb18
'''

import numpy as np
import cv2
import os
from datetime import date


faceCascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades + 'cascades/haarcascade_frontalface_default.xml'))
cap = cv2.VideoCapture(0)
now = date.today()

# dd:mm:YY
# filename = now.strftime("%d:%m:%Y-%H:%M:%S")

while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) # flip horizontally
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # save video stream
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(filename + '.mp4',fourcc, 20.0, (640,480))

    cv2.imshow('frame', frame)
    # cv2.imshow('gray', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
