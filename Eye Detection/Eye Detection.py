# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 09:38:30 2020

@author: gaurav
"""

import cv2 

CarCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

def detect_cars(frame):
    cars=CarCascade.detectMultiScale(frame,1.15 , 4)
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    return frame
carvideo=cv2.VideoCapture("Eyes.mp4")
def simulator():
    
    while carvideo.isOpened():
        ret, frame=carvideo.read()
        controlkey=cv2.waitKey(1)
        if ret:
            cars_frame=detect_cars(frame)
            cv2.imshow('frame',cars_frame)
        else:
            break
        if controlkey==ord('q'):
            break

if __name__ == '__main__':
    simulator()
carvideo.release()
cv2.destroyAllWindows()
    

