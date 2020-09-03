# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:37:14 2020

@author: gaurav
"""

import cv2
import numpy

cascade_src = 'two_wheeler.xml'

video_src = 'two_wheeler2.mp4'

cap = cv2.VideoCapture(video_src)

bike_cascade = cv2.CascadeClassifier(cascade_src)


while True:
    ret, img = cap.read()
    
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    bikes = bike_cascade.detectMultiScale(gray,1.1, 1)


    for (x,y,w,h) in bikes:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,'Bike',(x+x,y+y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
    
    cv2.imshow('video', img)
    
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()