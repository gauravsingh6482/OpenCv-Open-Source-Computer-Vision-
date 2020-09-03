# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 08:42:35 2020

@author: gaurav
"""

import cv2 


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_fullbody.xml")

video_capture = cv2.VideoCapture("Subway.mp4")
while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('Video', frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()       
    
    