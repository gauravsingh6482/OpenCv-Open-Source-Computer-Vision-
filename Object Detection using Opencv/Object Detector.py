# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:35:40 2020

@author: gaurav
"""
import cv2
import numpy as np

# Source data : Video File
video_source = 'videoplayback.mp4'

# Read the source video file
vid_file = cv2.VideoCapture(video_source)

# pre trained classifiers
car_classifier = 'cars.xml'
pedestrian_classifier = 'pedestrian.xml'
bus_classifier = 'Bus_front.xml'
twowheeler_classifier = 'two_wheeler.xml'


# Classified Trackers
car_tracker = cv2.CascadeClassifier(car_classifier)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier)
bus_tracker = cv2.CascadeClassifier(bus_classifier)
twowheeler_tracker = cv2.CascadeClassifier(twowheeler_classifier)


while True:
    # start reading video file frame by frame like an image
    (read_successful, frame) = vid_file.read()

    if read_successful:
        #convert to grey scale image
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect Cars, Pedestrians, Bus and 2Wheelers
    cars = car_tracker.detectMultiScale(gray_frame,1.1,1)
    pedestrians = pedestrian_tracker.detectMultiScale(gray_frame,1.1,1)
    bus = bus_tracker.detectMultiScale(gray_frame, 1.1, 1)
    twowheeler = twowheeler_tracker.detectMultiScale(gray_frame, 1.1, 1)


    # Draw rectangle around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #cv2.rectangle(gray_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Draw square around the pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Human', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw square around the bus
    for (x, y, w, h) in bus:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'Bus', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Draw square around the twowheeler
    for (x, y, w, h) in twowheeler:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (216, 255, 0), 2)
        cv2.putText(frame, 'Bike', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (216, 255, 0), 2)


    # display the imapge with the face spotted
    cv2.imshow('Detect Objects On Road',frame)

    # capture key
    key = cv2.waitKey(1)

    # Stop incase Esc is pressed
    if key == 27:
        break

# Release video capture object
vid_file.release()
cv2.destroyAllWindows()

print("That's it...")