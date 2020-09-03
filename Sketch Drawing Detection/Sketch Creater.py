# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 11:17:24 2020

@author: gaurav
"""


import cv2
import numpy as np

def sketch(images):
    #convert image to grayscal
    gray_img=cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    
    #Clean up image using guassian blur
    img_gray_blur=cv2.GaussianBlur(gray_img , (5,5), 0)
    
    #Extract Edges
    canny_img=cv2.Canny(img_gray_blur, 10, 70)
    
    # Thresholding
    ret, mask=cv2.threshold(canny_img, 70, 50, cv2.THRESH_BINARY_INV)
    return mask

cap=cv2.VideoCapture("video.mp4")
while True:
    ret,frame=cap.read()
    cv2.imshow("Live Sketch", sketch(frame))
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()