# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 12:03:10 2020

@author: gaurav
"""

import cv2
import numpy as np

image=cv2.imread("images.png",0)
image = cv2.GaussianBlur(image, (3, 3), 0)
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                               cv2.THRESH_BINARY, 3, 5) 

images=cv2.imwrite("result_image.png", thresh)

cv2.imshow("Adaptive Mean Thresholding",images) 

cv2.waitKey(0)

cv2.destroyAllWindows()
    