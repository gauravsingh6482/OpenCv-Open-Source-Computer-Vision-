# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 11:33:24 2020

@author: gaurav
"""


import cv2 
from PIL import Image

# Load the cascades 
catface_cascade = cv2.CascadeClassifier('Cat_haarcascade_frontalcatface_extended.xml') 
humanface_cascade = cv2.CascadeClassifier('Human_haarcascade_frontalface_default.xml')

#Resize

newsize = (600, 600) 
#First image retouches
imgr1 = Image.open("Human_face.jpg") 
imgr1 = imgr1.resize(newsize) 
imgr1.save("resized1.jpg")
#Second image retouches
imgr2 = Image.open("Cat_face.jpg") 
imgr2 = imgr2.resize(newsize) 
imgr2.save("resized2.jpg")

#Grayscale

imgr1 = imgr1.convert('L') 
imgr1.save('ready1.jpg') 
imgr2 = imgr2.convert('L') 
imgr2.save("ready2.jpg")

# Read the input image 

img1 = cv2.imread('ready1.jpg')
 
img2 = cv2.imread('ready2.jpg')

# Face Detection

human_faces = humanface_cascade.detectMultiScale(img1,     
scaleFactor=1.3, minNeighbors=5, minSize=(75, 75)) 

cat_faces = catface_cascade.detectMultiScale(img2, scaleFactor=1.3, 
minNeighbors=5, minSize=(75, 75))


# Draw rectangles

for (i, (x, y, w, h)) in enumerate(human_faces): 
    cv2.rectangle(img1, (x, y), (x+w, y+h), (220, 90, 230), 3)      
    cv2.putText(img1, "Human Face - #{}".format(i + 1), (x, y - 10), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 90, 230), 2) 

for (i, (x, y, w, h)) in enumerate(cat_faces): 
    cv2.rectangle(img2, (x, y), (x+w, y+h), (0,255, 0), 3) 
    cv2.putText(img2, "Cat Faces - #{}".format(i + 1), (x, y - 10), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
   

#Saving the images using imwrite method
    
human=cv2.imwrite("faces_detected1.png", img1)
cat=cv2.imwrite("faces_detected2.png", img2)


while True:
    cv2.imshow("Human", human)
    cv2.imshow("cat",cat)
    if cv2.waitKey(0)==13 or ord('q'):
        break

cv2.destroyAllWindows()



































