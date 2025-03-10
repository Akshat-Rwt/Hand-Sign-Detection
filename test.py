import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1) # This is the function in cvzome for detecting the hands. Only maxHands 1
classifier = Classifier("Model/keras_model.h5" , "Model/labels.txt") # This is used to classify

offset = 20 # Increase the Frame
imgSize = 300

# Save the Images
folder = "Data/C"
counter = 0 #how many images we save in this

labels = ["A", "B", "C"]
while True:
    success  ,img = cap.read()
    hands , img = detector.findHands(img) # Find the Hands

    # Crop the image
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox'] # Bounding Box

        # In first attempt it is black because pixel is 1 if we multiply by 255 it should be white.
        imgWhite = np.ones((imgSize , imgSize,3), np.uint8)*255 # It's from 0 to 255

        imgCrop = img[y - offset:y + h + offset,x - offset: x + w + offset]  # Starting height/width and the ending height/width # I can increase the frame.

        # Now we can over the crop image on the white Sheet so we can collect the data and all the data should be of same sized
        imgCropShape = imgCrop.shape

        # Now we can make the height and width of the image is same so every image should be in the same size. If height is bigger than the width then we can make height 300. Or if the width is bigger than the height then we can make the width 300.
        aspectRatio = h/w

        # Height is always 300.
        if aspectRatio > 1: # If height is greater than the width then .
            k = imgSize/h # Now height is 300
            # Width Calculated
            wCal = math.ceil(k*w) # Ceil is used to rounds a number up to the nearest integer

            imgResize = cv2.resize(imgCrop , (wCal , imgSize))
            imgResizeShape = imgResize.shape

            # Now we can make the crop image is in the centre when it overlay on white image
            # Width Gap
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:,  wGap:wCal+wGap] = imgResize
            prediction , index = classifier.getPrediction(img)
            print(prediction , index)

        # Now we can make it for the width
        else:
            k = imgSize / w  # Now Width is 300
            # Height Calculated
            hCal = math.ceil(k * h)  # Ceil is used to rounds a number up to the nearest integer

            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape

            # Now we can make the crop image is in the centre when it overlay on white image
            # Height Gap
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :  ] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
