import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1) # This is the function in cvzome for detecting the hands. Only maxHands 1

offset = 20 # Increase the Frame
imgSize = 300

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
            imgWhite[0:imgResizeShape[0], 0:imgResizeShape[1]] = imgResize


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

