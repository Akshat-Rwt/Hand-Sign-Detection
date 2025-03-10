import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

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

        # In first attempt it is black beacuse pixel is 1 if we multiply by 255 it should be white.
        imgWhite = np.ones((imgSize , imgSize,3), np.uint8)*255 # It's from 0 to 255
        imgCrop = img[y - offset:y + h + offset,x - offset: x + w + offset]  # Starting height/width and the ending height/width # I can increase the frame.

        # Now we can over the crop image on the white Sheet so we can collect the data and all the data should be of same sized
        imgCropShape = imgCrop.shape
        imgWhite[0:imgCropShape[0] , 0:imgCropShape[1]] = imgCrop


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

