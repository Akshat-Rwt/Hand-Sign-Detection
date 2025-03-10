import cv2
from cvzone.HandTrackingModule import HandDetector


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1) # This is the function in cvzome for detecting the hands. Only maxHands 1

while True:
    success  ,img = cap.read()
    hands , img = detector.findHands(img) # Find the Hands

    # Crop the image
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox'] # Bounding Box
        imgCrop = img[y:y+h , x:x+w] # Starting height/width and the ending height/width
        cv2.imshow("ImageCrop", imgCrop)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

