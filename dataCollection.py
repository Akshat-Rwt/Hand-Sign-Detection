import cv2
from cvzone.HandTrackingModule import HandDetector


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1) # This is the function in cvzome for detecting the hands. Only maxHands 1

while True:
    success  ,img = cap.read()
    hands , img = detector.findHands(img) # Find the Hands
    cv2.imshow("Image", img)
    cv2.waitKey(1)

