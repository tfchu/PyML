"""
Sequence
- get an img from webcam
- get contour of cap (color) part of the paint, and return tip location of the contour 
- draw a circle at the tip location, and add that location to myPoints (all points to draw circle)
- draw circle of all myPoints
- repeat
"""
import cv2
import numpy as np
from lib import getContoursVP
from time import sleep

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

# find HSV max/min with colorPicker.py [[R], [G], [B]]
myColors = [[133, 149, 106, 179, 255, 255], [26, 166, 0, 100, 255, 255], [40, 186, 72, 133, 255, 255]]
myColorValues = [[0, 0, 255], [0, 255, 0], [255, 0, 0]]
myPoints = []   # x, y, colorId

def findColor(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)    # get mask (color area of marker)
        x, y = getContoursVP(mask, imgResult)       # get contour of the mask, return tip of the pen
        # draw circle at tip of contour, with corresponding color
        # count: 0 (R), 1 (G), 2 (B)
        cv2.circle(imgResult, (x, y), 10, myColorValues[count], cv2.FILLED)
        # if contour is found (area > 500), add tip point and color id (x, y, count) to newPoints
        if x!=0 and y != 0:                         # (0, 0) if front part of pen is not found
            newPoints.append([x,y,count])           # count: color id
        count += 1                                  # count 0 (R), 1(G), 2 (B)
        # cv2.imshow(str(color[0]), mask) 
    return newPoints                                # e.g. R found, then newPoints = [[x, y, count]] (only 1 point)

# myPoints: [[x, y, color_id], [x, y, color_id], ...], containing all points for all colors
# myColorValues: [r, g, b]
def drawOnCanvas(myPoints, myColorValues):
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)

while True:
    success, img = cap.read()
    imgResult = img.copy()
    newPoints = findColor(img)      # get a list of points (x, y, color)
    if len(newPoints) != 0:
        for newP in newPoints:
            myPoints.append(newP)   # add the found point to myPoints
    if len(myPoints) != 0:
        drawOnCanvas(myPoints, myColorValues)   # draw circles on all of myPoints
    cv2.imshow("Video", imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break