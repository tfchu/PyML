'''
cascade classifiers, Paul Viola and Michael Jones, 2001
https://ieeexplore.ieee.org/document/990517
'''
import cv2
import numpy as np

# load the photograph
pixels = cv2.imread('../images/tony.jpg')
# load the pre-trained model
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# perform face detection
bboxes = classifier.detectMultiScale(pixels)
# print bounding box for each detected face
for box in bboxes:
	# extract
	xStart, yStart, width, height = box
	xEnd, yEnd = xStart + width, yStart + height
	# draw a rectangle over the pixels
	cv2.rectangle(pixels, (xStart, yStart), (xEnd, yEnd), (0,0,255), 1)
# show the image
cv2.imshow('face detection', pixels)
# keep the window open until we press a key
cv2.waitKey(0)
# close the window
cv2.destroyAllWindows()