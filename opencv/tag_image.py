import cv2
import numpy as np

print(cv2.__version__)
img = cv2.imread('../toolset_test/koala.png')   # (h, w, c)
(h, w) = img.shape[:2]

(startX, startY, endX, endY) = np.array([w/2-10, h/2-10, w/2+10, h/2+10]).astype('int')

cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
cv2.putText(img, 'koala', (startX - 10, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 225), 2)

cv2.imshow('Image', img)
cv2.waitKey(0)