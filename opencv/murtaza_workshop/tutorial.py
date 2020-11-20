# https://docs.opencv.org/3.4/index.html
import cv2
import numpy as np
from lib import stackImages, getContours

# # load images
# img = cv2.imread("Resources/lena.png")
# cv2.imshow("output", img)
# cv2.waitKey(0)      # time in ms, 0 to wait infinitely 

# # load video
# cap = cv2.VideoCapture("Resources/test_video.mp4")
# while True:
#     success, img = cap.read()   # save images (in video) and tell us whether successful or not
#     cv2.imshow("Video", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # webcam
# # id list
# # https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
# cap = cv2.VideoCapture(0)      # 0: id of default webcam
# cap.set(3, 640)                # 3: id of width
# cap.set(4, 480)                # 4: id of height
# # cap.set(10, 100)               # brightness
# while True:
#     success, img = cap.read()   # save images (in video) and tell us whether successful or not
#     cv2.imshow("Video", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # grascale, blur, (canny) edge detector, dilation, erosion
# # traditionally we use RGB for color images, but opencv uses BGR
# img = cv2.imread("Resources/lena.png")
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)  # kernel size has to be odd number, e.g. 3, 7, ..
# imgCanny = cv2.Canny(img, 100, 100)
# kernel = np.ones((5, 5), np.uint8)              # uint8: 0 ~ 255
# imgDilation = cv2.dilate(imgCanny, kernel, iterations=1)
# imgErosion = cv2.erode(imgDilation, kernel, iterations=1)
# cv2.imshow("Gray Image", imgGray)
# cv2.imshow("Blur Image", imgBlur)
# cv2.imshow("Canny Image", imgCanny)
# cv2.imshow("Dilation Image", imgDilation)
# cv2.imshow("Erosion Image", imgErosion)
# cv2.waitKey(0)

# # resize and crop
# # opencv x,y convension
# # ---> x (val1, val2, ...)
# # |
# # y
# # raw image: 
# # [
# #   [[bgr], [bgr], ...]], 
# #   [[bgr], [bgr], ...]], 
# #   ...
# # ]
# # 
# # 
# img = cv2.imread("Resources/lambo.png")
# print(img.shape)        # (462, 623, 3) or (height, width, channel BGR)
# imgResize = cv2.resize(img, (300, 200))     # dsize = (x, y) or (width, height)
# print(imgResize.shape)
# imgCropped = img[0:200, 200:500]            # [height, width]
# cv2.imshow("Image", img)
# cv2.imshow("Resized Image", imgResize)
# cv2.imshow("Cropped Image", imgCropped)
# cv2.waitKey(0)

# # draw shapes (line, rectangle, circle) and add text
# img = np.zeros((256, 512, 3), np.uint8) # (y, x, ch)
# height, width, ch = img.shape           # 256, 512, 3
# # img[:] = (255, 0, 0)
# cv2.line(img, (0,0), (width,height), (0,255,0), 3)  # pt: (x, y)
# cv2.rectangle(img, (0,0), (256, 128), (0,0,255), cv2.FILLED) # pt: (x, y)
# cv2.circle(img, (256, 128), 30, (255, 255, 0), 5)
# cv2.putText(img, "opencv", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 1)    # org: origin
# cv2.imshow("Image", img)
# cv2.waitKey(0)

# # warp perspective/birdview
# # https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga20f62aa3235d869c9956436c870893ae
# img = cv2.imread("Resources/cards.jpg")
# width, height = 250, 350    # a regular card is 2.5inch * 3.5inch
# pts1 = np.float32([[111,219], [287,188], [154,482], [352, 440]])    # corners of a card, from "paint"
# pts2 = np.float32([[0,0], [width, 0], [0, height], [width, height]])
# matrix = cv2.getPerspectiveTransform(pts1, pts2)
# imgOutput = cv2.warpPerspective(img, matrix, (width, height))
# cv2.imshow("Image", img)
# cv2.imshow("Output", imgOutput)
# cv2.waitKey(0)

# # join images
# # hstack/vstack limitation: all images with same channel, no built-in method to scale
# # use stackImages()
# img = cv2.imread("Resources/lena.png")
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgHor = np.hstack((img, img))  # display as img|img|img|... (horizontal)
# imgVer = np.vstack((img, img))  # vertical
# imgStack = stackImages(0.5, ([img, imgGray], [imgGray, img]))
# cv2.imshow("Horizontal", imgHor)
# cv2.imshow("Vertical", imgVer)
# cv2.imshow("Image Stack", imgStack)
# cv2.waitKey(0)

# # color detection - detect orange in img
# # HSV (hue, saturation, value, also known as HSB or hue, saturation, brightness)
# # best values visually
# # (h_min, h_max, s_min, s_max, v_min, v_max) = (0, 19, 110, 240, 153, 255)
# def empty(val):
#     pass
# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars", 640, 240)
# cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
# cv2.createTrackbar("Hue Max", "TrackBars", 19, 179, empty)
# cv2.createTrackbar("Sat Min", "TrackBars", 110, 255, empty)
# cv2.createTrackbar("Sat Max", "TrackBars", 240, 255, empty)
# cv2.createTrackbar("Val Min", "TrackBars", 153, 255, empty)
# cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)
# while True:
#     img = cv2.imread("Resources/lambo.png")
#     imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
#     h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
#     s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
#     s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
#     v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
#     v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
#     # print(h_min, h_max, s_min, s_max, v_min, v_max)
#     lower = np.array([h_min, s_min, v_min])
#     upper = np.array([h_max, s_max, v_max])
#     mask = cv2.inRange(imgHSV, lower, upper)
#     imgResult = cv2.bitwise_and(img, img, mask=mask)
#     # cv2.imshow("Image", img)
#     # cv2.imshow("HSV", imgHSV)
#     # cv2.imshow("Mask", mask)
#     # cv2.imshow("Result", imgResult)
#     imgStack = stackImages(0.6, ([img, imgHSV], [mask, imgResult]))
#     cv2.imshow("Stacked Image", imgStack)
#     cv2.waitKey(1)

# # detect shape 
# img = cv2.imread("Resources/shapes.png")
# imgContour = img.copy()
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1)
# imgCanny = cv2.Canny(imgBlur, 50, 50)               # edge detector
# getContours(imgCanny, imgContour)
# imgBlank = np.zeros_like(img)
# imgStack = stackImages(0.6, ([img, imgGray, imgBlur], [imgCanny, imgContour, imgBlank])) 
# cv2.imshow("Stack", imgStack)
# cv2.waitKey(0)

# face detection: use Viola and Jones harrcascade (old but fast)
# opencv cascades
# https://github.com/opencv/opencv/tree/master/data/haarcascades
faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
img = cv2.imread("Resources/lena.png")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)   # use img also ok
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0),2)
cv2.imshow("Result", img)
cv2.waitKey(0)