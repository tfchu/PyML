import cv2

# real-time face detection with webcam
faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)      # 0: id of default webcam
cap.set(3, 640)                # 3: id of width
cap.set(4, 480)                # 4: id of height
# cap.set(10, 100)               # brightness

while True: 
    # read image
    success, img = cap.read()   # save images (in video) and tell us whether successful or not
    # detect face
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)   # use img also ok
    # draw rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0),2)
    # show image
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break