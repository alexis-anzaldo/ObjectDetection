import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(1)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Create Trackbars
cv2.namedWindow("Frame")
cv2.createTrackbar("Neighbours", "Frame", 0, 20, nothing)

while True:
    _, frame = cap.read() 
    gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    neighbors = cv2.getTrackbarPos("Neighbours", "Frame")

    faces = face_cascade.detectMultiScale(gray, 
                                          scaleFactor=1.3, 
                                          minNeighbors=neighbors)
    for rect in faces:
        (x, y, w, h) = rect
        frame = cv2.rectangle(frame, 
                              pt1=(x, y), 
                              pt2=(x + w, y + h),
                              color=(0, 255, 0), 
                              thickness= 2)


    cv2.imshow("Frame", frame)


    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()