import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


cap = cv2.VideoCapture(1)

while True:
    _, frame = cap.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2,y2), color=(0,255,0), thickness=2)

        landmarks = predictor(gray, face)

        for n in range(0,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, center=(x,y),radius=4,color=(255,0,0),thickness=-1)

    cv2.imshow("Frame", frame)


    key = cv2.waitKey(1)
    if key == 27:
        break