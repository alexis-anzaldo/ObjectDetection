import cv2
import mediapipe as mp

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read() 
    height, width, _ = frame.shape

    # Transofrm image to RGB to work with mediapipe
    image_rgb = cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB)

    # Facial landmarks
    result = face_mesh.process(image_rgb)
    # Get coordinate of each point
    if result.multi_face_landmarks is not None:
        for facial_landmarks in result.multi_face_landmarks:
            for i in range (0,468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)

                cv2.circle(frame, center=(x,y), radius=1, color=(255,0,0), thickness=-1)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break