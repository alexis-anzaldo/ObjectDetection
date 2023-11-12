import cv2
from tracker import *

tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("./videos/highway.mp4")

# Object detection form stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100,
                                                     varThreshold=40)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    
    # Extract Region of interest
    roi = frame[:, :460]


    # Object Detection
    mask = object_detector.apply(roi)
    # Remove shadows
    _, mask = cv2.threshold(mask, thresh=254, maxval=255, type=cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 350:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, 
                          pt1=(x, y), 
                          pt2=(x + w, y + h), 
                          color=(0,255,0), 
                          thickness=3)

            detections.append([x, y, w, h])

    # Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, 
                    text=str(id), 
                    org=(x, y -15), 
                    fontFace=cv2.FONT_HERSHEY_PLAIN, 
                    fontScale=1, 
                    color=(255,0,0), 
                    thickness=2)

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)



    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()