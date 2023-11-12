import cv2
import numpy as np

roi = cv2.imread("./images/vitamin-crop.png")
w, h, _ = roi.shape
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist(hsv_roi, channels=[0], mask=None, histSize=[180] , ranges=[0,180])


cap = cv2.VideoCapture(1)

term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Get Backprojection
    mask = cv2.calcBackProject([hsv], channels=[0], hist=roi_hist,  ranges=[0,180], scale=1)
    
    ret, track_window = cv2.CamShift(mask, window=(200,200,w,h), criteria=term_criteria)
    
    pts = cv2.boxPoints(ret)
    pts = np.intp(pts)
    cv2.polylines(frame, pts=[pts], isClosed=True, color=(255,0,0), thickness=2)

    cv2.imshow("mask", mask)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()