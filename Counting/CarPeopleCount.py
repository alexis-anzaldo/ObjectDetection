from yolov5 import YOLOv5
import cv2

# set model params
model_path = "yolov5/weights/yolov5s.pt" # it automatically downloads yolov5s model to given path
device = "cpu" # or "cuda"

# init yolov5 model
od = YOLOv5(model_path, device)

cap = cv2.VideoCapture("./videos/highway.mp4")

while True:
    _, img = cap.read()
    height, width, _ = img.shape

    # Detect Objects
    #results = od.predict(img, conf=0.7, verbose=False)
    results = od.predict(img)
    print(results)

    cv2.imshow("Frame", img)

    key = cv2.waitKey(1)
    if key == 27:
        break