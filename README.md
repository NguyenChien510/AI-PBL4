### DETECT
yolo task=detect mode=predict model="./runs/detect/train/weights/best.pt" source="./datasets/images/train/beck1.jpg"


### TRAIN
yolo task=detect mode=train model=yolov8n.pt data=datasets/dataset.yaml epochs=10 imgsz=640