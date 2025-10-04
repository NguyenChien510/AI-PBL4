#Train with Python API
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="datasets/dataset.yaml", epochs=50, imgsz=640)