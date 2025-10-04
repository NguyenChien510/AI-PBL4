#Train with Python API
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

    # Train 
model.train(
        data="datasets\dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="train"
    )