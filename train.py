#Train with Python API
from ultralytics import YOLO
if __name__ == "__main__":
    # Load model local
    model = YOLO(r"D:\AI-PBL4\models\yolov8n.pt")

    # Train 
    model.train(
        data=r"D:\AI-PBL4\datasets\dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="train_local"
    )