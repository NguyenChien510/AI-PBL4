from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    # Train tiếp
    model.train(
        data="datasets/dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="train"
    )
