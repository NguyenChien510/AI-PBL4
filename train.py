from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("./runs/detect/train/weights/last.pt")  # Load model đã train
    
    # In thông tin model để xác nhận
    print("=== THÔNG TIN MODEL ===")
    print(f"Model: {model.__class__.__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.model.parameters())}")
    print(f"Model is training: {model.model.training}")
    
    # Train tiếp
    results = model.train(
        data="datasets/dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=32,
        name="train1",
        resume=False,  # Đảm bảo không resume mà fine-tune
        verbose=True   # Hiển thị chi tiết
    )