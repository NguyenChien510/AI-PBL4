import cv2
from ultralytics import YOLO
import time

# Load model YOLO
model = YOLO("./runs/detect/train1/weights/last.pt")

# Mở webcam
cap = cv2.VideoCapture(0)

print("🚀 Bắt đầu nhận diện real-time...")
print("Nhấn 'q' để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Dự đoán với YOLO
    results = model(frame)
    
    # Vẽ kết quả
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                # Lấy tọa độ
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                
                # Lấy tên người
                person_name = r.names[class_id]
                
                # Vẽ bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Vẽ tên và confidence
                label = f"{person_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Hiển thị frame
    cv2.imshow('Nhan Dien Khuon Mat', frame)
    
    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()