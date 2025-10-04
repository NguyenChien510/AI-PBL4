import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ===========================
# Load YOLO model (detect face)
# ===========================
yolo_model = YOLO("./runs/detect/train5/weights/best.pt")

# ===========================
# Load CNN liveness model
# ===========================
cnn_model = load_model("liveness.keras")   # hoặc "liveness.h5"

# ===========================
# Hàm tiền xử lý cho CNN
# ===========================
def preprocess_face(face, target_size=(32, 32)):
    face = cv2.resize(face, target_size)          # resize giống lúc train
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # chuyển RGB
    face = img_to_array(face) / 255.0             # chuẩn hóa [0,1]
    face = np.expand_dims(face, axis=0)           # thêm batch dimension
    return face

# ===========================
# Mở webcam
# ===========================
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detect face
    results = yolo_model(frame)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()   # [x1, y1, x2, y2]
        clss = r.boxes.cls.cpu().numpy()     # class index
        confs = r.boxes.conf.cpu().numpy()   # confidence
        names = r.names                      # class names

        for box, cls, conf in zip(boxes, clss, confs):
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]

            # Mặc định YOLO label
            label = f"{names[int(cls)]} {conf:.2f}"

            if face.size != 0:
                try:
                    # Preprocess cho CNN
                    face_input = preprocess_face(face)

                    # CNN dự đoán real/fake
                    pred = cnn_model.predict(face_input, verbose=0)
                    label_idx = np.argmax(pred)  # 0 = REAL, 1 = FAKE
                    spoof_label = "REAL" if label_idx == 0 else "FAKE"

                    # Ghép YOLO label + CNN label
                    label = f"{names[int(cls)]} {conf:.2f} | {spoof_label}"

                except Exception as e:
                    print("CNN Error:", e)

            # Vẽ bounding box
            color = (0, 255, 0) if "REAL" in label else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Vẽ label
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("YOLO + Liveness Webcam", frame)

    # Nhấn q để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

