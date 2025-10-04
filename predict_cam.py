import cv2
import torch
from ultralytics import YOLO
import torchvision.transforms as transforms
from cnn_model import AntiSpoofCNN

# Load YOLO model (detect face)
model = YOLO("./runs/detect/train5/weights/best.pt")

# Load CNN model (real/fake)
cnn_model = AntiSpoofCNN()
cnn_model.load_state_dict(torch.load("cnn_face_antispoof.pth", map_location="cpu"))
cnn_model.eval()

# Transform cho CNN
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Mở webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detect face
    results = model(frame)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()   # [x1, y1, x2, y2]
        clss = r.boxes.cls.cpu().numpy()     # class index
        confs = r.boxes.conf.cpu().numpy()   # confidence
        names = r.names                      # class names

        for box, cls, conf in zip(boxes, clss, confs):
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]

            # Default label
            label = f"{names[int(cls)]} {conf:.2f}"

            if face.size != 0:
                try:
                    # Preprocess cho CNN
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_tensor = transform(face_rgb).unsqueeze(0)

                    # CNN dự đoán real/fake
                    with torch.no_grad():
                        pred = cnn_model(face_tensor)
                        label_idx = torch.argmax(pred).item()
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

    cv2.imshow("YOLO + CNN Webcam", frame)

    # Nhấn q để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
