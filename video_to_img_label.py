import cv2
import os
import numpy as np

# --- Cấu hình ---
video_path = "videos/trongnguyenval.mp4"
class_id = 3
output_img_dir = "datasets/images/val"
output_label_dir = "datasets/labels/val"

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

confidence_threshold = 0.5
skip_frames = 1  # đọc mỗi frame thứ n

# --- Load SSD face detector ---
protoPath = "face_detector/deploy.prototxt"
modelPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# --- Video capture ---
cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    if frame_count % skip_frames != 0:
        continue

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Tính label nếu phát hiện mặt
    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype(int)
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)

            # Tính bbox normalized YOLO
            x_center = ((startX + endX) / 2) / w
            y_center = ((startY + endY) / 2) / h
            width = (endX - startX) / w
            height = (endY - startY) / h

            # Lưu file label
            saved_count += 1
            img_filename = f"TrongNguyen{saved_count:05d}.png"
            img_path = os.path.join(output_img_dir, img_filename)
            label_path = os.path.join(output_label_dir, os.path.splitext(img_filename)[0] + ".txt")

            # Lưu ảnh full frame
            cv2.imwrite(img_path, frame)

            # Lưu label
            with open(label_path, "w") as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            print(f"[INFO] Saved {img_path} and label {label_path}")

cap.release()
cv2.destroyAllWindows()
print(f"[DONE] Total images/labels saved: {saved_count}")
