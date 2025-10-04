import cv2
import os

def extract_frames(video_path, output_folder, label):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # resize ảnh về 128x128 cho CNN
        frame = cv2.resize(frame, (128, 128))
        cv2.imwrite(os.path.join(output_folder, f"{label}_{count}.jpg"), frame)
        count += 1
    cap.release()

# Ví dụ
extract_frames("videos/realVAL.mp4", "datasets/antispoof/val/real", "real")
extract_frames("videos/fakeVAL.mp4", "datasets/antispoof/val/fake", "fake")
