import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
    help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True,
    help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, default='face_detector',
    help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=1,
    help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())

# Load model SSD
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Tạo thư mục output nếu chưa có
os.makedirs(args["output"], exist_ok=True)

# Video input
vs = cv2.VideoCapture(args["input"])
read = 0
saved = 0

video_name = os.path.splitext(os.path.basename(args["input"]))[0]  # "real" hoặc "fake"

while True:
    grabbed, frame = vs.read()
    if not grabbed:
        break

    read += 1
    if read % args["skip"] != 0:
        continue

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")

            # Giới hạn box trong frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            filename = f"{video_name}_{saved:05d}.png"
            p = os.path.join(args["output"], filename)
            cv2.imwrite(p, face)
            saved += 1
            print(f"[INFO] saved {p} to disk")

vs.release()
cv2.destroyAllWindows()
