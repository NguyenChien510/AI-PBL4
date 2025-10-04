from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import pickle
import time
import os

# --- Load model YOLO để nhận diện tên người ---
yolo_model = YOLO("./runs/detect/train5/weights/best.pt")  

# --- Load Liveness CNN ---
liveness_model = load_model("liveness.keras")
le = pickle.loads(open("le.pickle", "rb").read())

# --- Load SSD Face Detector (fallback) ---
protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
ssd_net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# --- Mở webcam ---
cap = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    (h, w) = frame.shape[:2]
    detected = False  # kiểm tra xem YOLO có detect không
    
    # 1. YOLO detect tên người
    results = yolo_model(frame)
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        names = r.names
        
        for box, cls_id, score in zip(boxes, class_ids, scores):
            if score < 0.5:
                continue
            detected = True
            x1, y1, x2, y2 = box.astype(int)
            
            # Crop face để CNN check fake/real
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            
            face_cnn = cv2.resize(face, (64,64))
            face_cnn = face_cnn.astype("float")/255.0
            face_cnn = img_to_array(face_cnn)
            face_cnn = np.expand_dims(face_cnn, axis=0)
            
            preds = liveness_model.predict(face_cnn)[0]
            j = np.argmax(preds)
            live_label = "Real" if j==1 else "Fake"
            person_name = names[int(cls_id)]
            
            label = f"{person_name} | {live_label}: {preds[j]:.2f}"
            color = (0,255,0) if j==1 else (0,0,255)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    
    # 2. Fallback SSD nếu YOLO không detect
    if not detected:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0,
                                     (300,300), (104.0, 177.0, 123.0))
        ssd_net.setInput(blob)
        detections = ssd_net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence < 0.5:
                continue
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            startX, startY, endX, endY = box.astype(int)
            startX, startY = max(0,startX), max(0,startY)
            endX, endY = min(w,endX), min(h,endY)
            
            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
            
            face_cnn = cv2.resize(face, (64,64))
            face_cnn = face_cnn.astype("float")/255.0
            face_cnn = img_to_array(face_cnn)
            face_cnn = np.expand_dims(face_cnn, axis=0)
            
            preds = liveness_model.predict(face_cnn)[0]
            j = np.argmax(preds)
            live_label = "Real" if j==1 else "Fake"
            
            label = f"Unknown | {live_label}: {preds[j]:.2f}"
            color = (0,255,0) if j==1 else (0,0,255)
            cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (startX,startY), (endX,endY), color, 2)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
