import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import pickle
import os

# Cài đặt cho OpenCV (nếu cần dùng RTSP)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"

class FaceRecognitionSystem:
    def __init__(self, yolo_model_path, liveness_model_path, label_encoder_path, face_detector_path, confidence=0.5):
        # Load YOLO model cho nhận diện khuôn mặt
        print("[INFO] loading YOLO face recognition model...")
        self.yolo_model = YOLO(yolo_model_path)
        
        # Load face detector cho liveness detection
        print("[INFO] loading face detector for liveness...")
        protoPath = os.path.sep.join([face_detector_path, "deploy.prototxt"])
        modelPath = os.path.sep.join([face_detector_path, "res10_300x300_ssd_iter_140000.caffemodel"])
        self.face_net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        
        # Load liveness detection model
        print("[INFO] loading liveness detector...")
        self.liveness_model = load_model(liveness_model_path)
        self.le = pickle.loads(open(label_encoder_path, "rb").read())
        
        self.confidence = confidence
    
    def detect_liveness(self, frame, face_box):
        """Phát hiện fake/real cho khuôn mặt"""
        startX, startY, endX, endY = face_box
        
        # Lấy vùng khuôn mặt
        face = frame[startY:endY, startX:endX]
        
        # Kiểm tra nếu face rỗng
        if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
            return "unknown", 0.0
        
        # Preprocess ảnh cho liveness detection
        face = cv2.resize(face, (64, 64))
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        
        # Dự đoán
        preds = self.liveness_model.predict(face)[0]
        j = np.argmax(preds)
        label = self.le.classes_[j]
        confidence = preds[j]
        
        return label, confidence
    
    def process_frame(self, frame):
        """Xử lý frame để nhận diện khuôn mặt và kiểm tra liveness"""
        results = self.yolo_model(frame)
        
        # Tạo bản sao của frame để vẽ kết quả
        output_frame = frame.copy()
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Lấy tọa độ từ YOLO
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    yolo_confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    
                    # Lấy tên người từ YOLO
                    person_name = r.names[class_id]
                    
                    # Kiểm tra liveness cho khuôn mặt này
                    liveness_label, liveness_conf = self.detect_liveness(frame, (x1, y1, x2, y2))
                    
                    # Xác định màu sắc và thông tin hiển thị
                    if liveness_label == "fake":
                        color = (0, 0, 255)  # Đỏ cho fake
                        label_text = "FAKE"  # Không hiển thị tên cho fake
                    else:
                        color = (0, 255, 0)  # Xanh cho real
                        label_text = person_name  # Hiển thị tên cho real
                    
                    # Vẽ bounding box
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Vẽ thông tin
                    cv2.putText(output_frame, label_text, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return output_frame

def main():
    # Các đường dẫn model - cần điều chỉnh theo hệ thống của bạn
    YOLO_MODEL_PATH = "./runs/detect/train1/weights/last.pt"
    LIVENESS_MODEL_PATH = "liveness.keras"
    LABEL_ENCODER_PATH = "le.pickle"
    FACE_DETECTOR_PATH = "face_detector"
    
    # Khởi tạo hệ thống
    try:
        system = FaceRecognitionSystem(
            yolo_model_path=YOLO_MODEL_PATH,
            liveness_model_path=LIVENESS_MODEL_PATH,
            label_encoder_path=LABEL_ENCODER_PATH,
            face_detector_path=FACE_DETECTOR_PATH,
            confidence=0.5
        )
        print("🚀 Hệ thống nhận diện khởi động thành công!")
        print("Nhấn 'q' để thoát")
    except Exception as e:
        print(f"❌ Lỗi khi khởi tạo hệ thống: {e}")
        return
    
    # Mở webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Không thể mở webcam!")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Xử lý frame
        processed_frame = system.process_frame(frame)
        
        # Hiển thị kết quả
        cv2.imshow('Face Recognition & Liveness Detection', processed_frame)
        
        # Thoát khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()