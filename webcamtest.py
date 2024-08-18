from ultralytics import YOLO
import cv2

# Muat model YOLOv8
model = YOLO('yolov8n.pt')  # Gunakan 'yolov8n.pt' untuk model yang lebih ringan

# Buka webcam (0 untuk default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Lakukan deteksi objek pada frame
    results = model(frame)

    # Tampilkan hasil deteksi pada frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinat bounding box
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class ID

            # Tampilkan bounding box dan label pada frame
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tampilkan frame dengan hasil deteksi
    cv2.imshow("Deteksi Objek dengan YOLOv8", frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan webcam dan tutup jendela
cap.release()
cv2.destroyAllWindows()
