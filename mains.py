from ultralytics import YOLO
from datetime import datetime
import geocoder
import cv2
import firebase_admin
from firebase_admin import credentials, db
import json
import os
import time

# Load YOLO model
model = YOLO(r"D:\VINYASA\FINAL Project\Pothole_Final\runs\train\pothole_model4\weights\best.pt")

# Firebase setup
cred = credentials.Certificate("cred.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://pothole-detection-96a46-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Firebase DB reference
ref = db.reference('potholes')

# JSON log file setup
log_file = "pothole_log.json"
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        logs = json.load(f)
else:
    logs = []

# Start webcam
cap = cv2.VideoCapture(0)
start_time = time.time()

print("Starting real-time pothole detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Stop after 5 minutes (optional)
    if time.time() - start_time > 120:
        print("2 minutes passed. Stopping detection.")
        break

    # Run YOLO detection
    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Get GPS coordinates
            g = geocoder.ip('me')
            if g.latlng:
                lat, lng = g.latlng
                data = {
                    'timestamp': str(datetime.now()),
                    'location': {'lat': lat, 'lng': lng},
                    'label': label,
                    'confidence': conf
                }


                # Save to local JSON log
                logs.append(data)
                with open(log_file, "w") as f:
                    json.dump(logs, f, indent=4)

    # Show webcam feed with detections
    cv2.imshow("Pothole Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Detection stopped. Logs saved to pothole_log.json.")

print("\nDetection session ended.")
print("📍 View all detected potholes in Firebase here:")
print("https://pothole-detection-96a46-default-rtdb.asia-southeast1.firebasedatabase.app/potholes.json")
