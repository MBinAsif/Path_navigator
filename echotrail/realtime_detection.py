import cv2
import pyttsx3
import time
from ultralytics import YOLO

# Load YOLOv8 small model
model = YOLO('yolov8s.pt')

# TTS setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Classes to speak
allowed_classes = {"person", "chair", "bicycle", "car", "dog", "cat"}

# Avoid repeating same message
last_spoken = {}
cooldown = 5  # seconds

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("✅ YOLOv8s running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(frame_rgb, conf=0.35, verbose=False)

    frame_width = frame.shape[1]
    current_time = time.time()
    found_valid = False

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            print(f"Detected: {label} ({conf:.2f})")

            if label not in allowed_classes:
                continue

            found_valid = True

            # Estimate distance
            box_height = y2 - y1
            if box_height <= 0:
                continue
            object_height = 170 if label == "person" else 80  # avg height in cm
            focal_length = 615  # Approx focal length
            distance_cm = int((object_height * focal_length) / box_height)

            # Direction
            center_x = (x1 + x2) / 2
            if center_x < frame_width / 3:
                direction = "left"
            elif center_x < 2 * frame_width / 3:
                direction = "center"
            else:
                direction = "right"

            key = f"{label}-{direction}"
            if key not in last_spoken or (current_time - last_spoken[key]) > cooldown:
                spoken = f"A {label} on your {direction}, {distance_cm} centimeters away"
                print(f"🔊 {spoken}")
                engine.say(spoken)
                engine.runAndWait()
                last_spoken[key] = current_time

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Fallback: if no allowed objects were found
    if not found_valid:
        if "clear_path" not in last_spoken or (current_time - last_spoken["clear_path"]) > cooldown:
            print("🔊 Clear path ahead.")
            engine.say("Clear path ahead")
            engine.runAndWait()
            last_spoken["clear_path"] = current_time

    cv2.imshow("YOLOv8s - Real-Time Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
