from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from PIL import Image
import torch
import pyttsx3
import cv2
import numpy as np
import io
import time


# ✅ Load YOLOv5 model once when Django starts
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# ✅ Set up TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ✅ Classes you care about
allowed_classes = {"person", "chair", "bicycle", "car"}

# Cooldown mechanism to avoid repeating speech
last_spoken = {}
cooldown = 5  # seconds

# Load YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
engine = pyttsx3.init()
engine.setProperty('rate', 150)
allowed_classes = {"person", "chair", "stairs"}
last_spoken = {}
cooldown = 5  # seconds

class ObstacleDetectionAPIView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        file = request.FILES['image']
        image = Image.open(file).convert('RGB')
        frame = np.array(image)

        results = model(frame)
        spoken_messages = []
        now = time.time()

        for *box, conf, cls in results.xyxy[0]:
            cls_id = int(cls)
            label = model.names[cls_id]
            if label not in allowed_classes:
                continue

            # Direction
            x_center = (box[0] + box[2]) / 2
            width = frame.shape[1]
            if x_center < width / 3:
                direction = "left"
            elif x_center < 2 * width / 3:
                direction = "center"
            else:
                direction = "right"

            # Cooldown
            key = f"{label}_{direction}"
            if key not in last_spoken or (now - last_spoken[key]) > cooldown:
                spoken_messages.append(f"{label} on your {direction}")
                last_spoken[key] = now

        # Combine speech
        message = ", ".join(spoken_messages) if spoken_messages else "Clear path"
        engine.say(message)
        engine.runAndWait()

        return Response({"status": "ok", "message": message})
