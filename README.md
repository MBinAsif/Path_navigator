# 🔍 EchoTrail: Smart Navigation System for the Visually Impaired

EchoTrail is an AI-powered smart navigation system designed for visually impaired individuals. It integrates real-time object detection (YOLO), spatial audio alerts (Pygame), voice commands (SpeechRecognition), GPS tracking, and haptic feedback to enhance safety and mobility.

---

## ✨ Features

- 🧠 Real-time object detection using YOLOv8
- 🗣️ Voice command interface (Google SpeechRecognition)
- 🧭 GPS tracking with location awareness
- 📳 Haptic feedback support (Arduino-compatible)
- 🎧 3D spatial audio alerts
- 🧬 Route learning and familiar object memory
- 📁 Emergency contact management and alerting

---

## 🛠️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/echotrail.git
cd echotrail
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the system
```bash
python manage.py detect_obstacles
```

---

## ⚙️ Dependencies

- Python 3.10+ (⚠️ Not tested with 3.13 yet)
- Django
- OpenCV
- Pyttsx3 (Text-to-speech)
- Pygame (3D audio rendering)
- SpeechRecognition
- PyAudio *(Required for voice commands)*
- Ultralytics YOLOv8

### 🔧 Optional
- Arduino for haptic feedback (via serial communication)
- GPS module or simulated GPS API

---

## 📦 File Structure

```
echotrail/
├── api/
│   ├── management/commands/detect_obstacles.py  # Main command
│   ├── views.py, models.py                      # Django app
├── echotrail/
│   └── urls.py, settings.py
├── user_preferences.json                        # Saved preferences
├── emergency_contacts.json                      # Contact list
├── learned_routes.pkl                           # Route memory
├── location_memory.pkl                          # Familiar obstacles
└── README.md
```

---

## 🚨 Troubleshooting

- **No microphone?**
  > Voice commands will be disabled gracefully.

- **No audio output device?**
  > 3D spatial audio will be skipped.

- **PyAudio not installed?**
  > Use:  
  `pip install pipwin`  
  `pipwin install pyaudio`

- **YOLOv8 not found?**
  > Install Ultralytics:  
  `pip install ultralytics`

---

## 🧪 Demo Commands

- `"what's ahead"`
- `"quiet mode"`
- `"navigate to home"`
- `"emergency"`
- `"repeat"`
- `"where am I"`

---

## 📄 License

MIT License © 2025 EchoTrail Developers

---

## 🤝 Contributing

Pull requests and suggestions are welcome! Please fork the repo and open an issue for major changes.
