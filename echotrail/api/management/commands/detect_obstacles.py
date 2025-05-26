from django.core.management.base import BaseCommand
import cv2
import pyttsx3
import time
import numpy as np
import threading
import queue
from ultralytics import YOLO
from collections import defaultdict, deque
import math
import speech_recognition as sr
import json
import os
from datetime import datetime
import requests
import pygame
import serial
import pickle

class Command(BaseCommand):
    help = 'Comprehensive blind navigation system with voice commands, GPS, haptic feedback, and learning capabilities'

    def __init__(self):
        super().__init__()
        self.audio_queue = queue.Queue()
        self.is_speaking = False
        self.engine_lock = threading.Lock()
        self.voice_commands_active = True
        self.quiet_mode = False
        self.learning_mode = True
        
        # Voice recognition
        self.recognizer = sr.Recognizer()
        try:
            self.microphone = sr.Microphone()
        except OSError:
            print("⚠️ No input device available. Voice recognition disabled.")
            self.microphone = None
            self.voice_commands_active = False
        
        # GPS and location
        self.current_location = None
        self.route_history = deque(maxlen=100)
        self.learned_routes = self.load_learned_routes()
        self.poi_database = self.load_poi_database()
        
        # Haptic feedback
        self.haptic_device = None
        self.init_haptic_device()
        
        # 3D Audio
        self.audio_enabled = True
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        try:
            pygame.mixer.init()
        except pygame.error as e:
            print(f"⚠️ Audio initialization failed: {e}")
            self.audio_enabled = False
        
        # User preferences
        self.user_preferences = self.load_user_preferences()
        
        # Emergency contacts
        self.emergency_contacts = self.load_emergency_contacts()
        
        # Object memory for familiar areas
        self.location_memory = defaultdict(list)
        
    def handle(self, *args, **kwargs):
        print("🔁 Starting Comprehensive Blind Navigation System...")
        print("🎯 Features: Voice commands, GPS navigation, haptic feedback, spatial audio, learning AI")

        model = YOLO('yolov8s.pt')
        engine = pyttsx3.init()
        engine.setProperty('rate', self.user_preferences.get('speech_rate', 140))
        engine.setProperty('volume', self.user_preferences.get('volume', 0.9))
        
        # Enhanced object classes with priority levels
        navigation_objects = {
            # High priority (immediate safety concerns)
            "person": {"priority": 1, "height": 170, "alert_distance": 200, "haptic_pattern": "urgent_pulse"},
            "car": {"priority": 1, "height": 150, "alert_distance": 300, "haptic_pattern": "urgent_pulse"},
            "truck": {"priority": 1, "height": 200, "alert_distance": 400, "haptic_pattern": "urgent_pulse"},
            "bus": {"priority": 1, "height": 250, "alert_distance": 400, "haptic_pattern": "urgent_pulse"},
            "bicycle": {"priority": 1, "height": 100, "alert_distance": 150, "haptic_pattern": "quick_pulse"},
            "motorcycle": {"priority": 1, "height": 120, "alert_distance": 200, "haptic_pattern": "quick_pulse"},
            "traffic light": {"priority": 1, "height": 300, "alert_distance": 100, "haptic_pattern": "double_pulse"},
            "stop sign": {"priority": 1, "height": 200, "alert_distance": 150, "haptic_pattern": "double_pulse"},
            
            # Medium priority (navigation obstacles)
            "chair": {"priority": 2, "height": 80, "alert_distance": 100, "haptic_pattern": "single_pulse"},
            "bench": {"priority": 2, "height": 80, "alert_distance": 100, "haptic_pattern": "single_pulse"},
            "potted plant": {"priority": 2, "height": 60, "alert_distance": 80, "haptic_pattern": "gentle_pulse"},
            "suitcase": {"priority": 2, "height": 50, "alert_distance": 80, "haptic_pattern": "single_pulse"},
            "backpack": {"priority": 2, "height": 40, "alert_distance": 60, "haptic_pattern": "gentle_pulse"},
            "umbrella": {"priority": 2, "height": 100, "alert_distance": 70, "haptic_pattern": "gentle_pulse"},
            "fire hydrant": {"priority": 2, "height": 60, "alert_distance": 80, "haptic_pattern": "single_pulse"},
            
            # Lower priority (awareness objects)
            "dog": {"priority": 3, "height": 50, "alert_distance": 100, "haptic_pattern": "soft_pulse"},
            "cat": {"priority": 3, "height": 25, "alert_distance": 50, "haptic_pattern": "soft_pulse"},
            "handbag": {"priority": 3, "height": 30, "alert_distance": 50, "haptic_pattern": "soft_pulse"},
            "cell phone": {"priority": 3, "height": 15, "alert_distance": 30, "haptic_pattern": "soft_pulse"},
        }

        last_spoken = {}
        object_tracking = defaultdict(list)
        cooldown_priorities = {1: 2, 2: 4, 3: 6}
        
        # Start background threads
        threading.Thread(target=self.audio_worker, daemon=True).start()
        threading.Thread(target=self.voice_command_listener, daemon=True).start()
        threading.Thread(target=self.gps_tracker, daemon=True).start()
        threading.Thread(target=self.route_learner, daemon=True).start()

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return

        print("✅ Comprehensive Navigation System Active")
        print("🎙️  Voice Commands: 'what's ahead', 'repeat', 'quiet mode', 'scan left/right', 'emergency', 'navigate to'")
        print("📍 GPS tracking active")
        print("🔊 3D spatial audio enabled")
        print("📳 Haptic feedback ready")
        print("🧠 Learning mode active")
        print("Press 'q' to quit")

        frame_count = 0
        last_clear_path_time = 0
        last_location_update = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model.predict(frame_rgb, conf=0.25, verbose=False)

                frame_height, frame_width = frame.shape[:2]
                current_time = time.time()
                detected_objects = []
                
                # Process detections
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id]

                        if label not in navigation_objects:
                            continue

                        obj_info = navigation_objects[label]
                        box_width = x2 - x1
                        box_height = y2 - y1
                        
                        if box_height <= 0:
                            continue

                        # Enhanced distance calculation with calibration
                        focal_length = self.user_preferences.get('focal_length', 615)
                        distance_cm = int((obj_info["height"] * focal_length) / box_height)
                        
                        # More precise positioning
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Enhanced directional system (7 zones for better precision)
                        direction_zones = [
                            (0.0, 0.15, "far left"),
                            (0.15, 0.35, "left"),
                            (0.35, 0.45, "slightly left"),
                            (0.45, 0.55, "center"),
                            (0.55, 0.65, "slightly right"),
                            (0.65, 0.85, "right"),
                            (0.85, 1.0, "far right")
                        ]
                        
                        direction = "center"
                        for start, end, dir_name in direction_zones:
                            if start <= center_x/frame_width < end:
                                direction = dir_name
                                break
                        
                        # Vertical positioning
                        if center_y < frame_height * 0.25:
                            vertical_pos = "high"
                        elif center_y > frame_height * 0.75:
                            vertical_pos = "low"
                        else:
                            vertical_pos = "middle"

                        detected_objects.append({
                            'label': label,
                            'distance': distance_cm,
                            'direction': direction,
                            'vertical_pos': vertical_pos,
                            'priority': obj_info["priority"],
                            'center_x': center_x,
                            'center_y': center_y,
                            'confidence': conf,
                            'alert_distance': obj_info["alert_distance"],
                            'haptic_pattern': obj_info["haptic_pattern"],
                            'box_coords': (x1, y1, x2, y2)
                        })

                        # Visual feedback
                        color_map = {1: (0, 0, 255), 2: (0, 165, 255), 3: (0, 255, 0)}
                        color = color_map.get(obj_info["priority"], (255, 255, 255))
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label} {distance_cm}cm", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Sort by priority and distance
                detected_objects.sort(key=lambda x: (x['priority'], x['distance']))
                
                # Generate alerts if not in quiet mode
                if not self.quiet_mode and detected_objects:
                    self.generate_enhanced_alerts(detected_objects, last_spoken, 
                                                cooldown_priorities, current_time)
                    self.send_haptic_feedback(detected_objects)
                    last_clear_path_time = current_time
                elif not self.quiet_mode:
                    # Clear path announcement
                    if (current_time - last_clear_path_time) > 15:
                        if not self.is_speaking:
                            self.play_spatial_audio("Path is clear ahead", 0, 1.0)
                            last_clear_path_time = current_time

                # Movement analysis and learning
                if frame_count % 15 == 0:
                    self.analyze_movement_patterns(detected_objects, object_tracking, current_time)
                    
                # Location-based learning
                if frame_count % 30 == 0 and self.current_location:
                    self.update_location_memory(detected_objects)

                # GPS context updates
                if current_time - last_location_update > 5:
                    self.provide_location_context()
                    last_location_update = current_time

                cv2.imshow("Comprehensive Navigation System", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("🛑 Shutting down Comprehensive Navigation System...")
                    break

        finally:
            self.cleanup()
            cap.release()
            cv2.destroyAllWindows()

    def audio_worker(self):
        """Enhanced audio worker with 3D spatial audio"""
        engine = pyttsx3.init()
        engine.setProperty('rate', self.user_preferences.get('speech_rate', 140))
        engine.setProperty('volume', self.user_preferences.get('volume', 0.9))
        
        while True:
            try:
                item = self.audio_queue.get(timeout=1)
                if item is None:
                    break
                    
                if len(item) == 4:  # Spatial audio
                    message, priority, direction, distance = item
                    self.play_spatial_audio(message, direction, distance)
                else:  # Regular audio
                    message, priority = item
                    with self.engine_lock:
                        self.is_speaking = True
                        engine.say(message)
                        engine.runAndWait()
                        self.is_speaking = False
                
                self.audio_queue.task_done()
            except queue.Empty:
                continue

    def play_spatial_audio(self, message, direction_angle, distance_factor):
        """Play audio with 3D spatial positioning"""
        try:
            # Convert text to speech and save temporarily
            temp_engine = pyttsx3.init()
            temp_engine.setProperty('rate', self.user_preferences.get('speech_rate', 140))
            temp_file = "temp_audio.wav"
            temp_engine.save_to_file(message, temp_file)
            temp_engine.runAndWait()
            
            # Load and play with spatial positioning
            sound = pygame.mixer.Sound(temp_file)
            
            # Calculate stereo positioning (-1 to 1, left to right)
            stereo_pos = direction_angle / 180.0  # Convert to -1 to 1 range
            
            # Calculate volume based on distance (closer = louder)
            volume = max(0.3, min(1.0, 1.0 / max(1, distance_factor)))
            
            # Apply spatial audio
            left_volume = volume * (1 - max(0, stereo_pos))
            right_volume = volume * (1 + min(0, stereo_pos))
            
            channel = pygame.mixer.Channel(0)
            channel.set_volume(left_volume, right_volume)
            channel.play(sound)
            
            # Wait for completion
            while channel.get_busy():
                time.sleep(0.1)
                
            # Cleanup
            os.remove(temp_file)
            
        except Exception as e:
            # Fallback to regular TTS
            with self.engine_lock:
                self.is_speaking = True
                temp_engine = pyttsx3.init()
                temp_engine.say(message)
                temp_engine.runAndWait()
                self.is_speaking = False

    def voice_command_listener(self):
        """Enhanced voice command recognition"""
        print("🎙️  Voice command listener started")
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        
        while self.voice_commands_active:
            try:
                with self.microphone as source:
                    # Listen for voice with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                
                try:
                    command = self.recognizer.recognize_google(audio).lower()
                    self.process_voice_command(command)
                except sr.UnknownValueError:
                    pass  # Could not understand audio
                except sr.RequestError:
                    pass  # Could not request results
                    
            except sr.WaitTimeoutError:
                pass  # No speech detected
            except Exception as e:
                print(f"Voice command error: {e}")
                time.sleep(1)

    def process_voice_command(self, command):
        """Process recognized voice commands"""
        print(f"🎙️  Voice command: {command}")
        
        if "what's ahead" in command or "scan ahead" in command:
            self.scan_ahead()
        elif "repeat" in command or "say again" in command:
            self.repeat_last_alert()
        elif "quiet mode" in command or "silence" in command:
            self.toggle_quiet_mode()
        elif "scan left" in command:
            self.scan_direction("left")
        elif "scan right" in command:
            self.scan_direction("right")
        elif "emergency" in command or "help" in command:
            self.emergency_alert()
        elif "navigate to" in command or "go to" in command:
            destination = command.replace("navigate to", "").replace("go to", "").strip()
            self.start_navigation(destination)
        elif "where am i" in command or "location" in command:
            self.announce_location()
        elif "learn route" in command:
            self.start_route_learning()
        elif "speed up" in command:
            self.adjust_speech_rate(20)
        elif "slow down" in command:
            self.adjust_speech_rate(-20)
        elif "louder" in command or "volume up" in command:
            self.adjust_volume(0.1)
        elif "quieter" in command or "volume down" in command:
            self.adjust_volume(-0.1)

    def scan_ahead(self):
        """Provide detailed scan of what's ahead"""
        self.audio_queue.put(("Scanning ahead", 2))

    def scan_direction(self, direction):
        """Focus scanning on specific direction"""
        self.audio_queue.put((f"Scanning {direction} side", 2))

    def repeat_last_alert(self):
        """Repeat the last important alert"""
        if hasattr(self, 'last_alert'):
            self.audio_queue.put((self.last_alert, 1))
        else:
            self.audio_queue.put(("No recent alerts to repeat", 3))

    def toggle_quiet_mode(self):
        """Toggle quiet mode on/off"""
        self.quiet_mode = not self.quiet_mode
        status = "enabled" if self.quiet_mode else "disabled"
        self.audio_queue.put((f"Quiet mode {status}", 2))

    def emergency_alert(self):
        """Send emergency alert to contacts"""
        try:
            location_info = self.get_current_location_info()
            emergency_message = f"Emergency alert from navigation system. Location: {location_info}"
            
            # Send to emergency contacts
            for contact in self.emergency_contacts:
                self.send_emergency_sms(contact, emergency_message)
            
            self.audio_queue.put(("Emergency alert sent to contacts", 1))
            
        except Exception as e:
            self.audio_queue.put(("Emergency alert failed", 1))

    def gps_tracker(self):
        """GPS tracking and location updates"""
        while True:
            try:
                # Simulate GPS data (replace with actual GPS API)
                # In real implementation, use GPS hardware or smartphone GPS
                location_data = self.get_gps_data()
                
                if location_data:
                    self.current_location = location_data
                    self.route_history.append({
                        'location': location_data,
                        'timestamp': datetime.now()
                    })
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                print(f"GPS tracking error: {e}")
                time.sleep(10)

    def route_learner(self):
        """Learn and remember frequently used routes"""
        while True:
            try:
                if self.learning_mode and len(self.route_history) > 10:
                    self.analyze_route_patterns()
                
                time.sleep(30)  # Analyze every 30 seconds
                
            except Exception as e:
                print(f"Route learning error: {e}")
                time.sleep(60)

    def init_haptic_device(self):
        """Initialize haptic feedback device"""
        try:
            # Try to connect to haptic device via serial/USB
            # This would connect to devices like Arduino with vibration motors
            # or commercial haptic feedback devices
            ports = ['/dev/ttyUSB0', '/dev/ttyACM0', 'COM3', 'COM4']
            
            for port in ports:
                try:
                    self.haptic_device = serial.Serial(port, 9600)
                    print(f"📳 Haptic device connected on {port}")
                    break
                except:
                    continue
                    
        except Exception as e:
            print(f"📳 Haptic device not available: {e}")

    def send_haptic_feedback(self, objects):
        """Send haptic patterns based on detected objects"""
        if not self.haptic_device:
            return
            
        try:
            # Send haptic pattern for highest priority object
            if objects:
                priority_obj = objects[0]
                pattern = priority_obj['haptic_pattern']
                direction = priority_obj['direction']
                distance = priority_obj['distance']
                
                # Create haptic command
                haptic_cmd = f"{pattern},{direction},{distance}\n"
                self.haptic_device.write(haptic_cmd.encode())
                
        except Exception as e:
            print(f"Haptic feedback error: {e}")

    def generate_enhanced_alerts(self, objects, last_spoken, cooldown_priorities, current_time):
        """Generate enhanced alerts with spatial audio and context"""
        
        for obj in objects[:2]:  # Top 2 most important objects
            priority = obj['priority']
            distance = obj['distance']
            
            if distance > obj['alert_distance']:
                continue
            
            # Enhanced key with location context
            location_key = f"{self.current_location['lat']:.3f},{self.current_location['lng']:.3f}" if self.current_location else "unknown"
            key = f"{obj['label']}-{obj['direction']}-{distance//30*30}-{location_key}"
            
            # Adaptive cooldown based on user preferences and object priority
            base_cooldown = cooldown_priorities.get(priority, 5)
            cooldown = base_cooldown * self.user_preferences.get('alert_frequency', 1.0)
            
            if key not in last_spoken or (current_time - last_spoken[key]) > cooldown:
                # Generate contextual message with location awareness
                message = self.create_contextual_message(obj)
                
                # Calculate spatial audio parameters
                direction_angle = self.calculate_direction_angle(obj['direction'])
                distance_factor = min(obj['distance'] / 100, 3.0)
                
                if not self.is_speaking or priority == 1:
                    # Use spatial audio for enhanced experience
                    self.audio_queue.put((message, priority, direction_angle, distance_factor))
                    last_spoken[key] = current_time
                    self.last_alert = message
                    print(f"🔊 Priority {priority}: {message}")

    def create_contextual_message(self, obj):
        """Create contextual messages with location and learned information"""
        label = obj['label']
        distance = obj['distance']
        direction = obj['direction']
        
        # Add location context if available
        location_context = self.get_location_context()
        
        # Check if this is a known obstacle in this location
        familiar_obstacle = self.check_familiar_obstacle(obj)
        
        # Base message
        message = self.create_spatial_message(obj)
        
        # Add context
        if location_context:
            message += f" {location_context}"
            
        if familiar_obstacle:
            message += f" - this is a known obstacle here"
            
        return message

    def create_spatial_message(self, obj):
        """Enhanced spatial message creation"""
        label = obj['label']
        distance = obj['distance']
        direction = obj['direction']
        
        # More natural distance descriptions
        if distance < 30:
            distance_desc = "right in front of you"
        elif distance < 60:
            distance_desc = "very close"
        elif distance < 100:
            distance_desc = "close"
        elif distance < 200:
            distance_desc = "nearby"
        else:
            distance_desc = f"{distance} centimeters away"
        
        # Enhanced object descriptions with context
        if label == "person":
            if distance < 100:
                return f"Person {distance_desc} on your {direction} - please be careful"
            else:
                return f"Person approaching on your {direction}"
        elif label in ["car", "truck", "bus"]:
            return f"Vehicle {distance_desc} on your {direction} - caution advised"
        elif label == "traffic light":
            return f"Traffic light detected {direction} - check crossing signal"
        elif label == "stop sign":
            return f"Stop sign on your {direction}"
        else:
            return f"{label} {distance_desc} on your {direction}"

    def calculate_direction_angle(self, direction):
        """Convert direction to angle for spatial audio"""
        direction_angles = {
            "far left": -90,
            "left": -60,
            "slightly left": -30,
            "center": 0,
            "slightly right": 30,
            "right": 60,
            "far right": 90
        }
        return direction_angles.get(direction, 0)

    def get_location_context(self):
        """Get contextual information about current location"""
        if not self.current_location:
            return ""
            
        # Check POI database for nearby landmarks
        nearby_poi = self.find_nearby_poi(self.current_location)
        
        if nearby_poi:
            return f"near {nearby_poi['name']}"
        
        return ""

    def check_familiar_obstacle(self, obj):
        """Check if this obstacle is familiar in this location"""
        if not self.current_location:
            return False
            
        location_key = f"{self.current_location['lat']:.3f},{self.current_location['lng']:.3f}"
        
        if location_key in self.location_memory:
            for memory_obj in self.location_memory[location_key]:
                if (memory_obj['label'] == obj['label'] and 
                    abs(memory_obj['distance'] - obj['distance']) < 50):
                    return True
        
        return False

    def update_location_memory(self, objects):
        """Update memory of objects at current location"""
        if not self.current_location:
            return
            
        location_key = f"{self.current_location['lat']:.3f},{self.current_location['lng']:.3f}"
        
        # Store significant objects
        for obj in objects:
            if obj['priority'] <= 2:  # Only remember important objects
                memory_entry = {
                    'label': obj['label'],
                    'distance': obj['distance'],
                    'direction': obj['direction'],
                    'timestamp': datetime.now().isoformat()
                }
                
                self.location_memory[location_key].append(memory_entry)
                
                # Keep only recent memories (last 10 per location)
                if len(self.location_memory[location_key]) > 10:
                    self.location_memory[location_key] = self.location_memory[location_key][-10:]

    def analyze_movement_patterns(self, current_objects, object_tracking, current_time):
        """Analyze object movement for predictive alerts"""
        for obj in current_objects:
            if obj['priority'] == 1:  # Only track high-priority moving objects
                obj_id = f"{obj['label']}-{obj['direction']}"
                
                object_tracking[obj_id].append({
                    'distance': obj['distance'],
                    'time': current_time,
                    'center_x': obj['center_x']
                })
                
                # Keep only recent tracking data
                if len(object_tracking[obj_id]) > 5:
                    object_tracking[obj_id] = object_tracking[obj_id][-5:]
                
                # Analyze movement trend
                if len(object_tracking[obj_id]) >= 3:
                    distances = [entry['distance'] for entry in object_tracking[obj_id]]
                    if len(distances) >= 2:
                        # Check if object is approaching
                        if distances[-1] < distances[-2] and distances[-2] < distances[0]:
                            approach_rate = (distances[0] - distances[-1]) / (current_time - object_tracking[obj_id][0]['time'])
                            if approach_rate > 20:  # Approaching at >20 cm/s
                                self.audio_queue.put((f"Fast approaching {obj['label']} on your {obj['direction']}", 1))

    def get_gps_data(self):
        """Get current GPS coordinates (placeholder for real GPS)"""
        # In real implementation, integrate with:
        # - Android/iOS location services
        # - GPS hardware modules
        # - Smartphone GPS via API
        
        # Placeholder - return mock coordinates
        return {
            'lat': 37.7749,
            'lng': -122.4194,
            'accuracy': 5,
            'timestamp': datetime.now()
        }

    def find_nearby_poi(self, location):
        """Find nearby points of interest"""
        # In real implementation, query maps API or local POI database
        # For now, return mock data
        mock_pois = [
            {'name': 'Main Street Station', 'lat': 37.7749, 'lng': -122.4194},
            {'name': 'Central Park', 'lat': 37.7750, 'lng': -122.4195},
        ]
        
        # Simple distance calculation (in real app, use proper geo distance)
        for poi in mock_pois:
            if (abs(poi['lat'] - location['lat']) < 0.001 and 
                abs(poi['lng'] - location['lng']) < 0.001):
                return poi
        
        return None

    def start_navigation(self, destination):
        """Start GPS navigation to destination"""
        self.audio_queue.put((f"Starting navigation to {destination}", 2))
        # In real implementation:
        # - Query maps API for route
        # - Provide turn-by-turn directions
        # - Monitor progress

    def announce_location(self):
        """Announce current location"""
        if self.current_location:
            nearby_poi = self.find_nearby_poi(self.current_location)
            if nearby_poi:
                self.audio_queue.put((f"You are near {nearby_poi['name']}", 2))
            else:
                self.audio_queue.put((f"You are at coordinates {self.current_location['lat']:.4f}, {self.current_location['lng']:.4f}", 2))
        else:
            self.audio_queue.put(("Location not available", 2))

    def send_emergency_sms(self, contact, message):
        """Send emergency SMS (placeholder)"""
        # In real implementation, integrate with SMS API or smartphone
        print(f"Emergency SMS to {contact}: {message}")

    def adjust_speech_rate(self, change):
        """Adjust speech rate"""
        current_rate = self.user_preferences.get('speech_rate', 140)
        new_rate = max(80, min(200, current_rate + change))
        self.user_preferences['speech_rate'] = new_rate
        self.save_user_preferences()
        self.audio_queue.put((f"Speech rate adjusted to {new_rate}", 3))

    def adjust_volume(self, change):
        """Adjust volume"""
        current_volume = self.user_preferences.get('volume', 0.9)
        new_volume = max(0.1, min(1.0, current_volume + change))
        self.user_preferences['volume'] = new_volume
        self.save_user_preferences()
        self.audio_queue.put((f"Volume adjusted to {int(new_volume * 100)} percent", 3))

    def start_route_learning(self):
        """Start learning a new route"""
        self.learning_mode = True
        self.route_history.clear()
        self.audio_queue.put(("Route learning started - I'll remember this path", 2))

    def analyze_route_patterns(self):
        """Analyze route history to identify common paths"""
        if len(self.route_history) < 5:
            return
            
        # Analyze patterns in route history
        route_segments = []
        for i in range(len(self.route_history) - 1):
            segment = {
                'start': self.route_history[i]['location'],
                'end': self.route_history[i + 1]['location'],
                'time_taken': (self.route_history[i + 1]['timestamp'] - 
                              self.route_history[i]['timestamp']).total_seconds()
            }
            route_segments.append(segment)
        
        # Store learned route patterns
        self.learned_routes.append({
            'segments': route_segments,
            'learned_date': datetime.now().isoformat(),
            'frequency': 1
        })
        
        self.save_learned_routes()

    def provide_location_context(self):
        """Provide contextual information about current location"""
        if not self.current_location:
            return
            
        context_info = []
        
        # Check for intersection
        if self.is_at_intersection():
            context_info.append("intersection ahead")
        
        # Check for building entrances
        nearby_entrances = self.detect_building_entrances()
        if nearby_entrances:
            context_info.append(f"building entrance on your {nearby_entrances}")
        
        # Check for crosswalks
        if self.is_near_crosswalk():
            context_info.append("pedestrian crossing nearby")
        
        # Announce context if significant
        if context_info and not self.quiet_mode and not self.is_speaking:
            message = "Navigation context: " + ", ".join(context_info)
            self.audio_queue.put((message, 3))

    def is_at_intersection(self):
        """Detect if user is approaching an intersection"""
        # In real implementation, use GPS data and map APIs
        # For now, return mock detection
        return False

    def detect_building_entrances(self):
        """Detect nearby building entrances"""
        # Mock implementation - in real app, use computer vision + GPS
        return None

    def is_near_crosswalk(self):
        """Detect if near a crosswalk"""
        # Mock implementation - in real app, use visual recognition
        return False

    def get_current_location_info(self):
        """Get formatted current location information"""
        if self.current_location:
            nearby_poi = self.find_nearby_poi(self.current_location)
            if nearby_poi:
                return f"Near {nearby_poi['name']}"
            else:
                return f"Lat: {self.current_location['lat']:.4f}, Lng: {self.current_location['lng']:.4f}"
        return "Location unknown"

    def load_user_preferences(self):
        """Load user preferences from file"""
        try:
            with open('user_preferences.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'speech_rate': 140,
                'volume': 0.9,
                'alert_frequency': 1.0,
                'focal_length': 615,
                'preferred_voice': 'default',
                'haptic_intensity': 0.8
            }

    def save_user_preferences(self):
        """Save user preferences to file"""
        try:
            with open('user_preferences.json', 'w') as f:
                json.dump(self.user_preferences, f, indent=2)
        except Exception as e:
            print(f"Failed to save preferences: {e}")

    def load_emergency_contacts(self):
        """Load emergency contacts"""
        try:
            with open('emergency_contacts.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return [
                {'name': 'Emergency Contact 1', 'phone': '+1234567890'},
                {'name': 'Emergency Contact 2', 'phone': '+0987654321'}
            ]

    def load_learned_routes(self):
        """Load previously learned routes"""
        try:
            with open('learned_routes.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return []

    def save_learned_routes(self):
        """Save learned routes to file"""
        try:
            with open('learned_routes.pkl', 'wb') as f:
                pickle.dump(self.learned_routes, f)
        except Exception as e:
            print(f"Failed to save learned routes: {e}")

    def load_poi_database(self):
        """Load points of interest database"""
        try:
            with open('poi_database.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def cleanup(self):
        """Clean up resources and save data"""
        print("🧹 Cleaning up...")
        
        # Stop all background processes
        self.voice_commands_active = False
        self.learning_mode = False
        
        # Save current session data
        self.save_user_preferences()
        self.save_learned_routes()
        
        # Save location memory
        try:
            with open('location_memory.pkl', 'wb') as f:
                pickle.dump(dict(self.location_memory), f)
        except Exception as e:
            print(f"Failed to save location memory: {e}")
        
        # Close haptic device
        if self.haptic_device:
            try:
                self.haptic_device.close()
            except:
                pass
        
        # Stop audio queue
        self.audio_queue.put((None, 0))
        
        print("✅ Cleanup completed")

    def calibrate_system(self):
        """System calibration for individual user"""
        self.audio_queue.put(("Starting system calibration", 2))
        
        # Voice calibration
        self.audio_queue.put(("Please say 'calibration test' when ready", 2))
        # Wait for voice response and adjust recognition sensitivity
        
        # Distance calibration
        self.audio_queue.put(("Place a known object 1 meter in front of camera", 2))
        # Capture frame and calibrate distance calculations
        
        # Audio preference calibration
        self.audio_queue.put(("Testing audio levels - say stop when comfortable", 2))
        # Gradually increase volume until user says stop
        
        self.audio_queue.put(("Calibration completed and saved", 2))

    def export_session_data(self):
        """Export session data for analysis or sharing"""
        session_data = {
            'route_history': list(self.route_history),
            'location_memory': dict(self.location_memory),
            'user_preferences': self.user_preferences,
            'session_timestamp': datetime.now().isoformat()
        }
        
        filename = f"session_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            print(f"📊 Session data exported to {filename}")
        except Exception as e:
            print(f"Failed to export session data: {e}")

    def load_session_data(self, filename):
        """Load previously exported session data"""
        try:
            with open(filename, 'r') as f:
                session_data = json.load(f)
            
            # Restore relevant data
            if 'user_preferences' in session_data:
                self.user_preferences.update(session_data['user_preferences'])
            
            if 'location_memory' in session_data:
                for location, memories in session_data['location_memory'].items():
                    self.location_memory[location].extend(memories)
            
            self.audio_queue.put(("Previous session data loaded", 2))
            
        except Exception as e:
            print(f"Failed to load session data: {e}")

    def get_system_status(self):
        """Get comprehensive system status"""
        status = {
            'gps_active': self.current_location is not None,
            'voice_commands': self.voice_commands_active,
            'haptic_feedback': self.haptic_device is not None,
            'quiet_mode': self.quiet_mode,
            'learning_mode': self.learning_mode,
            'routes_learned': len(self.learned_routes),
            'locations_remembered': len(self.location_memory),
            'speech_rate': self.user_preferences.get('speech_rate', 140),
            'volume': self.user_preferences.get('volume', 0.9)
        }
        
        return status

    def announce_system_status(self):
        """Announce current system status"""
        status = self.get_system_status()
        
        status_message = f"System status: GPS {'active' if status['gps_active'] else 'inactive'}, "
        status_message += f"Voice commands {'on' if status['voice_commands'] else 'off'}, "
        status_message += f"Haptic feedback {'available' if status['haptic_feedback'] else 'unavailable'}, "
        status_message += f"Learning mode {'on' if status['learning_mode'] else 'off'}"
        
        self.audio_queue.put((status_message, 2))

# Additional utility functions for mobile app integration

