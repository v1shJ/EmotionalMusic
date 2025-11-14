"""
Live Emotion-Based Music Switching System
Logs emotions every 5 seconds, switches playlists only when songs change.
Best of both worlds: accurate emotion tracking + non-disruptive switching.
"""
import cv2
import time
import threading
from emotion_detector import EmotionDetector, predict_emotion
from spotify import sp, play_playlist, get_active_device, EMOTION_BASED_URI
import numpy as np

class LiveEmotionDetector:
    def __init__(self):
        self.detector = EmotionDetector()
        self.current_logged_emotion = None  # Current emotion from 5-second logging
        self.last_playlist_emotion = None  # Last emotion we switched playlist to
        self.emotion_log = []  # Log of emotions with timestamps
        self.last_switch_time = 0  # Prevent rapid switching
        self.switch_cooldown = 3  # Short cooldown to prevent rapid switching
        self.emotion_check_interval = 5  # Log emotion every 5 seconds
        self.last_emotion_check = 0  # Track when we last logged emotion
        self.running = False
        self.cap = None
        
        # FPS monitoring
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def start_camera(self, use_phone=False, camera_source=None):
        """Initialize webcam or phone camera with maximum performance settings"""
        
        if use_phone:
            if camera_source == "ivcam":
                # iVCam creates a virtual webcam device
                print("📱 Connecting to iPhone via iVCam...")
                # iVCam typically appears as camera index 1 or 2
                camera_found = False
                for camera_id in [1, 2, 3, 4]:
                    print(f"🔍 Trying camera index {camera_id}...")
                    self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)  # Use DirectShow for Windows
                    if self.cap.isOpened():
                        # Test if we can read a frame
                        ret, frame = self.cap.read()
                        if ret and frame is not None and frame.size > 0:
                            print(f"✅ iPhone camera found at index {camera_id}!")
                            # Set moderate resolution for stability
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                            # Test another frame to ensure stability
                            ret2, frame2 = self.cap.read()
                            if ret2 and frame2 is not None:
                                camera_found = True
                                break
                        self.cap.release()
                    
                if not camera_found:
                    print("❌ iVCam camera not found or unstable")
                    print("💡 Troubleshooting:")
                    print("   1. Close and restart iVCam on iPhone and PC")
                    print("   2. Check 'Connected' status in both apps")
                    print("   3. Try different USB cable or WiFi connection")
                    print("   4. Restart iVCam PC client as Administrator")
                    print("🔄 Falling back to computer webcam...")
                    self.cap = cv2.VideoCapture(0)
            
            elif camera_source and camera_source.startswith("http"):
                # IP-based camera (for other apps)
                print(f"🔗 Connecting to phone camera at: {camera_source}")
                video_url = f"{camera_source}/video"
                self.cap = cv2.VideoCapture(video_url)
                
                if not self.cap.isOpened():
                    print("❌ Failed to connect to IP camera")
                    print("🔄 Falling back to computer webcam...")
                    self.cap = cv2.VideoCapture(0)
                else:
                    print("✅ IP camera connected successfully!")
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            # Use computer webcam
            print("💻 Using computer webcam")
            self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Could not open any camera")
            return False
        
        # Set camera properties for maximum performance
        self.cap.set(cv2.CAP_PROP_FPS, 30)            # Reasonable FPS for emotion detection
        if not use_phone:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Computer webcam resolution
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Optimize for performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Reduce buffer lag
        self.cap.set(cv2.CAP_PROP_FPS, 30)            # Set reasonable FPS
        
        # Get actual camera capabilities
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if use_phone:
            if camera_source == "ivcam":
                camera_type = "📱 iPhone Camera (iVCam)"
            else:
                camera_type = "📱 Phone Camera (IP)"
        else:
            camera_type = "💻 Computer Webcam"
            
        print(f"✅ {camera_type} initialized")
        print(f"📹 Camera FPS: {actual_fps}")
        print(f"📐 Resolution: {actual_width}x{actual_height}")
        
        return True
        return True
    
    def detect_current_emotion(self):
        """Detect emotion at the current moment (for song change events)"""
        if not self.cap or not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # Mirror the frame horizontally (like a selfie camera)
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB for emotion detector
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.detector.detect_faces(frame_rgb)
        
        if len(faces) > 0:
            # Get the largest face (closest to camera)
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, w, h = largest_face
            
            # Extract face region from RGB frame
            from PIL import Image
            face_region = Image.fromarray(frame_rgb[y:y+h, x:x+w])
            
            # Predict emotion
            emotion, confidence = self.detector.predict_emotion(face_region)
            
            print(f"🎭 Detected emotion: {emotion} (confidence: {confidence:.2f})")
            return emotion.lower()
        else:
            print("⚠️ No face detected, using neutral")
            return "neutral"
    
    def capture_and_display_live(self):
        """Continuous live video feed with emotion detection"""
        if not self.cap or not self.cap.isOpened():
            return None, None
            
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None
        
        # Validate frame dimensions and data
        if frame.size == 0 or len(frame.shape) != 3:
            print("⚠️ Invalid frame received, skipping...")
            return None, None
            
        # Check if frame has proper dimensions
        height, width = frame.shape[:2]
        if height < 10 or width < 10:
            print("⚠️ Frame too small, skipping...")
            return None, None
            
        # Mirror the frame horizontally (like a selfie camera)
        try:
            frame = cv2.flip(frame, 1)
        except Exception as e:
            print(f"⚠️ Error flipping frame: {e}")
            return None, None
            
        # Convert BGR to RGB for emotion detector
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.detector.detect_faces(frame_rgb)
        
        current_emotion = None
        
        if len(faces) > 0:
            # Process all faces
            for i, (x, y, w, h) in enumerate(faces):
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Extract face region from RGB frame
                from PIL import Image
                face_region = Image.fromarray(frame_rgb[y:y+h, x:x+w])
                
                # Predict emotion
                emotion, confidence = self.detector.predict_emotion(face_region)
                
                # Add emotion text on frame
                cv2.putText(frame, f"Face {i+1}: {emotion} ({confidence:.2f})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Use the first face's emotion as current emotion
                if i == 0:
                    current_emotion = emotion.lower()
        else:
            # Add "No face detected" text on frame
            cv2.putText(frame, "No face detected", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add system status text with FPS monitoring
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Current Emotion: {self.current_logged_emotion or 'None'}", 
                   (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Playlist Emotion: {self.last_playlist_emotion or 'None'}", 
                   (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame, current_emotion
    
    def log_emotion_periodically(self):
        """Log emotion every 5 seconds (no playlist switching here)"""
        current_time = time.time()
        
        # Check if it's time for emotion logging
        if current_time - self.last_emotion_check < self.emotion_check_interval:
            return
        
        print(f"📝 Logging emotion (every {self.emotion_check_interval}s)...")
        
        # Detect current emotion
        current_emotion = self.detect_current_emotion()
        self.last_emotion_check = current_time
        
        if current_emotion:
            # Log the emotion with timestamp
            self.emotion_log.append({
                'emotion': current_emotion,
                'timestamp': current_time
            })
            
            # Keep only last 10 emotion logs (50 seconds of history)
            if len(self.emotion_log) > 10:
                self.emotion_log.pop(0)
            
            # Update current logged emotion
            self.current_logged_emotion = current_emotion
            
            print(f"� Logged emotion: {current_emotion} (Total logs: {len(self.emotion_log)})")
        else:
            print("⚠️ No face detected for emotion logging")
    
    def switch_playlist_on_song_change(self, current_track_id, last_track_id):
        """Switch playlist when song changes, using latest logged emotion"""
        if not self.current_logged_emotion:
            print("⚠️ No logged emotion available, skipping playlist switch")
            return
        
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_switch_time < self.switch_cooldown:
            print(f"⏳ Switch cooldown active ({self.switch_cooldown - (current_time - self.last_switch_time):.1f}s remaining)")
            return
        
        # Only switch if emotion is different from current playlist
        if self.current_logged_emotion == self.last_playlist_emotion:
            print(f"� Emotion unchanged ({self.current_logged_emotion}), keeping current playlist")
            return
        
        print(f"� Song changed! Switching playlist based on latest emotion: {self.current_logged_emotion}")
        print(f"🔄 Emotion changed from {self.last_playlist_emotion} → {self.current_logged_emotion}")
        
        # Get active device and switch playlist
        device_id = get_active_device(sp)
        if device_id:
            playlist_uri = self.get_playlist_for_emotion(self.current_logged_emotion)
            play_playlist(sp, device_id, playlist_uri)
            
            # Update tracking
            self.last_playlist_emotion = self.current_logged_emotion
            self.last_switch_time = current_time
            
            print(f"✅ Switched to {self.current_logged_emotion} playlist!")
    
    def get_playlist_for_emotion(self, emotion):
        """Get playlist URI for given emotion"""
        if emotion in EMOTION_BASED_URI:
            return EMOTION_BASED_URI[emotion]
        return EMOTION_BASED_URI["neutral"]
    
    def monitor_spotify_and_emotions(self):
        """Monitor Spotify for song changes and log emotions every 5 seconds"""
        last_track_id = None
        
        while self.running:
            try:
                # Check if Spotify is playing
                current = sp.current_playback()
                
                if current and current['is_playing']:
                    current_track_id = current['item']['id']
                    current_track = current['item']['name']
                    current_artist = current['item']['artists'][0]['name']
                    
                    # Log emotion every 5 seconds (independent of song changes)
                    self.log_emotion_periodically()
                    
                    # Check if song changed
                    if last_track_id and last_track_id != current_track_id:
                        print(f"🎵 Song changed: {current_artist} - {current_track}")
                        
                        # Switch playlist based on latest logged emotion
                        self.switch_playlist_on_song_change(current_track_id, last_track_id)
                    
                    last_track_id = current_track_id
                    
                else:
                    print("⏸️ Spotify not playing, but still logging emotions...")
                    # Still log emotions even when music isn't playing
                    self.log_emotion_periodically()
                    
            except Exception as e:
                print(f"Spotify monitoring error: {e}")
            
            time.sleep(1)  # Check every 1 second
    
    def run_live_detection(self):
        """Main function to run live emotion-based music switching with continuous video"""
        print("🚀 Starting live emotion-based music system...")
        
        # Camera should already be initialized by caller
        if not self.cap or not self.cap.isOpened():
            print("❌ Camera not initialized. Call start_camera() first.")
            return
        
        self.running = True
        
        # Start Spotify monitoring in separate thread
        spotify_thread = threading.Thread(target=self.monitor_spotify_and_emotions, daemon=True)
        spotify_thread.start()
        
        print("🎵 System active! Continuous video feed with smart emotion logging.")
        print("📹 Live video window will show with face detection boxes")
        print("📝 Emotion logging every 5 seconds")
        print("🎵 Playlist switches when songs change (using latest logged emotion)")
        print("🔄 Press 'q' in video window or Ctrl+C to quit")
        
        consecutive_failures = 0
        max_failures = 10
        
        try:
            while self.running:
                # Continuous live video feed with FPS monitoring
                frame, current_emotion = self.capture_and_display_live()
                
                # Update FPS counter
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.fps_start_time >= 1.0:  # Update FPS every second
                    self.current_fps = self.frame_count / (current_time - self.fps_start_time)
                    self.frame_count = 0
                    self.fps_start_time = current_time
                
                if frame is not None:
                    consecutive_failures = 0  # Reset failure counter
                    cv2.imshow('Live Emotion Detection', frame)
                    
                    # Check for quit command (minimal delay for max FPS)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print(f"❌ Too many consecutive frame failures ({max_failures})")
                        print("💡 Camera may have disconnected. Please restart the program.")
                        break
                    
                    # Small delay when frames fail to avoid busy waiting
                    time.sleep(0.1)
                
                # Remove artificial delay for maximum FPS when frames are good
                # time.sleep(0.1) - REMOVED for max performance
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping live detection...")
        
        finally:
            self.stop()
    
    def stop(self):
        """Clean up resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Live detection stopped")

def run_simple_live_detection():
    """Simple function to start live detection with computer webcam"""
    live_detector = LiveEmotionDetector()
    if live_detector.start_camera(use_phone=False, camera_source=None):
        live_detector.run_live_detection()

def run_iphone_camera_detection():
    """Function to start live detection with iPhone via iVCam"""
    live_detector = LiveEmotionDetector()
    if live_detector.start_camera(use_phone=True, camera_source="ivcam"):
        live_detector.run_live_detection()

def run_ip_camera_detection(phone_ip):
    """Function to start live detection with IP camera"""
    live_detector = LiveEmotionDetector()
    if live_detector.start_camera(use_phone=True, camera_source=phone_ip):
        live_detector.run_live_detection()

if __name__ == "__main__":
    print("=== Live Emotion-Based Music Switching ===")
    print("This will:")
    print("1. Show continuous live video feed with face detection")
    print("2. Monitor Spotify for song changes in background")
    print("3. Switch playlist when emotion changes (with cooldown)")
    print("4. Display green boxes around detected faces")
    
    # Camera selection
    print("\n📷 Camera Options:")
    print("1. Computer webcam")
    print("2. iPhone camera (via iVCam)")
    print("3. Android camera (via IP Webcam)")
    
    camera_choice = input("Choose camera (1, 2, or 3): ").strip()
    
    if camera_choice == "2":
        print("\n📱 iPhone Camera Setup (iVCam):")
        print("1. Install 'iVCam' app from App Store")
        print("2. Install iVCam PC client on your computer")
        print("3. Connect both devices to same WiFi")
        print("4. Start iVCam on iPhone and PC")
        print("5. Grant camera permission when prompted")
        
        print("\nMake sure:")
        print("- iVCam app is running on iPhone")
        print("- iVCam PC client is connected")
        print("- Camera shows 'Connected' status")
        print("- Spotify is playing music")
        
        input("\nPress Enter to start...")
        
        try:
            run_iphone_camera_detection()
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure iVCam is properly connected.")
    
    elif camera_choice == "3":
        print("\n📱 Android Camera Setup (IP Webcam):")
        print("1. Install 'IP Webcam' app on your Android phone")
        print("2. Start the server in the app")
        print("3. Make sure phone and computer are on same WiFi")
        print("4. Note the IP address shown in the app")
        print("   Example: http://192.168.1.100:8080")
        
        phone_ip = input("Enter phone IP address (with http://): ").strip()
        
        if not phone_ip.startswith("http://"):
            print("❌ IP should start with http://")
            phone_ip = "http://" + phone_ip
        
        print(f"\n🔗 Will connect to: {phone_ip}")
        print("Make sure:")
        print("- IP Webcam app is running")
        print("- Phone and computer on same WiFi")
        print("- Spotify is playing music")
        
        input("\nPress Enter to start...")
        
        try:
            run_ip_camera_detection(phone_ip)
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure IP Webcam is running and IP address is correct.")
    
    else:
        print("\nMake sure:")
        print("- Computer webcam is connected")
        print("- Spotify is playing music")
        print("- You're authenticated with Spotify")
        
        input("\nPress Enter to start...")
        
        try:
            run_simple_live_detection()
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure all requirements are installed and Spotify is set up properly.")