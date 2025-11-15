# Emotion Detection with Face Recognition

A minimal yet powerful emotion detection pipeline using OpenCV and a trained CNN model. Supports image files, real-time webcam analysis, and even iPhone camera integration. Includes optional Spotify automation that changes playlists based on detected emotions. Clean architecture, fast setup, and easy to extend.

## Files

### Core Files (Required)
- **`emotion_detector.py`** - Main emotion detection with face recognition
- **`spotify.py`** - Spotify integration for emotion-based music
- **`live_emotion.py`** - Live webcam emotion detection system
- **`run_live.py`** - Simple launcher for live detection
- **`requirements.txt`** - Essential dependencies only

### Legacy Files (Optional)
- **`emotionDetect.py`** - Your original emotion detection (for reference)
- **`callback_server.py`** - Spotify authentication helper

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test static image emotion detection:**
   ```python
   from emotion_detector import EmotionDetector
   
   detector = EmotionDetector()
   results = detector.process_image('your_image.jpg', 'output.jpg')
   print(f"Detected {results['faces_detected']} faces")
   ```

3. **Run live emotion detection:**
   ```bash
   python run_live.py
   ```

   **For iPhone Camera (Higher Quality):**
   - Install iVCam on your iPhone from the App Store
   - Install iVCam software on your Windows computer
   - Connect your iPhone and computer to the same Wi-Fi network
   - Launch iVCam on both devices to establish connection
   - Run the emotion detection - it will automatically detect and use your iPhone camera
4. **Run static image with Spotify:**
   ```bash
   python spotify.py
   ```

## What It Does

### Static Image Analysis
1. **Detects faces** using OpenCV (finds bounding boxes)
2. **Analyzes emotion** using YOUR trained CNN model 
3. **Draws bounding boxes** around faces with emotion labels
4. **Plays music** based on detected emotion

### Live Webcam System
1. **Monitors Spotify** for song changes automatically
2. **Detects emotion** only when a song changes or is skipped
3. **Switches playlists** only if emotion is different from last time
4. **No action** if emotion remains the same
5. **Phone Camera Support** - Use your iPhone camera via iVCam for higher quality emotion detection

## Simple Usage

### Static Image Analysis
```python
from emotion_detector import predict_emotion

# Simple function (backward compatible)
result = predict_emotion('image.jpg')
print(f"Emotion: {result['emotion']} (confidence: {result['confidence']:.2f})")
print(f"Faces found: {result['faces_detected']}")
```

### Live Emotion Detection
```python
from live_emotion import LiveEmotionDetector

detector = LiveEmotionDetector()
detector.run_live_detection()  # Starts webcam + Spotify monitoring
```

**Or just run:**
```bash
python run_live.py
```

## iPhone Camera Integration

For superior emotion detection quality, you can use your iPhone camera instead of your computer's webcam:

### Setup Steps:
1. **Install iVCam app** on your iPhone (free from App Store)
2. **Download iVCam software** for Windows from e2esoft.com
3. **Connect both devices** to the same Wi-Fi network
4. **Launch iVCam** on both iPhone and Windows
5. **Verify connection** - iVCam should show "Connected" status
6. **Run emotion detection** - the system will automatically detect and use your iPhone camera

### Benefits:
- **Higher resolution** camera for better face detection
- **Better lighting adaptation** with iPhone's advanced camera sensors  
- **Improved emotion accuracy** due to superior image quality
- **Flexible positioning** - place your iPhone at optimal angles

### Troubleshooting:
- Ensure both devices are on the same network
- Restart iVCam if connection fails
- Check that no other apps are using the camera
- Try different camera indices if detection fails

That's it! Clean and minimal with professional-grade camera support.
