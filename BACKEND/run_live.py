"""
Simple launcher for live emotion detection
Just run this file to start the system
"""
from live_emotion import LiveEmotionDetector

def main():
    print("🎭 Live Emotion-Based Music System")
    print("=" * 40)
    
    detector = LiveEmotionDetector()
    
    # Quick setup check
    print("Checking setup...")
    if not detector.start_camera():
        print("❌ Webcam not available")
        return
    detector.stop()  # Close camera for now
    
    print("✅ Webcam ready")
    print("✅ Emotion detector loaded")
    
    print("\nStarting live detection...")
    print("- Detects emotion every 5 seconds")
    print("- Switches music when songs end")  
    print("- Press 'q' in video window to quit")
    
    try:
        detector.run_live_detection()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()