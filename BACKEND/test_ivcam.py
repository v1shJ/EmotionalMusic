#!/usr/bin/env python3
"""
iVCam Connection Test Script
Helps diagnose and fix iVCam connection issues
"""

import cv2
import numpy as np

def test_camera_index(index):
    """Test a specific camera index"""
    print(f"\n🔍 Testing camera index {index}...")
    
    # Try with DirectShow backend (recommended for Windows)
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"❌ Camera {index}: Cannot open")
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📐 Camera {index}: {width}x{height} @ {fps} FPS")
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"❌ Camera {index}: Cannot read frame")
        cap.release()
        return False
    
    if frame.size == 0:
        print(f"❌ Camera {index}: Empty frame")
        cap.release()
        return False
    
    print(f"✅ Camera {index}: Frame read successful ({frame.shape})")
    
    # Test multiple frames for stability
    stable_frames = 0
    for i in range(5):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            stable_frames += 1
    
    print(f"📊 Camera {index}: {stable_frames}/5 stable frames")
    
    cap.release()
    
    return stable_frames >= 3

def main():
    print("🧪 iVCam Connection Diagnostic Tool")
    print("=" * 40)
    
    working_cameras = []
    
    # Test camera indices 0-5
    for i in range(6):
        if test_camera_index(i):
            working_cameras.append(i)
    
    print(f"\n📊 Summary:")
    print(f"Working cameras: {working_cameras}")
    
    if len(working_cameras) == 0:
        print("\n❌ No working cameras found!")
        print("\n💡 Troubleshooting steps:")
        print("1. Make sure iVCam is running on both iPhone and PC")
        print("2. Check that both show 'Connected' status")
        print("3. Try restarting iVCam PC client as Administrator")
        print("4. Check Windows Camera privacy settings")
        print("5. Try different USB cable or WiFi connection")
        
    elif len(working_cameras) == 1 and working_cameras[0] == 0:
        print(f"\n💻 Only computer webcam (index 0) found")
        print("📱 iVCam not detected. Check connection steps above.")
        
    else:
        print(f"\n✅ Multiple cameras detected!")
        if 0 in working_cameras:
            print("💻 Computer webcam: Index 0")
        
        ivcam_indices = [i for i in working_cameras if i > 0]
        if ivcam_indices:
            print(f"📱 iVCam likely at: Index {ivcam_indices}")
            
    print(f"\n🔧 Quick Test:")
    if working_cameras:
        test_index = working_cameras[-1]  # Use highest index (likely iVCam)
        print(f"Testing camera {test_index} with live preview...")
        
        cap = cv2.VideoCapture(test_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            print("📹 Press 'q' to quit, 'ESC' to exit")
            
            while True:
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Mirror the frame like your emotion detector
                    frame = cv2.flip(frame, 1)
                    cv2.imshow(f'Camera {test_index} Test', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        break
                else:
                    print("⚠️ Frame read failed")
                    break
                    
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Test completed")

if __name__ == "__main__":
    main()