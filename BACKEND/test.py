"""
Simple test script for the streamlined emotion detection system
"""
from emotion_detector import EmotionDetector, predict_emotion
from live_emotion import LiveEmotionDetector
import os

def test_static_system():
    print("=== Testing Static Image Emotion Detection ===\n")
    
    # Test image path
    test_image = '../TestingImages/neutral.jpeg'
    
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        print("Please update the path to a valid test image.")
        return False
    
    print(f"Testing with image: {test_image}")
    
    # Test 1: Simple function (backward compatible)
    print("\n1. Simple prediction (backward compatible):")
    result = predict_emotion(test_image)
    print(f"   Emotion: {result['emotion']}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Faces detected: {result['faces_detected']}")
    
    # Test 2: Detailed analysis with output image
    print("\n2. Detailed analysis with bounding boxes:")
    detector = EmotionDetector()
    detailed_results = detector.process_image(test_image, 'test_output.jpg')
    
    print(f"   Total faces: {detailed_results['faces_detected']}")
    
    for emotion_data in detailed_results['emotions']:
        bbox = emotion_data['bbox']
        print(f"   Face {emotion_data['face_id']}: {emotion_data['emotion']} "
              f"(confidence: {emotion_data['confidence']:.3f})")
        print(f"      Location: ({bbox[0]}, {bbox[1]}) Size: {bbox[2]}x{bbox[3]}")
    
    if 'output_saved' in detailed_results:
        print(f"\n   ✅ Annotated image saved: {detailed_results['output_saved']}")
    
    print("\n=== Static Image Test Complete ===")
    return True

def test_live_system():
    print("\n=== Testing Live Emotion Detection Setup ===\n")
    
    live_detector = LiveEmotionDetector()
    
    # Test camera initialization
    print("Testing webcam connection...")
    if live_detector.start_camera():
        print("✅ Webcam is working")
        live_detector.stop()
        
        print("\nTo test live detection, run:")
        print("  python run_live.py")
        return True
    else:
        print("❌ Webcam not available")
        return False

def test_system():
    print("=== Testing Complete System ===\n")
    
    # Test static image detection
    static_ok = test_static_system()
    
    # Test live system setup
    live_ok = test_live_system()
    
    print(f"\n=== Test Results ===")
    print(f"Static detection: {'✅ OK' if static_ok else '❌ Failed'}")
    print(f"Live detection: {'✅ OK' if live_ok else '❌ Failed'}")

if __name__ == "__main__":
    test_system()