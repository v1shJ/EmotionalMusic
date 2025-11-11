import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
from face_emotion_detector import FaceEmotionDetector
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)
        
        # After 3 pooling operations: 48 -> 24 -> 12 -> 6
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 1024)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 7)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = torch.relu(self.conv3(x))
        x = self.pool2(x)
        
        x = torch.relu(self.conv4(x))
        x = self.pool3(x)
        x = self.dropout2(x)
        
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x

emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Original predict_emotion function for backward compatibility
def predict_emotion_original(image_path, model_path='../model.pt'):
    """Original emotion prediction function without face detection"""
    loaded_model = EmotionCNN().to(device)
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    loaded_model.eval()
    image_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    from PIL import Image
    image = Image.open(image_path)
    image_tensor = image_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = loaded_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    predicted_emotion = emotion_labels[predicted_class.item()]
    confidence_score = confidence.item()
    all_probs = probabilities[0].cpu().numpy()
    all_predictions = {emotion_labels[i]: float(all_probs[i]) for i in range(len(emotion_labels))}
    
    return {
        'emotion': predicted_emotion,
        'confidence': confidence_score,
        'all_predictions': all_predictions
    }

# Enhanced predict_emotion function with face detection
def predict_emotion(image_path, model_path='../model.pt', use_face_detection=True, save_annotated=None):
    """
    Enhanced emotion prediction with optional face detection
    
    Args:
        image_path: Path to input image
        model_path: Path to emotion model
        use_face_detection: Whether to use face detection (default: True)
        save_annotated: Path to save annotated image with bounding boxes (optional)
        
    Returns:
        Dictionary with emotion prediction results
    """
    if use_face_detection:
        detector = FaceEmotionDetector(model_path)
        
        if save_annotated:
            # Process with face detection and save annotated image
            results = detector.process_image(image_path, save_annotated)
            primary_emotion = detector.get_primary_emotion(image_path)
            
            # Add face detection info to result
            primary_emotion['face_detection_results'] = {
                'faces_detected': results['faces_detected'],
                'all_faces': results['face_emotions'],
                'annotated_image_saved': save_annotated
            }
            return primary_emotion
        else:
            # Just get primary emotion with face detection info
            return detector.get_primary_emotion(image_path)
    else:
        # Use original method without face detection
        return predict_emotion_original(image_path, model_path)

# New function to get detailed face analysis
def analyze_faces_in_image(image_path, model_path='../model.pt', save_annotated=None):
    """
    Perform detailed face detection and emotion analysis
    
    Args:
        image_path: Path to input image
        model_path: Path to emotion model
        save_annotated: Path to save annotated image (optional)
        
    Returns:
        Detailed results for all detected faces
    """
    detector = FaceEmotionDetector(model_path)
    return detector.process_image(image_path, save_annotated)


if __name__ == "__main__":
    test_image_path = '../TestingImages/neutral.jpeg'  
    
    print("="*60)
    print("EMOTION DETECTION WITH FACE DETECTION")
    print("="*60)
    
    # Test enhanced emotion detection with face detection
    print("1. Testing with face detection enabled...")
    result_with_faces = predict_emotion(test_image_path, use_face_detection=True, 
                                       save_annotated='test_output_with_faces.jpg')
    print(f"Primary emotion: {result_with_faces['emotion']}")
    print(f"Confidence: {result_with_faces['confidence']:.3f}")
    print(f"Faces detected: {result_with_faces.get('faces_detected', 'N/A')}")
    
    print("\n" + "-"*40)
    
    # Test detailed face analysis
    print("2. Testing detailed face analysis...")
    detailed_results = analyze_faces_in_image(test_image_path, 
                                            save_annotated='detailed_face_analysis.jpg')
    
    print(f"Total faces detected: {detailed_results['faces_detected']}")
    for face_info in detailed_results['face_emotions']:
        bbox = face_info['bounding_box']
        emotion_pred = face_info['emotion_prediction']
        print(f"\nFace {face_info['face_id']}:")
        print(f"  Location: ({bbox['x']}, {bbox['y']}) - {bbox['width']}x{bbox['height']}")
        print(f"  Emotion: {emotion_pred['emotion']} ({emotion_pred['confidence']:.3f})")
    
    print("\n" + "-"*40)
    
    # Test original method for comparison
    print("3. Testing original method (without face detection)...")
    result_original = predict_emotion(test_image_path, use_face_detection=False)
    print(f"Original method result: {result_original['emotion']} ({result_original['confidence']:.3f})")
    
    print("\n" + "="*60)
    print("Analysis complete! Check the saved images:")
    print("- test_output_with_faces.jpg")  
    print("- detailed_face_analysis.jpg")