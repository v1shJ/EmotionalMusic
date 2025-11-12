"""
Streamlined Face Detection + Emotion Recognition System
Combines face detection with your trained CNN emotion model.
"""
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EmotionCNN(nn.Module):
    """Your original CNN architecture for emotion detection"""
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

class EmotionDetector:
    """Main class for face detection + emotion recognition"""
    
    def __init__(self, model_path='../model.pt'):
        self.emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
        
        # Load your trained emotion model
        self.model = EmotionCNN().to(device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def detect_faces(self, image):
        """Detect faces using OpenCV"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces
    
    def predict_emotion(self, face_image):
        """Predict emotion using your CNN model"""
        face_tensor = self.transform(face_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = self.model(face_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        emotion = self.emotion_labels[predicted_class.item()]
        confidence_score = confidence.item()
        
        return emotion, confidence_score
    
    def process_image(self, image_path, save_output=None):
        """
        Main function: detect faces and predict emotions
        
        Args:
            image_path: Path to input image
            save_output: Path to save annotated image (optional)
            
        Returns:
            dict: Results with detected faces and emotions
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = self.detect_faces(np.array(image))
        
        results = {
            'faces_detected': len(faces),
            'emotions': []
        }
        
        # Process each face
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region
            face_region = image.crop((x, y, x + w, y + h))
            
            # Predict emotion
            emotion, confidence = self.predict_emotion(face_region)
            
            results['emotions'].append({
                'face_id': i + 1,
                'emotion': emotion,
                'confidence': confidence,
                'bbox': [x, y, w, h]
            })
            
            # Draw bounding box and label
            cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"Face {i+1}: {emotion} ({confidence:.2f})"
            cv2.putText(image_cv, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Save annotated image if requested
        if save_output:
            processed_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            Image.fromarray(processed_image).save(save_output)
            results['output_saved'] = save_output
        
        return results

# Simple function for backward compatibility
def predict_emotion(image_path, model_path='../model.pt'):
    """Simple function that returns primary emotion (for compatibility)"""
    detector = EmotionDetector(model_path)
    results = detector.process_image(image_path)
    
    if results['faces_detected'] == 0:
        # Fallback: analyze whole image
        image = Image.open(image_path).convert('RGB')
        emotion, confidence = detector.predict_emotion(image)
        return {'emotion': emotion, 'confidence': confidence, 'faces_detected': 0}
    
    # Return highest confidence emotion
    best_emotion = max(results['emotions'], key=lambda x: x['confidence'])
    return {
        'emotion': best_emotion['emotion'],
        'confidence': best_emotion['confidence'], 
        'faces_detected': results['faces_detected']
    }

if __name__ == "__main__":
    # Test the system
    detector = EmotionDetector()
    
    test_image = '../TestingImages/neutral.jpeg'
    if os.path.exists(test_image):
        print("Testing emotion detection with face detection...")
        results = detector.process_image(test_image, 'result.jpg')
        
        print(f"Faces detected: {results['faces_detected']}")
        for emotion_data in results['emotions']:
            print(f"Face {emotion_data['face_id']}: {emotion_data['emotion']} ({emotion_data['confidence']:.3f})")
    else:
        print(f"Test image not found: {test_image}")