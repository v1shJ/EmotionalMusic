import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
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

def predict_emotion(image_path, model_path='../model.pt'):
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


if __name__ == "__main__":
    test_image_path = '../TestingImages/neutral.jpeg'  
    result = predict_emotion(test_image_path)
    print(result)