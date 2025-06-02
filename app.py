from flask import Flask, request, render_template, jsonify, send_file
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
from flask_cors import CORS
import numpy as np

app = Flask(__name__, static_folder='static')
CORS(app)

# Image preprocessing transforms
brain_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

chest_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Original Pneumonia Model (matching the saved weights)
class PneumoniaModel(nn.Module):
    def __init__(self):
        super(PneumoniaModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        # Original model had 131072 input features
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize models
try:
    # Initialize brain tumor model
    brain_model = models.resnet18(weights=None)
    brain_model.fc = nn.Linear(brain_model.fc.in_features, 1)
    brain_model.load_state_dict(torch.load('tumor_classification_resnet18.pth', map_location=device))
    brain_model.to(device)
    brain_model.eval()
    print("Successfully loaded brain tumor model")
except Exception as e:
    print(f"Error loading brain tumor model: {str(e)}")
    raise

try:
    # Initialize pneumonia model
    pneumonia_model = PneumoniaModel()
    pneumonia_model.load_state_dict(torch.load('chest_xray_model.pth', map_location=device))
    pneumonia_model.to(device)
    pneumonia_model.eval()
    print("Successfully loaded pneumonia model")
except Exception as e:
    print(f"Error loading pneumonia model: {str(e)}")
    raise

def preprocess_image(image_bytes, transform):
    """Image preprocessing with error handling"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return transform(image)
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def predict_brain_tumor(image_bytes):
    """Brain tumor prediction with confidence score"""
    try:
        image_tensor = preprocess_image(image_bytes, brain_transform).unsqueeze(0).to(device)
        with torch.no_grad():
            output = brain_model(image_tensor)
            probability = torch.sigmoid(output).item()
            confidence = max(probability, 1 - probability) * 100
            
            result = {
                'prediction': 'Tumor Detected' if probability > 0.5 else 'No Tumor Found',
                'confidence': f"{confidence:.2f}%",
                'probability': f"{probability:.3f}"
            }
            return result
    except Exception as e:
        raise ValueError(f"Error in brain tumor prediction: {str(e)}")

def predict_pneumonia(image_bytes):
    """Pneumonia prediction with confidence score"""
    try:
        image_tensor = preprocess_image(image_bytes, chest_transform).unsqueeze(0).to(device)
        with torch.no_grad():
            output = pneumonia_model(image_tensor)
            probability = torch.sigmoid(output).item()
            confidence = max(probability, 1 - probability) * 100
            
            result = {
                'prediction': 'PNEUMONIA' if probability > 0.5 else 'NORMAL',
                'confidence': f"{confidence:.2f}%",
                'probability': f"{probability:.3f}"
            }
            return result
    except Exception as e:
        raise ValueError(f"Error in pneumonia prediction: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_brain', methods=['POST'])
def predict_brain():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file format. Please upload PNG or JPG images.'}), 400
            
        image_bytes = file.read()
        result = predict_brain_tumor(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_chest', methods=['POST'])
def predict_chest():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file format. Please upload PNG or JPG images.'}), 400
            
        image_bytes = file.read()
        result = predict_pneumonia(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)