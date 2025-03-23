from flask import Flask, request, render_template, jsonify, send_file
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
brain_model = models.resnet18(weights=None)
brain_model.fc = nn.Linear(brain_model.fc.in_features, 1)
brain_model.load_state_dict(torch.load('tumor_classification_resnet18.pth', map_location=device))
brain_model.to(device)
brain_model.eval()

class ImprovedPneumoniaModel(torch.nn.Module):
    def __init__(self):
        super(ImprovedPneumoniaModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(128 * 32 * 32, 512)
        self.fc2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

pneumonia_model = ImprovedPneumoniaModel()
pneumonia_model.load_state_dict(torch.load('chest_xray_model.pth', map_location=torch.device('cpu')))
pneumonia_model.eval()

def predict_brain_tumor(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = brain_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = brain_model(image_tensor)
        prediction = torch.sigmoid(output).item()
    return 'Tumor Detected' if prediction > 0.5 else 'No Tumor Found'

def predict_pneumonia(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = chest_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = pneumonia_model(image_tensor)
        probability = torch.sigmoid(output).item()
    return 'PNEUMONIA' if probability > 0.5 else 'NORMAL'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_brain', methods=['POST'])
def predict_brain():
    try:
        file = request.files['image']
        image_bytes = file.read()
        result = predict_brain_tumor(image_bytes)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_chest', methods=['POST'])
def predict_chest():
    try:
        file = request.files['image']
        image_bytes = file.read()
        result = predict_pneumonia(image_bytes)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
