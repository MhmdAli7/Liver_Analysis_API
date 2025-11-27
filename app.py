import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS  # ADD THIS
from werkzeug.utils import secure_filename

# Import your risk model logic
try:
    from liver_cancer_risk import predict_user_input
except ImportError:
    print("⚠️ Warning: Could not import 'liver_cancer_risk'. Check folder structure.")


# ========================================================
# 1. DEFINE THE U-NET ARCHITECTURE
# ========================================================
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def CBR(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(),
                                            nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU())

        self.e1 = CBR(1, 64);
        self.p = nn.MaxPool2d(2)
        self.e2 = CBR(64, 128);
        self.e3 = CBR(128, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2);
        self.d1 = CBR(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2);
        self.d2 = CBR(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.e1(x);
        x2 = self.e2(self.p(x1));
        x3 = self.e3(self.p(x2))
        u1 = self.up1(x3);
        x4 = self.d1(torch.cat([u1, x2], 1))
        u2 = self.up2(x4);
        x5 = self.d2(torch.cat([u2, x1], 1))
        return self.out(x5)


# ========================================================
# 2. SETUP APP & MODEL
# ========================================================
app = Flask(__name__)
CORS(app)  # ADD THIS LINE - Enable CORS for all routes

device = torch.device('cpu')
model = UNet()

try:
    # Load weights (Fixing keys for CPU)
    state_dict = torch.load('models/liver_unet.pth', map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    print("✅ Segmentation Model Loaded!")
except Exception as e:
    print(f"❌ Error loading Segmentation Model: {e}")


# ========================================================
# 3. HELPER FUNCTIONS
# ========================================================
def process_image_for_model(image_bytes):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Resize & Normalize
    img_resized = cv2.resize(img, (256, 256))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0)

    return img_tensor, img_resized


# ========================================================
# 4. API ROUTES
# ========================================================

@app.route('/api/segment', methods=['POST'])
def segment_liver():
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        file = request.files['file']
        input_tensor, original_img = process_image_for_model(file.read())

        # 1. Run Inference
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output)
            mask = (prob > 0.3).float().numpy().squeeze()

        # 2. Create Visualization
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

        # Create Green Mask
        green_layer = np.zeros_like(original_rgb)
        green_layer[:, :, 1] = (mask * 255).astype(np.uint8)

        # Numpy blending
        combined = (original_rgb.astype(np.float32) * 0.7) + (green_layer.astype(np.float32) * 0.3)
        result_img = np.clip(combined, 0, 255).astype(np.uint8)

        # 3. Encode to Base64 String
        _, buffer = cv2.imencode('.png', result_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "status": "success",
            "image_base64": img_base64
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/predict-risk', methods=['POST'])
def predict_risk():
    try:
        data = request.get_json()
        # Call the function from your other project
        result = predict_user_input(data, model_preference='auto')
        return jsonify({"status": "success", "prediction": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)