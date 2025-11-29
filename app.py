import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import base64
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import your risk model logic
try:
    from liver_cancer_risk import predict_user_input
except ImportError:
    def predict_user_input(*_args, **_kwargs):
        raise ImportError("liver_cancer_risk package not available. Ensure folder exists and is deployed with models.")


# ========================================================
# 1. DEFINE THE U-NET ARCHITECTURE
# ========================================================
class UNet(nn.Module):
    # Architecture kept EXACTLY as provided to ensure weight compatibility
    def __init__(self):
        super().__init__()
        def CBR(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(),
                                            nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU())
        self.e1 = CBR(1, 64); self.p = nn.MaxPool2d(2)
        self.e2 = CBR(64, 128); self.e3 = CBR(128, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2); self.d1 = CBR(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2); self.d2 = CBR(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.e1(x); x2 = self.e2(self.p(x1)); x3 = self.e3(self.p(x2))
        u1 = self.up1(x3); x4 = self.d1(torch.cat([u1, x2], 1))
        u2 = self.up2(x4); x5 = self.d2(torch.cat([u2, x1], 1))
        return self.out(x5)


# ========================================================
# 2. SETUP APP & MODEL
# ========================================================
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max upload limit

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cpu')
model = UNet()
model_loaded = False
try:
    # Load weights (Fixing keys for CPU)
    state_dict = torch.load('models/liver_unet.pth', map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    torch.set_grad_enabled(False)
    model_loaded = True
    logger.info('Segmentation model loaded successfully.')
except Exception as e:
    logger.error(f'Failed to load segmentation model: {e}')


# ========================================================
# 3. HELPER FUNCTIONS
# ========================================================
def process_image_for_model(image_bytes):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError('Invalid image data or unsupported format.')
    img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)

    return img_tensor, img_resized


# ========================================================
# 4. API ROUTES
# ========================================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model_loaded}), 200


@app.route('/api/segment', methods=['POST'])
def segment_liver():
    if not model_loaded:
        return jsonify({'status': 'error', 'message': 'Segmentation model not loaded.'}), 500
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded (form-data key "file").'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'Empty filename.'}), 400
        input_tensor, original_img = process_image_for_model(file.read())
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output)
            mask = (prob > 0.2).float().cpu().numpy().squeeze()
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        green_layer = np.zeros_like(original_rgb)
        green_layer[:, :, 1] = (mask * 255).astype(np.uint8)
        combined = (original_rgb.astype(np.float32) * 0.7) + (green_layer.astype(np.float32) * 0.3)
        result_img = np.clip(combined, 0, 255).astype(np.uint8)
        success, buffer = cv2.imencode('.png', result_img)
        if not success:
            raise RuntimeError('Failed to encode result image.')
        img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        return jsonify({'status': 'success', 'image_base64': img_base64}), 200
    except ValueError as ve:
        logger.warning(f'Segment validation error: {ve}')
        return jsonify({'status': 'error', 'message': str(ve)}), 400
    except Exception as e:
        logger.exception('Unexpected error during segmentation')
        return jsonify({'status': 'error', 'message': 'Internal server error.'}), 500


@app.route('/api/predict-risk', methods=['POST'])
def predict_risk():
    try:
        if not request.is_json:
            return jsonify({'status': 'error', 'message': 'Request must be JSON.'}), 400
        data = request.get_json() or {}
        if not isinstance(data, dict):
            return jsonify({'status': 'error', 'message': 'JSON body must be an object.'}), 400
        result = predict_user_input(data, model_preference='auto')
        return jsonify({'status': 'success', 'prediction': result}), 200
    except ValueError as ve:
        logger.warning(f'Predict validation error: {ve}')
        return jsonify({'status': 'error', 'message': str(ve)}), 400
    except FileNotFoundError as fe:
        logger.error(f'Model artifact missing: {fe}')
        return jsonify({'status': 'error', 'message': 'Model artifacts missing on server.'}), 500
    except ImportError as ie:
        logger.error(f'Predict import error: {ie}')
        return jsonify({'status': 'error', 'message': 'Risk prediction module unavailable.'}), 500
    except Exception as e:
        logger.exception('Unexpected error during risk prediction')
        return jsonify({'status': 'error', 'message': 'Internal server error.'}), 500


# ========================================================
# 5. ERROR HANDLERS
# ========================================================
@app.errorhandler(404)
def not_found(_):
    return jsonify({'status': 'error', 'message': 'Route not found.'}), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({'status': 'error', 'message': 'Method not allowed.'}), 405


@app.errorhandler(500)
def internal_error(_):
    return jsonify({'status': 'error', 'message': 'Internal server error.'}), 500


# ========================================================
# 6. RUN THE APP
# ========================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'
    logger.info(f'Starting server on {host}:{port}')
    app.run(host=host, port=port, debug=False)
