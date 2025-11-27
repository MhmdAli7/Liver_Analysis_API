# Liver Analysis API

Production-ready Flask REST API serving:
- Liver segmentation via U-Net (PyTorch)
- Liver cancer risk prediction via pre-trained models & preprocessing pipeline

## Endpoints

1. `GET /health` — Basic health and model load status.
2. `POST /api/segment` — Form-data with key `file` containing an image. Returns JSON with `image_base64` overlay result.
3. `POST /api/predict-risk` — JSON body with patient data. Returns JSON prediction object.

## Deployment (Render)

1. Ensure this repo contains:
   - `requirements.txt` (present)
   - `.python-version` (pins Python 3.10.13 to guarantee wheel availability for torch, numpy, opencv)
2. Render settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
3. Models directory `models/` must include:
   - `liver_unet.pth`
   - `pipeline.pkl` and selected classifier artifacts (`random_forest.pkl`, etc.)

If you plan to enable the neural network classifier (`nn_classifier.h5`), uncomment TensorFlow in `requirements.txt` (may increase build size).

## Local Run

```cmd
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Then test:

```cmd
curl http://localhost:5000/health
```

Segmentation example (PowerShell):
```powershell
Invoke-RestMethod -Uri http://localhost:5000/api/segment -Method Post -Form @{file=Get-Item .\dummy_test.png}
```

Risk prediction example:
```powershell
$body = @{ age=65; gender='Male'; bmi=27; liver_function_score=85; alpha_fetoprotein_level=12; alcohol_consumption='Moderate'; smoking_status='Former'; physical_activity_level='Low'; hepatitis_b=0; hepatitis_c=0; cirrhosis_history=0; family_history_cancer=1; diabetes=0 } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:5000/api/predict-risk -Method Post -Body $body -ContentType 'application/json'
```

## Notes
- Base64 image output is PNG format.
- Threshold for segmentation mask is fixed at 0.3.
- All responses are JSON only.
- CORS is enabled for all routes by default.

## Troubleshooting
- If you see `ModuleNotFoundError: No module named 'numpy'` on Render, confirm `requirements.txt` is non-empty and the build command is correct.
- If PyTorch fails on Python 3.13, pin to 3.10.x or 3.11.x using `.python-version`.
- For missing model artifacts, ensure the `models/` directory is committed to the repository (not ignored by `.gitignore`).

## License
MIT (or specify your license here)

