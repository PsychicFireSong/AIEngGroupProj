# Quick Start

This repo trains a two-stage YOLO defect pipeline and serves it through a deployable web dashboard.

## 1. Set Up Python

From the repo root, activate your environment if needed:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## 2. Merge Datasets

```powershell
python scripts/merge_datasets.py --config configs/merge_config.yaml --preserve-splits
```

## 3. Train Models

Primary detector:

```powershell
yolo detect train model=yolo11s.pt data=merged_dataset/data.yaml epochs=100 imgsz=640
```

Severity classifier:

```powershell
python scripts/extract_severity_crops.py --config configs/merge_config.yaml
yolo classify train model=yolo11n-cls.pt data=severity_dataset epochs=50 imgsz=224
```

Copy or rename your trained checkpoints so the API can discover them, for example:

```powershell
New-Item -ItemType Directory -Force weights
Copy-Item runs\detect\stage1_defect_detector\weights\best.pt weights\defect_detector.pt
Copy-Item runs\classify\stage2_severity\weights\best.pt weights\severity_cls.pt
```

## 4. Run Inference API

```powershell
python -m uvicorn apps.inference_api:app --host 0.0.0.0 --port 8000 --reload
```

Check `http://localhost:8000/health`.

## 5. Run Deployable Dashboard

```powershell
cd apps/facility-dashboard
npm install
npm run dev
```

Open `http://localhost:3000`.

The dashboard is camera-first: visual evidence stays in the center, while live status, metrics, severity mix, defect list, selected finding, distribution, trend, and history palettes update around it.

## 6. Deployment Shape

Recommended final deployment:

- Deploy `apps/facility-dashboard` as the web frontend.
- Deploy `apps/inference_api.py` as a Python API service with access to the `.pt` model files and GPU if available.
- Set `NEXT_PUBLIC_INFERENCE_API_URL` in the frontend environment to the public API URL.

The Streamlit app is kept only as a legacy local demo.
